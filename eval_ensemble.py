import argparse
import os
import sys
import time
import random
import copy
import gc
import numpy as np
import cv2
from collections import defaultdict

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
import torch
import torch.nn.functional as F
import torchvision
from detection import utils
from detection.config import cfg as det_cfg
from detection.data.build import build_data_loaders
from detection.data.evaluations import voc_evaluation
from detection.modeling.build import build_detectors
from detection.utils.dist_utils import is_main_process, get_world_size


class SNormCaptureHook:
    def __init__(self):
        self.captured_input = None
        self.captured_output = None
        self.should_capture = False

    def __call__(self, module, inp, out):
        if self.should_capture:
            inp_t = inp[0] if isinstance(inp, tuple) else inp
            out_t = out[0] if isinstance(out, tuple) else out
            self.captured_input = inp_t.detach().float().cpu().clone()
            self.captured_output = out_t.detach().float().cpu().clone()


def ensemble_nms(all_boxes, all_scores, all_labels, iou_threshold=0.5, score_threshold=0.0):
    if len(all_boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    boxes = np.concatenate(all_boxes, axis=0) if len(all_boxes[0]) > 0 else np.zeros((0, 4))
    scores = np.concatenate(all_scores, axis=0) if len(all_scores[0]) > 0 else np.zeros(0)
    labels = np.concatenate(all_labels, axis=0) if len(all_labels[0]) > 0 else np.zeros(0)
    if len(boxes) == 0:
        return boxes, scores, labels
    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]
    if len(boxes) == 0:
        return boxes, scores, labels
    final_boxes, final_scores, final_labels = [], [], []
    unique_labels = np.unique(labels)
    for cls_id in unique_labels:
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        x1 = cls_boxes[:, 0]
        y1 = cls_boxes[:, 1]
        x2 = cls_boxes[:, 2]
        y2 = cls_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cls_scores.argsort()[::-1]
        kept = []
        while order.size > 0:
            i = order[0]
            kept.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        final_boxes.append(cls_boxes[kept])
        final_scores.append(cls_scores[kept])
        final_labels.append(np.full(len(kept), cls_id, dtype=np.int64))
    if len(final_boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    final_boxes = np.concatenate(final_boxes, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    final_labels = np.concatenate(final_labels, axis=0)
    return final_boxes, final_scores, final_labels


def _tensor_to_cv_img(tensor):
    t = tensor.detach().float().cpu().clamp(-3, 3)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim == 4:
        t = t[:1]
    img = (t * std + mean).clamp(0, 1)
    img = (img.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _cv_img_to_tensor(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def _draw_boxes(cv_img, boxes, scores=None, labels=None, color=(0, 0, 255), thickness=2, score_thr=0.3):
    out = cv_img.copy()
    if boxes is None or len(boxes) == 0:
        return out
    for i in range(len(boxes)):
        if scores is not None and scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label_str = ''
        if labels is not None:
            label_str = str(int(labels[i]))
        if scores is not None:
            label_str += ' %.2f' % scores[i]
        if label_str:
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(out, label_str, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_visual_comparison(vis_dir, dataset_name, img_idx, img_id, images_norm,
                           variant_outputs, variant_dets,
                           fused_boxes=None, fused_scores=None, fused_labels=None,
                           gt_boxes=None, gt_labels=None,
                           num_variants=3, moe_marks=None, seeds=None):
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, '%s_%03d_%s.png' % (dataset_name, img_idx, img_id))
    grid_list = []

    orig_cv = _tensor_to_cv_img(images_norm)
    if gt_boxes is not None and len(gt_boxes) > 0:
        orig_cv = _draw_boxes(orig_cv, gt_boxes, labels=gt_labels, color=(0, 255, 0), thickness=3)
    if fused_boxes is not None and len(fused_boxes) > 0:
        orig_cv = _draw_boxes(orig_cv, fused_boxes, scores=fused_scores,
                              labels=fused_labels, color=(0, 0, 255), thickness=2, score_thr=0.3)
    grid_list.append(_cv_img_to_tensor(orig_cv))

    for vi in range(num_variants):
        if variant_outputs[vi] is not None:
            out_cv = _tensor_to_cv_img(variant_outputs[vi])
            vboxes, vscores, vlabels = variant_dets[vi]
            if vboxes is not None and len(vboxes) > 0:
                out_cv = _draw_boxes(out_cv, vboxes, scores=vscores, labels=vlabels,
                                     color=(255, 165, 0), thickness=2, score_thr=0.3)
            grid_list.append(_cv_img_to_tensor(out_cv))
        else:
            grid_list.append(grid_list[0])

    big_grid = torchvision.utils.make_grid(grid_list, nrow=len(grid_list), padding=8)
    torchvision.utils.save_image(big_grid, vis_path)
    print('  [Vis] Saved: %s  (orig[GTgreen/fusedred] | %s)' % (
        vis_path, ' | '.join(['v%d(m%d)' % (vi, moe_marks[vi]) for vi in range(num_variants)])))
    return vis_path


@torch.no_grad()
def do_evaluation_ensemble(model, data_loader, device, output_dir,
                           num_variants=5, moe_marks=None, seeds=None,
                           iou_threshold=0.5, score_threshold=0.0,
                           dataset_name=None, save_vis=False, max_vis_per_dataset=5):
    if moe_marks is None:
        moe_marks = list(range(num_variants))
    if len(moe_marks) < num_variants:
        moe_marks = (moe_marks * ((num_variants // len(moe_marks)) + 1))[:num_variants]
    if seeds is None:
        seeds = list(range(42, 42 + num_variants))
    assert len(moe_marks) == num_variants
    assert len(seeds) == num_variants

    model.eval()
    dataset = data_loader.dataset
    dataset_name_str = dataset_name or dataset.dataset_name
    header = 'Ensemble Testing %s (%d variants):' % (dataset_name_str, num_variants)
    metric_logger = utils.MetricLogger(delimiter="  ")
    results_dict = {}
    variant_stats = {i: 0 for i in range(num_variants)}

    capture_hook = None
    hook_handle = None
    snorm_module = None
    if save_vis:
        capture_hook = SNormCaptureHook()
        try:
            snorm_module = model.teacher.proj_norm
        except AttributeError:
            try:
                snorm_module = model.module.teacher.proj_norm
            except Exception:
                snorm_module = None
        if snorm_module is not None:
            hook_handle = snorm_module.register_forward_hook(capture_hook)
            print('  [Vis] Registered hook on SNorm module: %s' % type(snorm_module).__name__)
        else:
            print('  [Vis] WARNING: Could not find SNorm module')

    vis_dir = os.path.join(output_dir, 'visualizations')
    vis_count = 0

    for images, img_metas, targets, _, _, _, _, _ in metric_logger.log_every(data_loader, 10, header):
        assert len(targets) == 1
        images = images.to(device)
        img_meta = img_metas[0]
        scale_factor = img_meta['scale_factor']
        img_info = img_meta['img_info']
        img_id = img_info.get('id', 'img_%d' % vis_count)

        all_boxes = []
        all_scores = []
        all_labels = []
        variant_outputs = [None] * num_variants
        variant_dets = []

        for vi in range(num_variants):
            random.seed(seeds[vi])
            np.random.seed(seeds[vi])
            torch.manual_seed(seeds[vi])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds[vi])

            if save_vis and capture_hook is not None:
                capture_hook.captured_input = None
                capture_hook.captured_output = None
                capture_hook.should_capture = True

            det = model(images, img_metas, targets, moe_mode=7, moe_mark=moe_marks[vi])[0]

            if save_vis and capture_hook is not None:
                capture_hook.should_capture = False
                if capture_hook.captured_output is not None:
                    variant_outputs[vi] = capture_hook.captured_output

            boxes = det['boxes'].float().cpu().numpy()
            scores = det['scores'].float().cpu().numpy()
            labels = det['labels'].cpu().numpy()
            variant_stats[vi] += len(boxes)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            variant_dets.append((boxes, scores, labels))

            del det, boxes, scores, labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fused_boxes, fused_scores, fused_labels = ensemble_nms(
            all_boxes, all_scores, all_labels,
            iou_threshold=iou_threshold, score_threshold=score_threshold
        )

        if save_vis and vis_count < max_vis_per_dataset and is_main_process():
            gt_boxes = None
            gt_labels = None
            if 'boxes' in targets[0]:
                gt_boxes = targets[0]['boxes'].cpu().numpy()
                gt_labels = targets[0]['labels'].cpu().numpy()
            save_visual_comparison(
                vis_dir, dataset_name_str, vis_count, str(img_id),
                images.detach().cpu(),
                variant_outputs, variant_dets,
                fused_boxes=fused_boxes, fused_scores=fused_scores, fused_labels=fused_labels,
                gt_boxes=gt_boxes, gt_labels=gt_labels,
                num_variants=num_variants, moe_marks=moe_marks, seeds=seeds,
            )
            vis_count += 1

        fused_boxes_orig = fused_boxes / scale_factor if len(fused_boxes) > 0 else fused_boxes
        result = {}
        result['boxes'] = fused_boxes_orig.tolist()
        result['scores'] = fused_scores.tolist()
        if hasattr(dataset, 'label2cat') and len(fused_labels) > 0:
            result['labels'] = [dataset.label2cat.get(int(l), int(l)) for l in fused_labels]
        else:
            result['labels'] = fused_labels.tolist() if hasattr(fused_labels, 'tolist') else list(fused_labels)
        results_dict[img_info['id']] = result

    if hook_handle is not None:
        hook_handle.remove()

    if get_world_size() > 1:
        torch.distributed.barrier()

    if is_main_process():
        print('\n--- Ensemble Variant Stats ---')
        for vi in range(num_variants):
            print('  variant[%d] moe_mark=%s seed=%s: total detections=%d' % (
                vi, moe_marks[vi], seeds[vi], variant_stats[vi]))
        print('  Fused with IoU NMS=%.2f score_thr=%.2f' % (iou_threshold, score_threshold))
        if save_vis:
            print('  Visualizations saved to: %s' % vis_dir)

    if get_world_size() > 1:
        from detection.utils.dist_utils import all_gather
        all_predictions = all_gather(results_dict)
        if not is_main_process():
            return {}
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
    else:
        predictions = results_dict

    if is_main_process() and len(predictions) > 0:
        try:
            voc_result = voc_evaluation(dataset, predictions, output_dir, iteration=None, use_07_metric=False)
            print('\n--- VOC Ensemble Results (%s) ---' % dataset_name_str)
            for k, v in voc_result.items():
                print('  %s: %.4f' % (k, v))
        except Exception as e:
            print('VOC evaluation error: %s' % e)
            voc_result = {}
        return voc_result

    return {}


def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference for Cross-Domain Detection')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-variants', type=int, default=3)
    parser.add_argument('--moe-marks', type=str, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument('--max-vis-per-dataset', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print('=' * 70)
    print(' Ensemble Inference for Cross-Domain Detection')
    print('=' * 70)

    if not os.path.isfile(args.config_file):
        print('Config file not found: %s' % args.config_file)
        return
    det_cfg.merge_from_file(args.config_file)
    det_cfg.freeze()

    print('  Config:       %s' % args.config_file)
    print('  Checkpoint:   %s' % args.resume)
    print('  GPU:          %s' % args.gpu)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('  GPU Device:   %s' % torch.cuda.get_device_name(0))

    num_variants = args.num_variants
    moe_marks = [int(x) for x in args.moe_marks.split(',')] if args.moe_marks else list(range(num_variants))
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else list(range(42, 42 + num_variants))
    if len(moe_marks) < num_variants:
        moe_marks = (moe_marks * ((num_variants // len(moe_marks)) + 1))[:num_variants]
    if len(seeds) < num_variants:
        seeds = (seeds * ((num_variants // len(seeds)) + 1))[:num_variants]

    print('  Mode:         ensemble (%d variants)' % num_variants)

    model = build_detectors(det_cfg)
    print('  Loading checkpoint from %s ...' % args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('  Missing keys: %d, Unexpected keys: %d' % (len(missing), len(unexpected)))
    print('  Checkpoint loaded successfully')
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print('  Model parameters: %.1fM' % (total_params / 1e6))

    print('\n  Ensemble config:')
    for vi in range(num_variants):
        print('    variant[%d]: moe_mode=7, moe_mark=%s, seed=%s' % (vi, moe_marks[vi], seeds[vi]))
    print('  Fusion: NMS IoU=%.2f, score_thr=%.2f' % (args.iou_threshold, args.score_threshold))

    output_dir = args.output_dir or os.path.dirname(args.resume)
    print('  Output dir:   %s' % output_dir)

    test_loaders = build_data_loaders(det_cfg.DATASETS.TESTS, transforms=det_cfg.INPUT.TRANSFORMS_TEST,
                                      is_train=False, distributed=False, batch_size=1, num_workers=0)
    if not isinstance(test_loaders, (list, tuple)):
        test_loaders = (test_loaders,)

    all_results = {}
    for data_loader in test_loaders:
        dataset_name = data_loader.dataset.dataset_name
        print('\n' + '-' * 60)
        print(' Ensemble (%d variants) on %s' % (num_variants, dataset_name))
        print('-' * 60)
        result = do_evaluation_ensemble(
            model, data_loader, device, output_dir,
            num_variants=num_variants,
            moe_marks=moe_marks,
            seeds=seeds,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            dataset_name=dataset_name,
            save_vis=args.save_vis,
            max_vis_per_dataset=args.max_vis_per_dataset,
        )
        all_results[dataset_name] = result

    print('\n' + '=' * 70)
    print(' All results summary:')
    for ds_name, res in all_results.items():
        print('  %s:' % ds_name)
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, (int, float)):
                    print('    %s: %.4f' % (k, v))
    print('=' * 70)


if __name__ == '__main__':
    main()