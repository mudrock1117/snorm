import os
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import pycocotools.mask as mask_util

from detection import utils
from detection.data.evaluations import coco_evaluation, voc_evaluation
from detection.data.transforms import de_normalize
from detection.layers.mask_ops import paste_masks_in_image
from detection.utils import colormap
from detection.utils.dist_utils import is_main_process, all_gather, get_world_size
from detection.utils.visualizer import Visualizer


def _compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_predictions_to_gt(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    if num_pred == 0 or num_gt == 0:
        return [], [], [], list(range(num_gt)), list(range(num_pred))

    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = _compute_iou(pred_boxes[i], gt_boxes[j])

    matched_gt = set()
    matched_pred = set()
    matches = []

    order = np.argsort(-np.array(pred_scores))
    for pred_idx in order:
        best_iou = 0
        best_gt = -1
        for gt_idx in range(num_gt):
            if gt_idx in matched_gt:
                continue
            if iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt = gt_idx
        if best_iou >= iou_thresh:
            matched_gt.add(best_gt)
            matched_pred.add(pred_idx)
            matches.append((pred_idx, best_gt, best_iou))

    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_pred = [i for i in range(num_pred) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred, unmatched_gt, unmatched_pred


def save_error_analysis(image, img_meta, pred_boxes, pred_labels, pred_scores,
                        gt_boxes, gt_labels, dataset, output_dir, file_stem, threshold=0.5):
    if not is_main_process():
        return

    error_dir = os.path.join(output_dir, 'error_analysis')
    os.makedirs(error_dir, exist_ok=True)

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return

    pred_boxes_list = pred_boxes.tolist() if hasattr(pred_boxes, 'tolist') else list(pred_boxes)
    pred_labels_list = pred_labels.tolist() if hasattr(pred_labels, 'tolist') else list(pred_labels)
    pred_scores_list = pred_scores.tolist() if hasattr(pred_scores, 'tolist') else list(pred_scores)
    gt_boxes_list = gt_boxes.tolist() if hasattr(gt_boxes, 'tolist') else list(gt_boxes)
    gt_labels_list = gt_labels.tolist() if hasattr(gt_labels, 'tolist') else list(gt_labels)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    image_np = de_normalize(image, img_meta)
    ax.imshow(image_np)

    if len(gt_boxes_list) > 0:
        for i, (x1, y1, x2, y2) in enumerate(gt_boxes_list):
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            gt_cls_id = gt_labels_list[i]
            if hasattr(dataset, 'label2cat'):
                gt_cls_id = dataset.label2cat.get(gt_cls_id, gt_cls_id)
            gt_cls = dataset.CLASSES[gt_cls_id] if hasattr(dataset, 'CLASSES') and 0 <= gt_cls_id < len(dataset.CLASSES) else str(gt_labels_list[i])
            ax.text(x1, max(0, y1 - 5), 'GT:{}'.format(gt_cls),
                    color='lime', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    if len(pred_boxes_list) > 0:
        for i, (x1, y1, x2, y2) in enumerate(pred_boxes_list):
            score = pred_scores_list[i] if i < len(pred_scores_list) else 0
            if score < threshold:
                continue
            pred_cls_id = pred_labels_list[i] if i < len(pred_labels_list) else -1
            if hasattr(dataset, 'label2cat'):
                pred_cls_id = dataset.label2cat.get(pred_cls_id, pred_cls_id)
            pred_cls = dataset.CLASSES[pred_cls_id] if hasattr(dataset, 'CLASSES') and 0 <= pred_cls_id < len(dataset.CLASSES) else str(pred_cls_id)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, min(image.shape[-2] - 5, y2 + 15), 'Pred:{}:{:.2f}'.format(pred_cls, score),
                    color='red', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    matches, unmatched_gt, unmatched_pred, _, _ = _match_predictions_to_gt(
        pred_boxes_list, pred_labels_list, pred_scores_list,
        gt_boxes_list, gt_labels_list, iou_thresh=0.5
    )

    error_info = []

    for pred_idx, gt_idx, iou in matches:
        pred_cls_id = pred_labels_list[pred_idx]
        gt_cls_id = gt_labels_list[gt_idx]
        if hasattr(dataset, 'label2cat'):
            pred_cls_id = dataset.label2cat.get(pred_cls_id, pred_cls_id)
            gt_cls_id = dataset.label2cat.get(gt_cls_id, gt_cls_id)
        if pred_cls_id != gt_cls_id:
            pred_cls_name = dataset.CLASSES[pred_cls_id] if hasattr(dataset, 'CLASSES') and 0 <= pred_cls_id < len(dataset.CLASSES) else str(pred_cls_id)
            gt_cls_name = dataset.CLASSES[gt_cls_id] if hasattr(dataset, 'CLASSES') and 0 <= gt_cls_id < len(dataset.CLASSES) else str(gt_cls_id)
            error_info.append('  [ClsErr] Pred={} GT={} IoU={:.2f} Score={:.2f}'.format(
                pred_cls_name, gt_cls_name, iou, pred_scores_list[pred_idx]))
            x1, y1, x2, y2 = pred_boxes_list[pred_idx]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor='magenta', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, 'ERR', color='magenta', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    for gt_idx in unmatched_gt:
        gt_cls_id = gt_labels_list[gt_idx]
        if hasattr(dataset, 'label2cat'):
            gt_cls_id = dataset.label2cat.get(gt_cls_id, gt_cls_id)
        gt_cls_name = dataset.CLASSES[gt_cls_id] if hasattr(dataset, 'CLASSES') and 0 <= gt_cls_id < len(dataset.CLASSES) else str(gt_labels_list[gt_idx])
        error_info.append('  [Miss] GT={}'.format(gt_cls_name))
        x1, y1, x2, y2 = gt_boxes_list[gt_idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 20), 'MISS', color='yellow', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))

    for pred_idx in unmatched_pred:
        if pred_scores_list[pred_idx] < threshold:
            continue
        pred_cls_id = pred_labels_list[pred_idx]
        if hasattr(dataset, 'label2cat'):
            pred_cls_id = dataset.label2cat.get(pred_cls_id, pred_cls_id)
        pred_cls_name = dataset.CLASSES[pred_cls_id] if hasattr(dataset, 'CLASSES') and 0 <= pred_cls_id < len(dataset.CLASSES) else str(pred_cls_id)
        error_info.append('  [FP] Pred={} Score={:.2f}'.format(pred_cls_name, pred_scores_list[pred_idx]))
        x1, y1, x2, y2 = pred_boxes_list[pred_idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor='cyan', facecolor='none', linestyle=':')
        ax.add_patch(rect)
        ax.text(x1, min(image.shape[-2] - 20, y2 + 30), 'FP', color='cyan', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))

    n_errors = len(error_info)
    ax.set_title('Error Analysis ({} errors)\n'.format(n_errors) +
                 '\n'.join(error_info[:5]) +
                 ('\n...and {} more'.format(n_errors - 5) if n_errors > 5 else ''),
                 fontsize=9, loc='left')
    ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(error_dir, '{}_errors.png'.format(file_stem))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions


def evaluation(model, data_loaders, device, types=('coco',), output_dir='./evaluations/', iteration=None, viz=False):
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = (data_loaders,)
    for root, dirs, files in os.walk(output_dir):
        for d in dirs:
            if d == 'error_analysis':
                import shutil
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    results = dict()
    for data_loader in data_loaders:
        dataset = data_loader.dataset
        _output_dir = os.path.join(output_dir, 'evaluations', dataset.dataset_name)
        os.makedirs(_output_dir, exist_ok=True)
        result = do_evaluation(model, data_loader, device, types=types, output_dir=_output_dir, iteration=iteration, viz=viz)
        results[dataset.dataset_name] = result
    return results


COLORMAP = colormap.colormap(rgb=True, maximum=1)


def save_visualization(dataset, img_meta, result, output_dir, threshold=0.8, fmt='.pdf'):
    save_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    file_name = img_meta['img_info']['file_name']
    img = Image.open(os.path.join(dataset.images_dir, file_name))
    w, h = img.size
    scale = 1.0
    # w, h = int(w * scale), int(h * scale)
    # img = img.resize((w, h))

    vis = Visualizer(img, metadata=None)

    boxes = np.array(result['boxes']) * scale
    labels = np.array(result['labels'])
    scores = np.array(result['scores'])
    #masks = result['masks']
    indices = scores > threshold
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    #masks = [m for m, b in zip(masks, indices) if b]

    # colors = COLORMAP[(labels + 2) % len(COLORMAP)]

    colors = [np.array([1.0, 0, 0])] * len(labels)

    labels = ['{}:{:.0f}%'.format(dataset.CLASSES[label], score * 100) for label, score in zip(labels, scores)]
    out = vis.overlay_instances(
        boxes=boxes,
        labels=labels,
        masks=None,
        assigned_colors=colors,
        alpha=0.8,
    )
    out.save(os.path.join(save_dir, os.path.basename(file_name).replace('.', '_') + fmt))


@torch.no_grad()
def do_evaluation(model, data_loader, device, types, output_dir, iteration=None, viz=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    dataset = data_loader.dataset
    header = 'Testing {}:'.format(dataset.dataset_name)
    results_dict = {}
    has_mask = False
    error_counts = {'saved': 0}
    MAX_ERROR_IMAGES = 5
    
    for images, img_metas, targets, _, _, _, _, _ in metric_logger.log_every(data_loader, 10, header):
        assert len(targets) == 1
        images = images.to(device)
        model_time = time.time()
        
        det = model(images, img_metas, targets, moe_mode=7)[0]
            
        boxes, scores, labels = det['boxes'], det['scores'], det['labels']

        model_time = time.time() - model_time

        img_meta = img_metas[0]
        scale_factor = img_meta['scale_factor']
        img_info = img_meta['img_info']
        file_name = img_info.get('file_name', 'unknown')
        file_stem = os.path.splitext(os.path.basename(file_name))[0] if file_name else str(img_info.get('id', 'unknown'))

        gt_boxes_raw = targets[0]['boxes']
        gt_labels_raw = targets[0]['labels']

        pred_boxes_raw = boxes.clone()
        pred_labels_raw = labels.clone()

        if is_main_process():
            pred_boxes_np = pred_boxes_raw.cpu().numpy()
            pred_labels_np = pred_labels_raw.cpu().numpy()
            pred_scores_np = scores.cpu().numpy()
            gt_boxes_np = gt_boxes_raw.cpu().numpy() if isinstance(gt_boxes_raw, torch.Tensor) else np.array(gt_boxes_raw)
            gt_labels_np = gt_labels_raw.cpu().numpy() if isinstance(gt_labels_raw, torch.Tensor) else np.array(gt_labels_raw)

            matches, unmatched_gt, unmatched_pred, _, _ = _match_predictions_to_gt(
                pred_boxes_np, pred_labels_np, pred_scores_np,
                gt_boxes_np, gt_labels_np, iou_thresh=0.5
            )

            _label2cat = getattr(dataset, 'label2cat', None)
            has_cls_error = False
            for m in matches:
                pred_cls = pred_labels_np[m[0]]
                gt_cls = gt_labels_np[m[1]]
                if _label2cat is not None:
                    pred_cls = _label2cat.get(pred_cls, pred_cls)
                    gt_cls = _label2cat.get(gt_cls, gt_cls)
                if pred_cls != gt_cls:
                    has_cls_error = True
                    break
            has_miss = len(unmatched_gt) > 0
            fp_indices = [p for p in unmatched_pred if pred_scores_np[p] >= 0.5]
            has_fp = len(fp_indices) > 0

            if (has_cls_error or has_miss or has_fp) and error_counts['saved'] < MAX_ERROR_IMAGES:
                save_error_analysis(
                    images[0].cpu(), img_meta,
                    pred_boxes_np, pred_labels_np, pred_scores_np,
                    gt_boxes_np, gt_labels_np,
                    dataset, output_dir, file_stem, threshold=0.5
                )
                error_counts['saved'] += 1

        if viz:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            plt.switch_backend('TKAgg')
            image = de_normalize(images[0], img_meta)
            plt.subplot(122)
            plt.imshow(image)
            plt.title('Predict')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes.tolist(), labels.tolist())):
                if scores[i] > 0.65:
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                    category_id = dataset.label2cat[label]
                    plt.text(x1, y1, '{}:{:.2f}'.format(dataset.CLASSES[category_id], scores[i]), color='r')
                    plt.gca().add_patch(rect)

            plt.subplot(121)
            plt.imshow(image)
            plt.title('GT')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(targets[0]['boxes'].tolist(), targets[0]['labels'].tolist())):
                category_id = dataset.label2cat[label]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                plt.text(x1, y1, '{}'.format(dataset.CLASSES[category_id]))
                plt.gca().add_patch(rect)
            plt.show()

        boxes /= scale_factor
        result = {}

        if 'masks' in det:
            has_mask = True
            (w, h) = img_meta['origin_img_shape']
            masks = paste_masks_in_image(det['masks'], boxes, (h, w))
            rles = []
            for mask in masks.cpu().numpy():
                mask = mask >= 0.5
                mask = mask_util.encode(np.array(mask[0][:, :, None], order='F', dtype='uint8'))[0]
                mask['counts'] = mask['counts'].decode('utf-8')
                rles.append(mask)
            result['masks'] = rles

        boxes = boxes.tolist()
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        if hasattr(dataset, 'label2cat'):
            labels = [dataset.label2cat[label.item() if hasattr(label, 'item') else label] for label in labels]
        scores = scores.tolist()

        result['boxes'] = boxes
        result['scores'] = scores
        result['labels'] = labels

        results_dict.update({
            img_info['id']: result
        })
        metric_logger.update(model_time=model_time)

    if get_world_size() > 1:
        dist.barrier()

    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    if not is_main_process():
        return {}
    if error_counts['saved'] > 0:
        print('Error images saved: {} -> {}'.format(error_counts['saved'], os.path.join(output_dir, 'error_analysis')))
    results = {}
    if has_mask:
        result = coco_evaluation(dataset, predictions, output_dir, iteration=iteration)
        results.update(result)
    if 'voc' in types:
        result = voc_evaluation(dataset, predictions, output_dir, iteration=iteration, use_07_metric=False)
        results.update(result)
    return results