#!/usr/bin/env python3
"""
Visualize detection errors with SNorm/AHDR-rendered images as background.
Saves up to MAX_PER_DOMAIN images per test domain.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from yacs.config import CfgNode
from detection.modeling.build import build_detectors
from detection.data.dataloaders import make_data_loader
from detection.data.transforms import de_normalize


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


def _match_pred_gt(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    if num_pred == 0 or num_gt == 0:
        return [], list(range(num_gt)), list(range(num_pred))

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
    return matches, unmatched_gt, unmatched_pred


def _tensor_to_rgb(tensor_chw, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    arr = tensor_chw.detach().cpu().numpy()
    arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(std) + np.array(mean)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/config.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/best.pth')
    parser.add_argument('--output', type=str,
                        default='debug/city_2_foggy/s0_dbias_color_fusion_w_elNet/evaluations')
    parser.add_argument('--max-per-domain', type=int, default=5)
    parser.add_argument('--score-thresh', type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[1/4] Loading config from {args.config}")
    with open(args.config, 'r') as f:
        cfg = CfgNode(new_allowed=True)
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.config)
    cfg.freeze()

    print(f"[2/4] Building model and loading checkpoint {args.checkpoint}")
    model = build_detectors(cfg)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"    Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    (first 5) {missing[:5]}")
    model.to(device)
    model.eval()

    hook_storage = {'snorm_rendered': None}

    def _snorm_hook(module, input, output):
        if isinstance(output, tuple):
            hook_storage['snorm_rendered'] = output[0].detach().cpu()
        else:
            hook_storage['snorm_rendered'] = output.detach().cpu()

    hook_handles = []
    for name, module in model.named_modules():
        if 'proj_norm' in name and hasattr(module, 'render_net'):
            h = module.register_forward_hook(_snorm_hook)
            hook_handles.append(h)
            print(f"    Hooked SNorm at: {name}")
    if not hook_handles:
        print("    WARNING: no SNorm module with render_net found!")

    print(f"[3/4] Building test dataloaders")
    test_names = list(cfg.DATASETS.TESTS)
    print(f"    Test domains: {test_names}")

    print(f"[4/4] Running evaluation and saving rendered error images (max {args.max_per_domain}/domain)")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    for test_name in test_names:
        print(f"\n--- Domain: {test_name} ---")
        loader = make_data_loader(cfg, split='test', dataset_name=test_name, is_distributed=False)
        dataset = loader.dataset
        domain_out_dir = os.path.join(args.output, dataset.dataset_name, 'rendered_error_analysis')
        os.makedirs(domain_out_dir, exist_ok=True)

        saved = {'cls_err': 0, 'miss': 0, 'fp': 0}

        for images, img_metas, targets, _, _, _, _, _ in loader:
            file_name = img_metas[0]['img_info'].get('file_name', 'unknown')
            file_stem = os.path.splitext(os.path.basename(file_name))[0]
            images = images.to(device)
            hook_storage['snorm_rendered'] = None

            with torch.no_grad():
                det = model(images, img_metas, targets, moe_mode=7)[0]

            if hook_storage['snorm_rendered'] is None:
                continue

            rendered_chw = hook_storage['snorm_rendered'][0]
            rendered_rgb = _tensor_to_rgb(rendered_chw)

            boxes = det['boxes'].cpu().numpy()
            scores = det['scores'].cpu().numpy()
            labels = det['labels'].cpu().numpy()
            scale_factor = img_metas[0]['scale_factor']

            gt_boxes = targets[0]['boxes'].numpy()
            gt_labels = targets[0]['labels'].numpy()

            scale_factor = scale_factor if isinstance(scale_factor, float) else float(scale_factor[0])
            boxes_orig = boxes / scale_factor
            gt_boxes_orig = gt_boxes / scale_factor

            matches, unmatched_gt, unmatched_pred = _match_pred_gt(
                boxes_orig, labels, scores, gt_boxes_orig, gt_labels, iou_thresh=0.5
            )

            _label2cat = getattr(dataset, 'label2cat', None)
            def _to_cat(lbl_id):
                if _label2cat is not None:
                    lbl_id = _label2cat.get(lbl_id, lbl_id)
                return lbl_id

            has_cls_err = any(
                _to_cat(labels[mi]) != _to_cat(gt_labels[gi]) for mi, gi, _ in matches
            )
            has_miss = len(unmatched_gt) > 0
            fp_idx = [p for p in unmatched_pred if scores[p] >= args.score_thresh]
            has_fp = len(fp_idx) > 0

            need_save = (
                (has_cls_err and saved['cls_err'] < args.max_per_domain) or
                (has_miss and saved['miss'] < args.max_per_domain) or
                (has_fp and saved['fp'] < args.max_per_domain)
            )
            if not need_save:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            ax.imshow(rendered_rgb)

            for i, (x1, y1, x2, y2) in enumerate(gt_boxes_orig):
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                gt_cls_id = _to_cat(int(gt_labels[i]))
                gt_cls = dataset.CLASSES[gt_cls_id] if 0 <= gt_cls_id < len(dataset.CLASSES) else str(gt_cls_id)
                ax.text(x1, max(0, y1 - 5), f'GT:{gt_cls}',
                        color='lime', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

            for i, (x1, y1, x2, y2) in enumerate(boxes_orig):
                s = scores[i]
                if s < args.score_thresh:
                    continue
                p_cls_id = _to_cat(int(labels[i]))
                p_cls = dataset.CLASSES[p_cls_id] if 0 <= p_cls_id < len(dataset.CLASSES) else str(p_cls_id)
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, min(rendered_rgb.shape[0] - 5, y2 + 15), f'Pred:{p_cls}:{s:.2f}',
                        color='red', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

            err_lines = []
            for pred_idx, gt_idx, iou in matches:
                p_cls = _to_cat(int(labels[pred_idx]))
                g_cls = _to_cat(int(gt_labels[gt_idx]))
                if p_cls != g_cls:
                    err_lines.append(f"[ClsErr] Pred={dataset.CLASSES[p_cls]} GT={dataset.CLASSES[g_cls]} IoU={iou:.2f} Score={scores[pred_idx]:.2f}")
                    x1, y1, x2, y2 = boxes_orig[pred_idx]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=3, edgecolor='magenta', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1, 'ERR', color='magenta', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

            for gt_idx in unmatched_gt:
                g_cls = _to_cat(int(gt_labels[gt_idx]))
                err_lines.append(f"[Miss] GT={dataset.CLASSES[g_cls]}")
                x1, y1, x2, y2 = gt_boxes_orig[gt_idx]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=3, edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, max(0, y1 - 20), 'MISS', color='yellow', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))

            for pred_idx in unmatched_pred:
                if scores[pred_idx] < args.score_thresh:
                    continue
                p_cls = _to_cat(int(labels[pred_idx]))
                err_lines.append(f"[FP] Pred={dataset.CLASSES[p_cls]} Score={scores[pred_idx]:.2f}")
                x1, y1, x2, y2 = boxes_orig[pred_idx]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=3, edgecolor='cyan', facecolor='none', linestyle=':')
                ax.add_patch(rect)
                ax.text(x1, min(rendered_rgb.shape[0] - 20, y2 + 30), 'FP', color='cyan',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))

            title = f"{file_stem}  (SNorm+AHDR rendered)\n" + "\n".join(err_lines[:6])
            if len(err_lines) > 6:
                title += f"\n...and {len(err_lines) - 6} more"
            ax.set_title(title, fontsize=9, loc='left')
            ax.axis('off')
            plt.tight_layout()

            save_path = os.path.join(domain_out_dir, f'{file_stem}_rendered_errors.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            if has_cls_err:
                saved['cls_err'] += 1
            if has_miss:
                saved['miss'] += 1
            if has_fp:
                saved['fp'] += 1
            print(f"    Saved [{saved['cls_err']}/{saved['miss']}/{saved['fp']}] {file_stem} -> {save_path}")

            done = (
                saved['cls_err'] >= args.max_per_domain and
                saved['miss'] >= args.max_per_domain and
                saved['fp'] >= args.max_per_domain
            )
            if done:
                print(f"    Reached max {args.max_per_domain} per error type, stopping domain.")
                break

    print(f"\nDone. Output dir: {args.output}")


if __name__ == '__main__':
    main()