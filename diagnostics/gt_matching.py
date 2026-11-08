"""Class-aware, single-sample GT matching for SNorm step sweeps."""
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _as_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def box_iou(box_a, box_b):
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def match_step(detections, gt_boxes, gt_labels, iou_threshold, score_threshold):
    pred_boxes = detections.get("boxes", [])
    pred_scores = detections.get("scores", [])
    pred_labels = detections.get("labels", [])
    if not (len(pred_boxes) == len(pred_scores) == len(pred_labels)):
        raise ValueError("detection boxes/scores/labels lengths differ")

    eligible = [
        index for index, score in enumerate(pred_scores)
        if float(score) >= float(score_threshold)
    ]
    eligible.sort(key=lambda index: float(pred_scores[index]), reverse=True)
    unmatched_gt = set(range(len(gt_boxes)))
    predictions = []
    gt_matches = {}

    for pred_index in eligible:
        pred_label = int(pred_labels[pred_index])
        best_gt = None
        best_iou = 0.0
        for gt_index in sorted(unmatched_gt):
            if int(gt_labels[gt_index]) != pred_label:
                continue
            iou = box_iou(pred_boxes[pred_index], gt_boxes[gt_index])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        matched = best_gt is not None and best_iou >= float(iou_threshold)
        if matched:
            unmatched_gt.remove(best_gt)
            gt_matches[best_gt] = {
                "prediction_index": pred_index,
                "score": float(pred_scores[pred_index]),
                "iou": float(best_iou),
                "box": pred_boxes[pred_index],
            }
        predictions.append(
            {
                "prediction_index": pred_index,
                "label": pred_label,
                "score": float(pred_scores[pred_index]),
                "box": pred_boxes[pred_index],
                "matched": bool(matched),
                "gt_index": int(best_gt) if matched else None,
                "iou": float(best_iou),
            }
        )

    tp = len(gt_matches)
    fp = len(predictions) - tp
    fn = len(unmatched_gt)
    labels_present = sorted(
        set(int(value) for value in gt_labels)
        | set(int(item["label"]) for item in predictions)
    )
    per_class = {}
    for label in labels_present:
        class_predictions = [
            item for item in predictions if int(item["label"]) == label
        ]
        class_gt = [
            index for index, value in enumerate(gt_labels) if int(value) == label
        ]
        class_tp = sum(1 for item in class_predictions if item["matched"])
        per_class[str(label)] = {
            "tp": class_tp,
            "fp": len(class_predictions) - class_tp,
            "fn": sum(1 for index in class_gt if index in unmatched_gt),
            "gt_count": len(class_gt),
            "eligible_prediction_count": len(class_predictions),
        }
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_class": per_class,
        "eligible_prediction_count": len(predictions),
        "predictions": predictions,
        "gt_matches": {str(key): value for key, value in gt_matches.items()},
        "unmatched_gt_indices": sorted(unmatched_gt),
    }


def analyze_step_sweep(
    records,
    target,
    output_dir,
    sample_id,
    iou_threshold=0.5,
    score_threshold=0.0,
):
    gt_boxes = _as_list(target["boxes"])
    gt_labels = [int(value) for value in _as_list(target["labels"])]
    step_results = []
    for record in records:
        step = int(record["steps"])
        matched = match_step(
            record["detections"], gt_boxes, gt_labels,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        matched["step"] = step
        step_results.append(matched)

    per_gt = []
    for gt_index, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        comparison = {
            "gt_index": gt_index,
            "label": label,
            "box": box,
            "steps": {},
        }
        for result in step_results:
            match = result["gt_matches"].get(str(gt_index))
            comparison["steps"][str(result["step"])] = {
                "detected": match is not None,
                "score": match["score"] if match is not None else None,
                "iou": match["iou"] if match is not None else None,
                "prediction_index": (
                    match["prediction_index"] if match is not None else None
                ),
            }
        per_gt.append(comparison)

    report = {
        "analysis_scope": (
            "single transformed sample only; TP/FP/FN are not VOC mAP and "
            "must not be compared as full-dataset metrics"
        ),
        "coordinate_system": "resized detector input coordinates",
        "sample_id": sample_id,
        "iou_threshold": float(iou_threshold),
        "score_threshold": float(score_threshold),
        "gt_count": len(gt_boxes),
        "steps": step_results,
        "per_gt_comparison": per_gt,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "gt_step_analysis.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output / "gt_step_changes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "gt_index", "label", "step", "detected", "score", "iou",
            "prediction_index",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in per_gt:
            for step, values in sorted(
                item["steps"].items(), key=lambda pair: int(pair[0])
            ):
                writer.writerow(
                    {
                        "gt_index": item["gt_index"],
                        "label": item["label"],
                        "step": step,
                        "detected": values["detected"],
                        "score": values["score"],
                        "iou": values["iou"],
                        "prediction_index": values["prediction_index"],
                    }
                )

    summary_path = output / "gt_step_summary.txt"
    lines = [
        "Single-sample diagnostic only; this is NOT VOC mAP.",
        "sample=%s iou_threshold=%.3f score_threshold=%.3f"
        % (sample_id, float(iou_threshold), float(score_threshold)),
    ]
    for result in step_results:
        lines.append(
            "step=%d TP=%d FP=%d FN=%d eligible_predictions=%d"
            % (
                result["step"], result["tp"], result["fp"], result["fn"],
                result["eligible_prediction_count"],
            )
        )
    for item in per_gt:
        values = []
        for step, status in sorted(
            item["steps"].items(), key=lambda pair: int(pair[0])
        ):
            if status["detected"]:
                values.append(
                    "s%s:score=%.4f,iou=%.4f"
                    % (step, status["score"], status["iou"])
                )
            else:
                values.append("s%s:miss" % step)
        lines.append(
            "gt=%d label=%d %s"
            % (item["gt_index"], item["label"], " | ".join(values))
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, json_path, csv_path, summary_path


def save_ground_truth(target, img_meta, path):
    payload = {
        "coordinate_system": "resized detector input coordinates",
        "scale_factor": _json_value(img_meta.get("scale_factor")),
        "img_shape": _json_value(img_meta.get("img_shape")),
        "boxes": _as_list(target["boxes"]),
        "labels": [int(value) for value in _as_list(target["labels"])],
    }
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
