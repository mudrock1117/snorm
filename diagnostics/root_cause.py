"""Offline fixed-sample root-cause analysis for SNorm diagnostics.

This module consumes existing gt_step_analysis.json files. It does not run the
detector and never treats fixed-sample counts as VOC mAP.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from .gt_matching import box_iou


SIZES = ("small", "medium", "large")
FP_TYPES = (
    "duplicate_fp",
    "classification_fp",
    "localization_fp",
    "background_fp",
)
STEPS = (0, 1, 6)
PAIRS = ((0, 1), (1, 6), (0, 6))


def _domain_name(path):
    raw = path.parent.parent.name
    lower = raw.lower()
    if "fog" in lower:
        return "Fog"
    if "night" in lower:
        return "Night"
    raise RuntimeError("cannot infer Fog/Night domain from %s" % path)


def _new_step_bucket():
    return {
        "samples": set(),
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "size": {
            size: {
                "gt": 0,
                "tp": 0,
                "fn": 0,
                "rpn_uncovered": 0,
                "rpn_covered_detection_failure": 0,
            }
            for size in SIZES
        },
        "fp_types": {name: 0 for name in FP_TYPES},
        "fp_types_by_class": defaultdict(lambda: {name: 0 for name in FP_TYPES}),
        "rpn": {
            threshold: {"covered": 0, "gt": 0}
            for threshold in ("0.3", "0.5", "0.7")
        },
        "rpn_by_size": {
            threshold: {
                size: {"covered": 0, "gt": 0} for size in SIZES
            }
            for threshold in ("0.3", "0.5", "0.7")
        },
    }


def _new_transition_bucket():
    return {
        "samples": set(),
        "gt": 0,
        "persistent_tp": 0,
        "recovered_fn_to_tp": 0,
        "degraded_tp_to_fn": 0,
        "persistent_fn": 0,
        "rpn_uncovered_to_covered": 0,
        "rpn_covered_to_uncovered": 0,
        "rpn_persistent_uncovered": 0,
        "rpn_persistent_covered": 0,
        "recovered_by_size": {size: 0 for size in SIZES},
        "degraded_by_size": {size: 0 for size in SIZES},
        "recovered_from_rpn_uncovered": 0,
        "recovered_from_downstream_failure": 0,
        "degraded_to_rpn_uncovered": 0,
        "degraded_to_downstream_failure": 0,
        "new_fp_by_type": {name: 0 for name in FP_TYPES},
        "resolved_fp_by_type": {name: 0 for name in FP_TYPES},
    }


def _step_map(report, path):
    result = {}
    for item in report.get("steps", []):
        result[int(item["step"])] = item
    missing = [step for step in STEPS if step not in result]
    if missing:
        raise RuntimeError("%s lacks required steps %s" % (path, missing))
    return result


def _status(gt_item, step, path):
    status = gt_item.get("steps", {}).get(str(step))
    if not isinstance(status, dict):
        raise RuntimeError(
            "%s lacks per-GT status for gt=%s step=%d"
            % (path, gt_item.get("gt_index"), step)
        )
    return status


def _classify_fp(prediction, gt_items, matched_gt, iou_threshold, overlap_threshold):
    pred_box = prediction["box"]
    pred_label = int(prediction["label"])
    overlaps = [
        (int(gt["gt_index"]), int(gt["label"]), box_iou(pred_box, gt["box"]))
        for gt in gt_items
    ]

    # Mutually exclusive priority:
    # 1) duplicate: same-class GT already claimed by a better-scored TP;
    # 2) classification: high-IoU overlap with a different-class GT;
    # 3) localization: any overlap in [overlap_threshold, iou_threshold);
    # 4) background: max IoU below overlap_threshold.
    duplicate = [
        item for item in overlaps
        if item[1] == pred_label
        and item[2] >= iou_threshold
        and item[0] in matched_gt
    ]
    if duplicate:
        best = max(duplicate, key=lambda item: item[2])
        return "duplicate_fp", best[0], best[2]

    wrong_class = [
        item for item in overlaps
        if item[1] != pred_label and item[2] >= iou_threshold
    ]
    if wrong_class:
        best = max(wrong_class, key=lambda item: item[2])
        return "classification_fp", best[0], best[2]

    best = max(overlaps, key=lambda item: item[2]) if overlaps else (None, None, 0.0)
    if best[2] >= overlap_threshold:
        return "localization_fp", best[0], best[2]
    return "background_fp", best[0], best[2]


def _analyze_report(report, path, step_buckets, transition_buckets,
                    iou_threshold, overlap_threshold):
    score_threshold = float(report.get("score_threshold", -1.0))
    if abs(score_threshold - 0.5) > 1e-9:
        raise RuntimeError(
            "%s was matched at score_threshold=%s, but this analysis requires "
            "existing score>=0.5 matching. Recreate only the offline GT report "
            "at 0.5; detector inference is not required."
            % (path, score_threshold)
        )
    if abs(float(report.get("iou_threshold", -1.0)) - iou_threshold) > 1e-9:
        raise RuntimeError(
            "%s uses IoU threshold %s instead of %s"
            % (path, report.get("iou_threshold"), iou_threshold)
        )

    domain = _domain_name(path)
    sample_id = report.get("sample_id", path.parent.name)
    gt_items = report.get("per_gt_comparison", [])
    gt_count = int(report.get("gt_count", len(gt_items)))
    if len(gt_items) != gt_count:
        raise RuntimeError("%s per_gt count does not equal gt_count" % path)
    steps = _step_map(report, path)
    fp_records_by_step = {}

    for step in STEPS:
        result = steps[step]
        tp = int(result["tp"])
        fp = int(result["fp"])
        fn = int(result["fn"])
        if tp + fn != gt_count:
            raise RuntimeError(
                "%s step=%d violates TP + FN = GT (%d + %d != %d)"
                % (path, step, tp, fn, gt_count)
            )

        coverage = result.get("rpn_gt_coverage")
        if not isinstance(coverage, dict) or "per_gt" not in coverage:
            raise RuntimeError(
                "%s step=%d lacks offline RPN-GT coverage; existing inference "
                "cannot support failure-stage analysis" % (path, step)
            )
        coverage_by_gt = {
            int(item["gt_index"]): item for item in coverage["per_gt"]
        }

        fp_counts = {name: 0 for name in FP_TYPES}
        fp_by_class = defaultdict(lambda: {name: 0 for name in FP_TYPES})
        fp_records = []
        matched_gt = set(int(key) for key in result.get("gt_matches", {}).keys())
        for prediction in result.get("predictions", []):
            if prediction.get("matched"):
                continue
            fp_type, _, _ = _classify_fp(
                prediction, gt_items, matched_gt,
                iou_threshold, overlap_threshold,
            )
            fp_counts[fp_type] += 1
            fp_by_class[str(int(prediction["label"]))][fp_type] += 1
            fp_records.append({
                "type": fp_type,
                "label": int(prediction["label"]),
                "box": prediction["box"],
                "score": float(prediction["score"]),
            })
        fp_records_by_step[step] = fp_records
        if sum(fp_counts.values()) != fp:
            raise RuntimeError(
                "%s step=%d violates sum(FP types) = FP (%d != %d)"
                % (path, step, sum(fp_counts.values()), fp)
            )

        per_size = {
            size: {
                "gt": 0,
                "tp": 0,
                "fn": 0,
                "rpn_uncovered": 0,
                "rpn_covered_detection_failure": 0,
            }
            for size in SIZES
        }
        for gt in gt_items:
            gt_index = int(gt["gt_index"])
            size = gt.get("scale")
            if size not in SIZES:
                raise RuntimeError("%s has unknown target scale %s" % (path, size))
            status = _status(gt, step, path)
            per_size[size]["gt"] += 1
            if status.get("detected"):
                per_size[size]["tp"] += 1
            else:
                per_size[size]["fn"] += 1
                proposal = coverage_by_gt.get(gt_index)
                if proposal is None:
                    raise RuntimeError(
                        "%s lacks RPN coverage for gt=%d step=%d"
                        % (path, gt_index, step)
                    )
                if float(proposal["best_proposal_iou"]) < 0.5:
                    per_size[size]["rpn_uncovered"] += 1
                else:
                    per_size[size]["rpn_covered_detection_failure"] += 1

        size_fn = sum(values["fn"] for values in per_size.values())
        stage_fn = sum(
            values["rpn_uncovered"]
            + values["rpn_covered_detection_failure"]
            for values in per_size.values()
        )
        if size_fn != fn or stage_fn != fn:
            raise RuntimeError(
                "%s step=%d violates sum(FN size/stage) = FN "
                "(size=%d stage=%d fn=%d)" % (path, step, size_fn, stage_fn, fn)
            )

        for group in (domain, "Overall"):
            bucket = step_buckets[(group, step)]
            bucket["samples"].add("%s/%s" % (domain, sample_id))
            bucket["gt"] += gt_count
            bucket["tp"] += tp
            bucket["fp"] += fp
            bucket["fn"] += fn
            for size in SIZES:
                for key, value in per_size[size].items():
                    bucket["size"][size][key] += value
            for name in FP_TYPES:
                bucket["fp_types"][name] += fp_counts[name]
            for label, values in fp_by_class.items():
                for name in FP_TYPES:
                    bucket["fp_types_by_class"][label][name] += values[name]
            for threshold in ("0.3", "0.5", "0.7"):
                values = coverage.get("thresholds", {}).get(threshold)
                if not values:
                    raise RuntimeError(
                        "%s step=%d lacks RPN threshold %s"
                        % (path, step, threshold)
                    )
                bucket["rpn"][threshold]["covered"] += int(values["covered"])
                bucket["rpn"][threshold]["gt"] += int(values["gt_count"])
                for size in SIZES:
                    scale_values = coverage.get("per_scale", {}).get(
                        threshold, {}
                    ).get(size)
                    if not scale_values:
                        raise RuntimeError(
                            "%s step=%d lacks RPN scale coverage %s/%s"
                            % (path, step, threshold, size)
                        )
                    target = bucket["rpn_by_size"][threshold][size]
                    target["covered"] += int(scale_values["covered"])
                    target["gt"] += int(scale_values["gt_count"])

    # Compare final prediction FPs by class and spatial IoU, never by array index.
    # A Step6 FP is "new" when no unused Step0 FP of the same predicted class
    # overlaps it at IoU>=0.5. The reverse defines resolved Step0 FPs.
    step0_fps = fp_records_by_step[0]
    step6_fps = fp_records_by_step[6]
    used_step0 = set()
    matched_step6 = set()
    for step6_index, current in enumerate(step6_fps):
        candidates = []
        for step0_index, previous in enumerate(step0_fps):
            if step0_index in used_step0 or previous["label"] != current["label"]:
                continue
            overlap = box_iou(previous["box"], current["box"])
            if overlap >= 0.5:
                candidates.append((overlap, step0_index))
        if candidates:
            _, step0_index = max(candidates)
            used_step0.add(step0_index)
            matched_step6.add(step6_index)
    new_fp_types = {name: 0 for name in FP_TYPES}
    resolved_fp_types = {name: 0 for name in FP_TYPES}
    for index, item in enumerate(step6_fps):
        if index not in matched_step6:
            new_fp_types[item["type"]] += 1
    for index, item in enumerate(step0_fps):
        if index not in used_step0:
            resolved_fp_types[item["type"]] += 1

    for from_step, to_step in PAIRS:
        for group in (domain, "Overall"):
            bucket = transition_buckets[(group, from_step, to_step)]
            bucket["samples"].add("%s/%s" % (domain, sample_id))
            bucket["gt"] += gt_count
            if from_step == 0 and to_step == 6:
                for name in FP_TYPES:
                    bucket["new_fp_by_type"][name] += new_fp_types[name]
                    bucket["resolved_fp_by_type"][name] += resolved_fp_types[name]
        for gt in gt_items:
            size = gt["scale"]
            before = _status(gt, from_step, path)
            after = _status(gt, to_step, path)
            before_tp = bool(before.get("detected"))
            after_tp = bool(after.get("detected"))
            before_covered = float(before.get("rpn_best_iou")) >= 0.5
            after_covered = float(after.get("rpn_best_iou")) >= 0.5
            for group in (domain, "Overall"):
                bucket = transition_buckets[(group, from_step, to_step)]
                if before_tp and after_tp:
                    bucket["persistent_tp"] += 1
                elif not before_tp and after_tp:
                    bucket["recovered_fn_to_tp"] += 1
                    bucket["recovered_by_size"][size] += 1
                    if before_covered:
                        bucket["recovered_from_downstream_failure"] += 1
                    else:
                        bucket["recovered_from_rpn_uncovered"] += 1
                elif before_tp and not after_tp:
                    bucket["degraded_tp_to_fn"] += 1
                    bucket["degraded_by_size"][size] += 1
                    if after_covered:
                        bucket["degraded_to_downstream_failure"] += 1
                    else:
                        bucket["degraded_to_rpn_uncovered"] += 1
                else:
                    bucket["persistent_fn"] += 1

                if not before_covered and after_covered:
                    bucket["rpn_uncovered_to_covered"] += 1
                elif before_covered and not after_covered:
                    bucket["rpn_covered_to_uncovered"] += 1
                elif before_covered and after_covered:
                    bucket["rpn_persistent_covered"] += 1
                else:
                    bucket["rpn_persistent_uncovered"] += 1


def _ratio(value, total):
    return float(value) / total if total else 0.0


def _serialize_step(group, step, bucket):
    result = {
        "group": group,
        "step": step,
        "samples": len(bucket["samples"]),
        "gt": bucket["gt"],
        "tp": bucket["tp"],
        "fp": bucket["fp"],
        "fn": bucket["fn"],
        "precision": _ratio(bucket["tp"], bucket["tp"] + bucket["fp"]),
        "recall": _ratio(bucket["tp"], bucket["gt"]),
        "fn_vs_fp_driver": "FN" if bucket["fn"] > bucket["fp"] else (
            "FP" if bucket["fp"] > bucket["fn"] else "equal"
        ),
        "size": {},
        "fp_types": {},
        "fp_types_by_class": {},
        "rpn_recall": {},
        "rpn_recall_by_size": {},
    }
    for size in SIZES:
        values = dict(bucket["size"][size])
        values["recall"] = _ratio(values["tp"], values["gt"])
        values["rpn_uncovered_ratio_of_fn"] = _ratio(
            values["rpn_uncovered"], values["fn"]
        )
        values["rpn_covered_detection_failure_ratio_of_fn"] = _ratio(
            values["rpn_covered_detection_failure"], values["fn"]
        )
        result["size"][size] = values
    for name in FP_TYPES:
        result["fp_types"][name] = {
            "count": bucket["fp_types"][name],
            "ratio": _ratio(bucket["fp_types"][name], bucket["fp"]),
        }
    for label, values in sorted(bucket["fp_types_by_class"].items()):
        result["fp_types_by_class"][label] = {
            name: {
                "count": values[name],
                "ratio_within_class_fp": _ratio(values[name], sum(values.values())),
            }
            for name in FP_TYPES
        }
    for threshold in ("0.3", "0.5", "0.7"):
        values = bucket["rpn"][threshold]
        result["rpn_recall"][threshold] = _ratio(
            values["covered"], values["gt"]
        )
        result["rpn_recall_by_size"][threshold] = {}
        for size in SIZES:
            scale = bucket["rpn_by_size"][threshold][size]
            result["rpn_recall_by_size"][threshold][size] = _ratio(
                scale["covered"], scale["gt"]
            )
    return result


def _serialize_transition(group, from_step, to_step, bucket, step_lookup):
    result = {
        key: value for key, value in bucket.items()
        if key not in ("samples", "recovered_by_size", "degraded_by_size")
    }
    result.update({
        "group": group,
        "from_step": from_step,
        "to_step": to_step,
        "samples": len(bucket["samples"]),
        "recovered_by_size": dict(bucket["recovered_by_size"]),
        "degraded_by_size": dict(bucket["degraded_by_size"]),
    })
    before = step_lookup[(group, from_step)]
    after = step_lookup[(group, to_step)]
    result["fp_total_before"] = before["fp"]
    result["fp_total_after"] = after["fp"]
    result["fp_total_delta"] = after["fp"] - before["fp"]
    result["fp_type_delta"] = {
        name: (
            after["fp_types"][name]["count"]
            - before["fp_types"][name]["count"]
        )
        for name in FP_TYPES
    }
    error_delta = (
        after["fp"] + after["fn"] - before["fp"] - before["fn"]
    )
    result["total_detection_error_delta"] = error_delta
    result["effectiveness"] = (
        "beneficial" if error_delta < 0
        else "harmful" if error_delta > 0
        else "neutral"
    )
    return result


def _diagnosis(group, step, transition):
    size_items = sorted(
        (
            (size, values["fn"], 1.0 - values["recall"])
            for size, values in step["size"].items()
        ),
        key=lambda item: (item[2], item[1]),
        reverse=True,
    )
    fp_items = sorted(
        (
            (name, values["count"], values["ratio"])
            for name, values in step["fp_types"].items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    uncovered = sum(
        values["rpn_uncovered"] for values in step["size"].values()
    )
    downstream = sum(
        values["rpn_covered_detection_failure"]
        for values in step["size"].values()
    )
    stage_total = downstream + uncovered
    downstream_share = _ratio(downstream, stage_total)
    uncovered_share = _ratio(uncovered, stage_total)
    render_score = (
        max(0, transition["total_detection_error_delta"])
        + transition["degraded_tp_to_fn"]
    )
    evidence = [
        ("ROI/downstream detection", downstream),
        ("RPN coverage", uncovered),
        ("AHDR/render iteration", render_score),
    ]
    evidence.sort(key=lambda item: item[1], reverse=True)
    if transition["effectiveness"] == "harmful" and stage_total:
        supported_direction = "combined AHDR + %s" % (
            "ROI" if downstream >= uncovered else "RPN"
        )
    elif downstream_share >= 0.6:
        supported_direction = "ROI/downstream detection"
    elif uncovered_share >= 0.6:
        supported_direction = "RPN coverage"
    elif stage_total:
        supported_direction = "combined RPN + ROI"
    else:
        supported_direction = "AHDR/local enhancement diagnosis"
    ranking = [
        {"rank": index + 1, "candidate": name, "evidence_count": count}
        for index, (name, count) in enumerate(evidence)
    ]
    return {
        "group": group,
        "evaluated_step": 6,
        "low_precision_driver": step["fn_vs_fp_driver"],
        "fn_stage_driver": (
            "rpn_covered_detection_failure"
            if downstream > uncovered
            else "rpn_uncovered" if uncovered > downstream else "equal"
        ),
        "fn_stage_counts": {
            "rpn_uncovered": uncovered,
            "rpn_covered_detection_failure": downstream,
        },
        "highest_fn_rate_size": size_items[0][0] if size_items else None,
        "size_fn_counts_and_rates": {
            size: {"fn": fn, "fn_rate": rate}
            for size, fn, rate in size_items
        },
        "dominant_fp_type": fp_items[0][0] if fp_items else None,
        "fp_type_counts": {
            name: {"count": count, "ratio": ratio}
            for name, count, ratio in fp_items
        },
        "step0_to_step6": {
            "recovered_gt": transition["recovered_fn_to_tp"],
            "degraded_gt": transition["degraded_tp_to_fn"],
            "step6_fp_total_delta": transition["fp_total_delta"],
            "step6_fp_type_delta": transition["fp_type_delta"],
            "new_step6_fp_by_type": transition["new_fp_by_type"],
            "resolved_step0_fp_by_type": transition["resolved_fp_by_type"],
            "recovered_from_rpn_uncovered": transition["recovered_from_rpn_uncovered"],
            "recovered_from_downstream_failure": transition["recovered_from_downstream_failure"],
            "effectiveness": transition["effectiveness"],
        },
        "root_cause_ranking": ranking,
        "evidence_supported_direction": supported_direction,
        "recommendation_scope": (
            "Evidence ranking only. No model change or VOC-mAP claim is made."
        ),
    }


def analyze_root_cause(report_root, output_dir,
                       iou_threshold=0.5, overlap_threshold=0.3):
    report_root = Path(report_root)
    paths = sorted(report_root.rglob("gt_step_analysis.json"))
    if not paths:
        raise RuntimeError(
            "no gt_step_analysis.json found under %s" % report_root
        )
    step_buckets = defaultdict(_new_step_bucket)
    transition_buckets = defaultdict(_new_transition_bucket)
    sample_manifest = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        _analyze_report(
            report, path, step_buckets, transition_buckets,
            float(iou_threshold), float(overlap_threshold),
        )
        sample_manifest.append({
            "domain": _domain_name(path),
            "sample_id": report.get("sample_id", path.parent.name),
            "report": str(path),
        })

    groups = ("Fog", "Night", "Overall")
    steps = []
    lookup = {}
    for group in groups:
        for step in STEPS:
            if (group, step) not in step_buckets:
                raise RuntimeError("no reports available for %s step=%d" % (group, step))
            item = _serialize_step(group, step, step_buckets[(group, step)])
            steps.append(item)
            lookup[(group, step)] = item

    transitions = []
    transition_lookup = {}
    for group in groups:
        for from_step, to_step in PAIRS:
            item = _serialize_transition(
                group, from_step, to_step,
                transition_buckets[(group, from_step, to_step)], lookup,
            )
            transitions.append(item)
            transition_lookup[(group, from_step, to_step)] = item

    diagnosis = [
        _diagnosis(
            group, lookup[(group, 6)],
            transition_lookup[(group, 0, 6)],
        )
        for group in groups
    ]
    fog_diagnosis = diagnosis[0]
    night_diagnosis = diagnosis[1]
    fog_night_comparison = {
        "same_fn_stage_driver": (
            fog_diagnosis["fn_stage_driver"]
            == night_diagnosis["fn_stage_driver"]
        ),
        "fog_fn_stage_driver": fog_diagnosis["fn_stage_driver"],
        "night_fn_stage_driver": night_diagnosis["fn_stage_driver"],
        "same_dominant_fp_type": (
            fog_diagnosis["dominant_fp_type"]
            == night_diagnosis["dominant_fp_type"]
        ),
        "fog_dominant_fp_type": fog_diagnosis["dominant_fp_type"],
        "night_dominant_fp_type": night_diagnosis["dominant_fp_type"],
        "fog_supported_direction": fog_diagnosis["evidence_supported_direction"],
        "night_supported_direction": night_diagnosis["evidence_supported_direction"],
    }
    payload = {
        "analysis_scope": (
            "fixed-sample offline diagnostic only; NOT full-dataset VOC mAP"
        ),
        "source_root": str(report_root),
        "report_count": len(paths),
        "score_threshold": 0.5,
        "tp_iou_threshold": float(iou_threshold),
        "localization_overlap_threshold": float(overlap_threshold),
        "fp_classification_priority": list(FP_TYPES),
        "fp_priority_explanation": (
            "duplicate (same-class matched GT) > classification "
            "(different-class IoU>=0.5) > localization "
            "(max IoU in [0.3,0.5)) > background (max IoU<0.3)"
        ),
        "step0_step6_fp_identity_rule": (
            "class-equal greedy spatial matching at prediction IoU>=0.5; "
            "never array-index matching"
        ),
        "samples": sample_manifest,
        "steps": steps,
        "transitions": transitions,
        "diagnosis": diagnosis,
        "fog_night_comparison": fog_night_comparison,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "diagnostic_root_cause.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output / "diagnostic_root_cause.csv"
    rows = []
    for item in steps:
        base = {"group": item["group"], "step": item["step"]}
        for metric in ("gt", "tp", "fp", "fn", "precision", "recall"):
            rows.append(dict(base, row_type="step_metric", metric=metric,
                             value=item[metric]))
        for size, values in item["size"].items():
            for metric, value in values.items():
                rows.append(dict(base, row_type="fn_size_stage", size=size,
                                 metric=metric, value=value))
        for fp_type, values in item["fp_types"].items():
            rows.append(dict(base, row_type="fp_type", fp_type=fp_type,
                             metric="count", value=values["count"]))
            rows.append(dict(base, row_type="fp_type", fp_type=fp_type,
                             metric="ratio", value=values["ratio"]))
        for label, types in item["fp_types_by_class"].items():
            for fp_type, values in types.items():
                rows.append(dict(
                    base, row_type="fp_type_class", class_label=label,
                    fp_type=fp_type, metric="count", value=values["count"],
                ))
        for threshold, value in item["rpn_recall"].items():
            rows.append(dict(
                base, row_type="rpn_recall", threshold=threshold,
                metric="recall", value=value,
            ))
        for threshold, sizes in item["rpn_recall_by_size"].items():
            for size, value in sizes.items():
                rows.append(dict(
                    base, row_type="rpn_recall_size", threshold=threshold,
                    size=size, metric="recall", value=value,
                ))
    for item in transitions:
        base = {
            "group": item["group"],
            "from_step": item["from_step"],
            "to_step": item["to_step"],
            "row_type": "transition",
        }
        for metric in (
            "persistent_tp", "recovered_fn_to_tp", "degraded_tp_to_fn",
            "persistent_fn", "rpn_uncovered_to_covered",
            "rpn_covered_to_uncovered", "fp_total_delta",
            "total_detection_error_delta",
            "recovered_from_rpn_uncovered",
            "recovered_from_downstream_failure",
            "degraded_to_rpn_uncovered",
            "degraded_to_downstream_failure",
        ):
            rows.append(dict(base, metric=metric, value=item[metric]))
        for fp_type, value in item["fp_type_delta"].items():
            rows.append(dict(
                base, metric="fp_type_delta", fp_type=fp_type, value=value
            ))
        for fp_type, value in item["new_fp_by_type"].items():
            rows.append(dict(
                base, metric="new_step6_fp", fp_type=fp_type, value=value
            ))
        for fp_type, value in item["resolved_fp_by_type"].items():
            rows.append(dict(
                base, metric="resolved_step0_fp", fp_type=fp_type, value=value
            ))

    fields = [
        "row_type", "group", "step", "from_step", "to_step", "size",
        "class_label", "fp_type", "threshold", "metric", "value",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    txt_path = output / "diagnostic_root_cause.txt"
    lines = [
        "Fixed-sample offline diagnostic only; NOT VOC mAP.",
        "reports=%d score>=0.5 TP-IoU>=%.2f" % (len(paths), iou_threshold),
        "FP priority: duplicate > classification > localization > background.",
        "",
    ]
    for item in diagnosis:
        transition = item["step0_to_step6"]
        lines.extend([
            "[%s]" % item["group"],
            "Low accuracy count driver: %s" % item["low_precision_driver"],
            "FN stage driver: %s (%s)"
            % (item["fn_stage_driver"], item["fn_stage_counts"]),
            "Highest FN-rate size: %s; sizes=%s"
            % (item["highest_fn_rate_size"], item["size_fn_counts_and_rates"]),
            "Dominant FP type: %s; FP=%s"
            % (item["dominant_fp_type"], item["fp_type_counts"]),
            "Step0->6 recovered=%d degraded=%d effectiveness=%s"
            % (
                transition["recovered_gt"], transition["degraded_gt"],
                transition["effectiveness"],
            ),
            "Step6 FP delta=%d; type delta=%s; new=%s; resolved=%s"
            % (
                transition["step6_fp_total_delta"],
                transition["step6_fp_type_delta"],
                transition["new_step6_fp_by_type"],
                transition["resolved_step0_fp_by_type"],
            ),
            "Evidence-supported direction: %s"
            % item["evidence_supported_direction"],
            "Root-cause ranking: %s" % item["root_cause_ranking"],
            "",
        ])
    lines.extend([
        "[Fog vs Night]",
        str(fog_night_comparison),
        "",
    ])
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, txt_path
