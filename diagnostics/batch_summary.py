"""Aggregate fixed-sample diagnostics without computing VOC mAP."""
import csv
import json
from pathlib import Path


SIZES = ("small", "medium", "large")
THRESHOLDS = ("0.3", "0.5", "0.7")


def _empty_size():
    return {
        "gt": 0,
        "tp": 0,
        "fn": 0,
        "rpn_covered": {key: 0 for key in THRESHOLDS},
        "rpn_gt": {key: 0 for key in THRESHOLDS},
        "rpn_uncovered": 0,
        "rpn_covered_detection_failure": 0,
    }


def _empty_bucket():
    return {
        "samples": set(),
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "rpn_covered": {key: 0 for key in THRESHOLDS},
        "rpn_gt": {key: 0 for key in THRESHOLDS},
        "rpn_uncovered_fn": 0,
        "rpn_covered_detection_failure_fn": 0,
        "size": {size: _empty_size() for size in SIZES},
    }


def _ratio(value, total):
    return float(value) / total if total else 0.0


def summarize_reports(report_paths):
    buckets = {}
    sample_manifests = []
    for path in report_paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        domain = path.parent.parent.name
        sample_id = report["sample_id"]
        sample_manifests.append(
            {"domain": domain, "sample_id": sample_id, "report": str(path)}
        )
        per_gt = report.get("per_gt_comparison", [])
        for step_result in report.get("steps", []):
            step = int(step_result["step"])
            for group in (domain, "overall"):
                bucket = buckets.setdefault((group, step), _empty_bucket())
                bucket["samples"].add("%s/%s" % (domain, sample_id))
                bucket["gt"] += int(report.get("gt_count", 0))
                bucket["tp"] += int(step_result["tp"])
                bucket["fp"] += int(step_result["fp"])
                bucket["fn"] += int(step_result["fn"])
                coverage = step_result.get("rpn_gt_coverage")
                if isinstance(coverage, dict):
                    for threshold in THRESHOLDS:
                        values = coverage.get("thresholds", {}).get(threshold)
                        if values:
                            bucket["rpn_covered"][threshold] += int(values["covered"])
                            bucket["rpn_gt"][threshold] += int(values["gt_count"])
                sources = step_result.get("fn_sources", {})
                bucket["rpn_uncovered_fn"] += int(
                    sources.get("rpn_uncovered", 0)
                )
                bucket["rpn_covered_detection_failure_fn"] += int(
                    sources.get("rpn_covered_detection_failure", 0)
                )

                for item in per_gt:
                    size = item.get("scale")
                    if size not in SIZES:
                        continue
                    status = item.get("steps", {}).get(str(step), {})
                    size_bucket = bucket["size"][size]
                    size_bucket["gt"] += 1
                    detected = bool(status.get("detected"))
                    if detected:
                        size_bucket["tp"] += 1
                    else:
                        size_bucket["fn"] += 1
                        source = status.get("fn_source")
                        if source in (
                                "rpn_uncovered",
                                "rpn_covered_detection_failure"):
                            size_bucket[source] += 1
                    best_iou = status.get("rpn_best_iou")
                    if best_iou is not None:
                        for threshold in THRESHOLDS:
                            size_bucket["rpn_gt"][threshold] += 1
                            if float(best_iou) >= float(threshold):
                                size_bucket["rpn_covered"][threshold] += 1

    rows = []
    for (group, step), bucket in sorted(buckets.items()):
        row = {
            "group": group,
            "step": step,
            "samples": len(bucket["samples"]),
            "gt": bucket["gt"],
            "tp": bucket["tp"],
            "fp": bucket["fp"],
            "fn": bucket["fn"],
            "precision": _ratio(bucket["tp"], bucket["tp"] + bucket["fp"]),
            "recall": _ratio(bucket["tp"], bucket["tp"] + bucket["fn"]),
            "rpn_uncovered_fn": bucket["rpn_uncovered_fn"],
            "rpn_covered_detection_failure_fn": (
                bucket["rpn_covered_detection_failure_fn"]
            ),
        }
        for threshold in THRESHOLDS:
            row["rpn_recall_%s" % threshold] = _ratio(
                bucket["rpn_covered"][threshold],
                bucket["rpn_gt"][threshold],
            )
        for size in SIZES:
            values = bucket["size"][size]
            prefix = size + "_"
            row[prefix + "gt"] = values["gt"]
            row[prefix + "tp"] = values["tp"]
            row[prefix + "fn"] = values["fn"]
            row[prefix + "recall"] = _ratio(values["tp"], values["gt"])
            row[prefix + "rpn_uncovered"] = values["rpn_uncovered"]
            row[prefix + "rpn_covered_detection_failure"] = values[
                "rpn_covered_detection_failure"
            ]
            for threshold in THRESHOLDS:
                row[prefix + "rpn_recall_%s" % threshold] = _ratio(
                    values["rpn_covered"][threshold],
                    values["rpn_gt"][threshold],
                )
        if row["tp"] + row["fn"] != row["gt"]:
            raise RuntimeError("batch summary violates TP + FN = GT")
        if sum(row[size + "_fn"] for size in SIZES) != row["fn"]:
            raise RuntimeError("batch summary size FN does not equal total FN")
        for size in SIZES:
            if (
                row[size + "_rpn_uncovered"]
                + row[size + "_rpn_covered_detection_failure"]
                != row[size + "_fn"]
            ):
                raise RuntimeError(
                    "batch summary %s FN stages do not equal size FN" % size
                )
        if sum(row[size + "_rpn_uncovered"] for size in SIZES) != row["rpn_uncovered_fn"]:
            raise RuntimeError("size RPN-uncovered FN does not equal total")
        if sum(
            row[size + "_rpn_covered_detection_failure"] for size in SIZES
        ) != row["rpn_covered_detection_failure_fn"]:
            raise RuntimeError("size downstream FN does not equal total")
        rows.append(row)
    return rows, sample_manifests


def write_batch_summary(report_root, output_dir):
    report_root = Path(report_root)
    report_paths = sorted(report_root.rglob("gt_step_analysis.json"))
    if not report_paths:
        raise RuntimeError("no gt_step_analysis.json found under %s" % report_root)
    rows, samples = summarize_reports(report_paths)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = {
        "analysis_scope": (
            "fixed-sample diagnostic aggregate only; NOT full-dataset VOC mAP"
        ),
        "size_definition": (
            "unchanged existing diagnostic definition in resized detector coordinates"
        ),
        "report_count": len(report_paths),
        "samples": samples,
        "summary": rows,
    }
    json_path = output / "batch_summary.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output / "batch_summary.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    text_path = output / "batch_summary.txt"
    lines = [
        "Fixed-sample diagnostic aggregate only; NOT VOC mAP.",
        "reports=%d" % len(report_paths),
    ]
    for row in rows:
        lines.append("")
        lines.append(
            "%s step=%d samples=%d GT=%d TP=%d FP=%d FN=%d P=%.4f R=%.4f"
            % (
                row["group"], row["step"], row["samples"], row["gt"],
                row["tp"], row["fp"], row["fn"], row["precision"], row["recall"],
            )
        )
        lines.append(
            "  RPN Recall@0.3/0.5/0.7=%.4f/%.4f/%.4f FN(rpn/det)=%d/%d"
            % (
                row["rpn_recall_0.3"], row["rpn_recall_0.5"],
                row["rpn_recall_0.7"], row["rpn_uncovered_fn"],
                row["rpn_covered_detection_failure_fn"],
            )
        )
        for size in SIZES:
            prefix = size + "_"
            lines.append(
                "  %s: GT=%d TP=%d FN=%d Recall=%.4f "
                "RPN@0.3/0.5/0.7=%.4f/%.4f/%.4f FN(rpn/det)=%d/%d"
                % (
                    size, row[prefix + "gt"], row[prefix + "tp"],
                    row[prefix + "fn"], row[prefix + "recall"],
                    row[prefix + "rpn_recall_0.3"],
                    row[prefix + "rpn_recall_0.5"],
                    row[prefix + "rpn_recall_0.7"],
                    row[prefix + "rpn_uncovered"],
                    row[prefix + "rpn_covered_detection_failure"],
                )
            )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, text_path
