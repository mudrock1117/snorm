"""Aggregate fixed-sample diagnostics without computing VOC mAP."""
import csv
import json
from pathlib import Path


def _empty_bucket():
    return {
        "samples": set(),
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "rpn_covered": {"0.3": 0, "0.5": 0, "0.7": 0},
        "rpn_gt": {"0.3": 0, "0.5": 0, "0.7": 0},
        "scale_detected": {"small": 0, "medium": 0, "large": 0},
        "scale_gt": {"small": 0, "medium": 0, "large": 0},
        "rpn_uncovered_fn": 0,
        "rpn_covered_detection_failure_fn": 0,
    }


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
                key = (group, step)
                bucket = buckets.setdefault(key, _empty_bucket())
                bucket["samples"].add("%s/%s" % (domain, sample_id))
                bucket["gt"] += int(report.get("gt_count", 0))
                bucket["tp"] += int(step_result["tp"])
                bucket["fp"] += int(step_result["fp"])
                bucket["fn"] += int(step_result["fn"])
                coverage = step_result.get("rpn_gt_coverage")
                if isinstance(coverage, dict):
                    for threshold in ("0.3", "0.5", "0.7"):
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
                    scale = item.get("scale")
                    status = item.get("steps", {}).get(str(step), {})
                    if scale in bucket["scale_gt"]:
                        bucket["scale_gt"][scale] += 1
                        if status.get("detected"):
                            bucket["scale_detected"][scale] += 1

    rows = []
    for (group, step), bucket in sorted(buckets.items()):
        precision_denominator = bucket["tp"] + bucket["fp"]
        recall_denominator = bucket["tp"] + bucket["fn"]
        row = {
            "group": group,
            "step": step,
            "samples": len(bucket["samples"]),
            "gt": bucket["gt"],
            "tp": bucket["tp"],
            "fp": bucket["fp"],
            "fn": bucket["fn"],
            "precision": (
                float(bucket["tp"]) / precision_denominator
                if precision_denominator else 0.0
            ),
            "recall": (
                float(bucket["tp"]) / recall_denominator
                if recall_denominator else 0.0
            ),
            "rpn_recall_0.3": (
                float(bucket["rpn_covered"]["0.3"]) / bucket["rpn_gt"]["0.3"]
                if bucket["rpn_gt"]["0.3"] else 0.0
            ),
            "rpn_recall_0.5": (
                float(bucket["rpn_covered"]["0.5"]) / bucket["rpn_gt"]["0.5"]
                if bucket["rpn_gt"]["0.5"] else 0.0
            ),
            "rpn_recall_0.7": (
                float(bucket["rpn_covered"]["0.7"]) / bucket["rpn_gt"]["0.7"]
                if bucket["rpn_gt"]["0.7"] else 0.0
            ),
            "small_recall": (
                float(bucket["scale_detected"]["small"])
                / bucket["scale_gt"]["small"]
                if bucket["scale_gt"]["small"] else 0.0
            ),
            "medium_recall": (
                float(bucket["scale_detected"]["medium"])
                / bucket["scale_gt"]["medium"]
                if bucket["scale_gt"]["medium"] else 0.0
            ),
            "large_recall": (
                float(bucket["scale_detected"]["large"])
                / bucket["scale_gt"]["large"]
                if bucket["scale_gt"]["large"] else 0.0
            ),
            "rpn_uncovered_fn": bucket["rpn_uncovered_fn"],
            "rpn_covered_detection_failure_fn": (
                bucket["rpn_covered_detection_failure_fn"]
            ),
        }
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
        lines.append(
            (
                "%s step=%d samples=%d GT=%d TP=%d FP=%d FN=%d "
                "P=%.4f R=%.4f RPN@.3/.5/.7=%.4f/%.4f/%.4f "
                "small/medium/large=%.4f/%.4f/%.4f "
                "FN(rpn/det)=%d/%d"
            )
            % (
                row["group"], row["step"], row["samples"], row["gt"],
                row["tp"], row["fp"], row["fn"], row["precision"],
                row["recall"], row["rpn_recall_0.3"],
                row["rpn_recall_0.5"], row["rpn_recall_0.7"],
                row["small_recall"], row["medium_recall"],
                row["large_recall"], row["rpn_uncovered_fn"],
                row["rpn_covered_detection_failure_fn"],
            )
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, text_path
