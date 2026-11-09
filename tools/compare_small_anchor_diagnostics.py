#!/usr/bin/env python
"""Compare fixed-sample E2 and A-small-anchor summaries without value claims."""
from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    ("small_rpn_recall_0.5", "small RPN Recall@0.5"),
    ("small_recall", "small Recall"),
    ("medium_recall", "medium Recall"),
    ("large_recall", "large Recall"),
    ("rpn_uncovered_fn", "RPN-uncovered FN"),
)


def domain_name(group):
    lower = str(group).lower()
    if "fog" in lower:
        return "Fog"
    if "night" in lower:
        return "Night"
    if lower == "overall":
        return "Overall"
    return None


def load_rows(path, step):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for row in payload.get("summary", []):
        if int(row["step"]) != int(step):
            continue
        domain = domain_name(row["group"])
        if domain is None:
            continue
        if domain in rows:
            raise RuntimeError("duplicate %s step=%d row in %s" % (domain, step, path))
        rows[domain] = row
    missing = [name for name in ("Fog", "Night", "Overall") if name not in rows]
    if missing:
        raise RuntimeError("%s lacks summary groups: %s" % (path, missing))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--experiment-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step", type=int, default=6)
    args = parser.parse_args()

    baseline = load_rows(args.baseline_summary, args.step)
    experiment = load_rows(args.experiment_summary, args.step)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    lines = [
        "E2 baseline vs A-small-anchor",
        "Fixed-sample diagnostics only; NOT VOC mAP.",
        "step=%d" % args.step,
    ]
    for domain in ("Fog", "Night", "Overall"):
        lines.extend(["", domain + ":"])
        for key, label in METRICS:
            if key not in baseline[domain] or key not in experiment[domain]:
                raise RuntimeError("summary is missing required metric %s" % key)
            old = float(baseline[domain][key])
            new = float(experiment[domain][key])
            delta = new - old
            records.append({
                "domain": domain,
                "step": args.step,
                "metric": key,
                "baseline": old,
                "experiment": new,
                "delta": delta,
            })
            lines.append(
                "  %-27s %.6f -> %.6f -> delta %+.6f"
                % (label + ":", old, new, delta)
            )

    payload = {
        "analysis_scope": "fixed-sample diagnostics only; NOT VOC mAP",
        "baseline": str(args.baseline_summary),
        "experiment": str(args.experiment_summary),
        "step": args.step,
        "comparisons": records,
    }
    json_path = output / "E2_vs_A_smallanchor.json"
    csv_path = output / "E2_vs_A_smallanchor.csv"
    text_path = output / "E2_vs_A_smallanchor.txt"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("domain", "step", "metric", "baseline", "experiment", "delta"),
        )
        writer.writeheader()
        writer.writerows(records)
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("JSON:", json_path)
    print("CSV:", csv_path)
    print("TXT:", text_path)


if __name__ == "__main__":
    main()
