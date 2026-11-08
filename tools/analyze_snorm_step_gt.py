#!/usr/bin/env python
"""Analyze an existing step sweep JSON against saved transformed GT."""
import argparse
import json
from pathlib import Path

import torch

from diagnostics.gt_matching import analyze_step_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-json", required=True)
    parser.add_argument("--ground-truth-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    args = parser.parse_args()

    sweep_path = Path(args.sweep_json)
    gt_path = Path(args.ground_truth_json)
    records = json.loads(sweep_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    target = {
        "boxes": torch.tensor(ground_truth["boxes"], dtype=torch.float32),
        "labels": torch.tensor(ground_truth["labels"], dtype=torch.int64),
    }
    sample_id = args.sample_id
    if sample_id is None:
        sample_id = records[0].get("sample_id", sweep_path.stem) if records else sweep_path.stem
    _, json_path, csv_path, summary_path = analyze_step_sweep(
        records,
        target,
        args.output_dir,
        sample_id,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    print("single-sample analysis only; NOT VOC mAP")
    print("json:", json_path)
    print("csv:", csv_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
