#!/usr/bin/env python
"""Generate the final offline SNorm detection root-cause report."""
from __future__ import print_function

import argparse

from diagnostics.root_cause import analyze_root_cause


def main():
    parser = argparse.ArgumentParser(
        description="Analyze existing fixed-sample reports without inference"
    )
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        args.report_root.rstrip("/\\") + "/root_cause"
    )
    paths = analyze_root_cause(
        args.report_root,
        output_dir,
        iou_threshold=args.iou_threshold,
        overlap_threshold=args.overlap_threshold,
    )
    print("Offline analysis complete; no detector inference was run.")
    print("Fixed-sample metrics only; NOT VOC mAP.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
