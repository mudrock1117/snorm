#!/usr/bin/env python
"""Aggregate Fog/Night fixed-sample diagnostic reports."""
from __future__ import print_function

import argparse

from diagnostics.batch_summary import write_batch_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate fixed-sample SNorm diagnostics (not VOC mAP)"
    )
    parser.add_argument(
        "--report-root",
        required=True,
        help="Root containing <domain>/<sample>/gt_step_analysis.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to <report-root>/summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.report_root.rstrip("/\\") + "/summary")
    json_path, csv_path, text_path = write_batch_summary(
        args.report_root, output_dir
    )
    print("Fixed-sample diagnostic aggregate only; NOT VOC mAP.")
    print("JSON: %s" % json_path)
    print("CSV: %s" % csv_path)
    print("Summary: %s" % text_path)


if __name__ == "__main__":
    main()
