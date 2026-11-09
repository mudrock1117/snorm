#!/usr/bin/env python
"""Stage A0: audited checkpoint expansion plus RPN anchor sanity."""
from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F

from detection.config import cfg
from detection.modeling.build import build_detectors
from diagnostics.small_anchor import load_small_anchor_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-height", type=int, default=40)
    parser.add_argument("--feature-width", type=int, default=80)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = build_detectors(cfg).to(device)
    load_small_anchor_checkpoint(
        model,
        args.checkpoint,
        old_anchor_sizes=(64, 128, 256, 512),
        new_anchor_sizes=tuple(cfg.MODEL.RPN.ANCHOR_SIZES),
    )
    rpn = model.teacher.rpn_norm
    scales = tuple(int(v) for v in cfg.MODEL.RPN.ANCHOR_SIZES)
    ratios = tuple(float(v) for v in cfg.MODEL.RPN.ASPECT_RATIOS)
    expected_per_location = len(scales) * len(ratios)
    feature = torch.zeros(
        1,
        rpn.conv.in_channels,
        args.feature_height,
        args.feature_width,
        device=device,
    )
    with torch.no_grad():
        anchors = rpn.anchor_generator([feature])
        hidden = F.relu(rpn.conv(feature))
        cls_logits = rpn.cls_logits(hidden)
        bbox_pred = rpn.bbox_pred(hidden)

    expected_total = (
        args.feature_height * args.feature_width * expected_per_location
    )
    if tuple(cls_logits.shape) != (
            1, expected_per_location,
            args.feature_height, args.feature_width):
        raise RuntimeError("cls_logits shape does not match configured anchors")
    if tuple(bbox_pred.shape) != (
            1, expected_per_location * 4,
            args.feature_height, args.feature_width):
        raise RuntimeError("bbox_pred shape does not match configured anchors")
    if len(anchors) != 1 or anchors[0].size(0) != expected_total:
        raise RuntimeError("AnchorGenerator total anchor count mismatch")
    if 32 not in scales:
        raise RuntimeError("32 anchor was not configured")

    print("RPN anchor sanity:")
    print("scales =", list(scales))
    print("ratios =", list(ratios))
    print("feature =", list(feature.shape))
    print("anchor stride =", rpn.anchor_generator.strides)
    print("number per location =", expected_per_location)
    print("anchors per scale per location =", len(ratios))
    print("total anchors =", int(anchors[0].size(0)))
    print("cls_logits =", list(cls_logits.shape))
    print("bbox_pred =", list(bbox_pred.shape))
    print("source train dataset =", list(cfg.DATASETS.TRAINS))
    print("target/test datasets =", list(cfg.DATASETS.TESTS))
    print("PASS")


if __name__ == "__main__":
    main()
