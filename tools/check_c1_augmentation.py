#!/usr/bin/env python
"""Read-only visualization audit for C1 source augmentation."""
from __future__ import absolute_import, print_function

import argparse
import json
import os
import random

import numpy as np
import torch
import torchvision

from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.data.source_diversification import SourceAdaptiveDiversification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", default="diagnostic_runs/c1_source_diversification")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    loader = build_data_loaders(
        cfg.DATASETS.TRAINS, transforms=cfg.INPUT.TRANSFORMS_TRAIN,
        is_train=True, distributed=False, batch_size=1, num_workers=0,
        shuffle=False,
    )
    augment = SourceAdaptiveDiversification(cfg.C1_SOURCE_DIVERSIFICATION)
    manifest = []
    for index, batch in enumerate(loader):
        if index >= args.num_samples:
            break
        images, metas, targets = batch[0], batch[1], batch[2]
        boxes_before = targets[0]["boxes"].clone()
        augmented, stats = augment(images, metas, targets)
        if not torch.equal(boxes_before, targets[0]["boxes"]):
            raise RuntimeError("C1 changed GT boxes")
        original01 = (images * .5 + .5).clamp(0, 1)
        augmented01 = (augmented * .5 + .5).clamp(0, 1)
        difference = (augmented01 - original01).abs()
        prefix = os.path.join(args.output_dir, "sample_%03d" % index)
        torchvision.utils.save_image(original01, prefix + "_original.png")
        torchvision.utils.save_image(augmented01, prefix + "_augmented.png")
        torchvision.utils.save_image(difference / difference.max().clamp_min(1e-6), prefix + "_difference.png")
        torchvision.utils.save_image(stats.get("object_mask", torch.zeros_like(images[:, :1])), prefix + "_object_mask.png")
        torchvision.utils.save_image(stats.get("small_mask", torch.zeros_like(images[:, :1])), prefix + "_small_mask.png")
        manifest.append({
            "index": index, "img_shape": list(metas[0]["img_shape"]),
            "padded_shape": list(images.shape), "boxes": boxes_before.tolist(),
            "stats": dict((k, v) for k, v in stats.items()
                          if k not in ("object_mask", "small_mask")),
        })
        print("[C1 VIS] saved", prefix)
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print("[C1 VIS PASS] samples =", len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
