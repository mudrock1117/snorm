#!/usr/bin/env python
"""Probe compute_norm_parameters on one fixed sample and restore state by default."""
import argparse
from pathlib import Path

import torch

from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.modeling.build import build_detectors
from diagnostics.detection_analysis import load_sample_ids, seed_everything
from diagnostics.epoch_probe import probe_epoch_boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    model = build_detectors(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state, strict=False)
    print("missing_keys:", incompatible.missing_keys)
    print("unexpected_keys:", incompatible.unexpected_keys)
    model.eval()

    wanted = set(load_sample_ids(args.sample_ids))
    loaders = build_data_loaders(
        cfg.DATASETS.TESTS,
        transforms=cfg.INPUT.TRANSFORMS_TEST,
        is_train=False,
        distributed=False,
        num_workers=0,
    )
    selected = None
    for loader in loaders:
        for batch in loader:
            images, img_metas, targets = batch[0], batch[1], batch[2]
            info = img_metas[0]["img_info"]
            file_name = str(info.get("file_name", info.get("id", "unknown")))
            candidates = {file_name, Path(file_name).name, Path(file_name).stem, str(info.get("id", ""))}
            if wanted.intersection(candidates):
                selected = (images.to(device), img_metas, targets)
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("none of the fixed sample IDs was found")

    images, img_metas, targets = selected
    snorm = model.teacher.proj_norm

    @torch.no_grad()
    def render_callback():
        seed_everything(args.seed)
        return snorm(images)[0].detach()

    @torch.no_grad()
    def detection_callback():
        seed_everything(args.seed)
        return model(images, img_metas, targets, moe_mode=7)[0]

    report = probe_epoch_boundary(
        snorm,
        output_dir=args.output_dir,
        render_callback=render_callback,
        detection_callback=detection_callback,
        apply_update=False,
    )
    print("probe saved:", report)
    print("SNorm state restored; checkpoint/model state was not permanently modified")


if __name__ == "__main__":
    main()
