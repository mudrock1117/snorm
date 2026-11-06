#!/usr/bin/env python
"""Run fixed-sample SNorm diagnostics without entering the VOC evaluation loop."""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.modeling.build import build_detectors
from diagnostics.detection_analysis import load_sample_ids, run_step_sweep, seed_everything


def _draw_boxes(image_path, output_path, gt_boxes, detection):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        draw.rectangle(tuple(box), outline=(0, 255, 0), width=2)
    for box, score in zip(detection.get("boxes", []), detection.get("scores", [])):
        if score >= 0.5:
            draw.rectangle(tuple(box), outline=(255, 0, 0), width=2)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-ids", required=True, help="one file stem/name/id per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", default="0,1,2,3,4,5,6")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--save-tensors", action="store_true")
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
    if not wanted:
        raise ValueError("sample list is empty")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["SNORM_DIAG_SAVE_TENSORS"] = "1" if args.save_tensors else "0"
    os.environ["SNORM_DIAG_SAVE_PNG"] = "1"
    steps = tuple(int(item) for item in args.steps.split(","))

    found = set()
    loaders = build_data_loaders(
        cfg.DATASETS.TESTS,
        transforms=cfg.INPUT.TRANSFORMS_TEST,
        is_train=False,
        distributed=False,
        num_workers=0,
    )
    for loader in loaders:
        domain = loader.dataset.dataset_name
        for images, img_metas, targets, _, _, _, _, _ in loader:
            info = img_metas[0]["img_info"]
            file_name = str(info.get("file_name", info.get("id", "unknown")))
            candidates = {file_name, Path(file_name).name, Path(file_name).stem, str(info.get("id", ""))}
            matched = wanted.intersection(candidates)
            if not matched:
                continue
            sample_id = sorted(matched)[0]
            sample_dir = output_root / domain / sample_id
            images = images.to(device)
            path = run_step_sweep(
                model, images, img_metas, targets, str(sample_dir),
                sample_id=sample_id, steps=steps, seed=args.seed,
            )
            records = json.loads(path.read_text(encoding="utf-8"))
            gt_boxes = targets[0]["boxes"].detach().cpu().tolist()
            overlay_dir = sample_dir / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            for record in records:
                step = int(record["steps"])
                trace_dir = (
                    sample_dir / "ahdr" / ("steps_%s" % sample_id)
                    / ("%s_steps_%d" % (sample_id, step)) / "call_000000"
                )
                image_path = trace_dir / (
                    "step00_save_im0.png" if step == 0 else "step%02d_blended.png" % step
                )
                if image_path.exists():
                    _draw_boxes(
                        image_path,
                        overlay_dir / ("step%02d_gt_pred.png" % step),
                        gt_boxes,
                        record["detections"],
                    )
            found.add(sample_id)
            print("diagnosed", domain, sample_id, "->", sample_dir)
            if found == wanted:
                break
        if found == wanted:
            break

    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError("sample IDs not found: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
