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
from diagnostics.checkpoint import load_detector_checkpoint
from diagnostics.detection_analysis import load_sample_ids, run_step_sweep, seed_everything


def _rectangle(draw, box, color):
    left, top, right, bottom = [int(round(value)) for value in box]
    for offset in range(2):
        draw.rectangle(
            (left - offset, top - offset, right + offset, bottom + offset),
            outline=color,
        )


def _draw_boxes(image_path, output_path, gt_boxes, detection):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        _rectangle(draw, box, (0, 255, 0))
    for box, score in zip(detection.get("boxes", []), detection.get("scores", [])):
        if score >= 0.5:
            _rectangle(draw, box, (255, 0, 0))
    image.save(output_path)


def _validate_geometry(images, img_meta, target):
    height, width = int(images.size(-2)), int(images.size(-1))
    if "scale_factor" not in img_meta:
        raise KeyError("img_meta is missing scale_factor")
    boxes = target.get("boxes")
    if boxes is None:
        raise KeyError("target is missing transformed boxes")
    if boxes.numel() > 0:
        if float(boxes.min().item()) < -1.0:
            raise ValueError("transformed GT boxes contain negative coordinates")
        if float(boxes[:, 0::2].max().item()) > width + 1.0:
            raise ValueError("GT x coordinates exceed resized tensor width")
        if float(boxes[:, 1::2].max().item()) > height + 1.0:
            raise ValueError("GT y coordinates exceed resized tensor height")
    return height, width


def _sample_candidates(img_meta):
    if not isinstance(img_meta, dict):
        raise TypeError("img_meta must be a dictionary")
    info = img_meta.get("img_info")
    if not isinstance(info, dict):
        raise TypeError("img_meta['img_info'] must be a dictionary")
    file_name = str(info.get("file_name", info.get("id", "unknown")))
    return info, {
        file_name, Path(file_name).name, Path(file_name).stem, str(info.get("id", ""))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-ids", required=True, help="one file stem/name/id per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", default="0,1,2,3,4,5,6")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--allow-critical-missing", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    model = build_detectors(cfg).to(device)
    load_detector_checkpoint(
        model, args.checkpoint,
        allow_critical_missing=args.allow_critical_missing,
    )
    model.eval()

    test_names = list(cfg.DATASETS.TESTS)
    if not any("fog" in name.lower() for name in test_names):
        raise ValueError("DATASETS.TESTS has no Fog dataset: %s" % test_names)
    if not any("night" in name.lower() for name in test_names):
        raise ValueError("DATASETS.TESTS has no Night dataset: %s" % test_names)
    print("diagnostic_test_datasets:", test_names)
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
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 8:
                raise ValueError("test loader must return the repository's 8-item batch")
            images, img_metas, targets = batch[0], batch[1], batch[2]
            if images.ndim != 4 or images.size(0) != 1:
                raise ValueError("diagnostic loader requires image batch size 1")
            if len(img_metas) != 1 or len(targets) != 1:
                raise ValueError("img_metas and targets must each contain one sample")
            info, candidates = _sample_candidates(img_metas[0])
            matched = wanted.intersection(candidates)
            if not matched:
                continue
            sample_id = sorted(matched)[0]
            sample_dir = output_root / domain / sample_id
            _validate_geometry(images, img_metas[0], targets[0])
            images = images.to(device)
            path = run_step_sweep(
                model, images, img_metas, targets, str(sample_dir),
                sample_id=sample_id, steps=steps, seed=args.seed,
            )
            records = json.loads(path.read_text(encoding="utf-8"))
            gt_boxes = targets[0]["boxes"].detach().cpu().tolist()
            overlay_dir = sample_dir / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            first_trace = (
                sample_dir / "ahdr" / ("steps_%s" % sample_id)
                / ("%s_steps_%d" % (sample_id, int(steps[0]))) / "call_000000"
            )
            raw_input_path = first_trace / "step00_raw_input.png"
            if raw_input_path.exists():
                _draw_boxes(
                    raw_input_path, overlay_dir / "original_gt.png",
                    gt_boxes, {"boxes": [], "scores": []},
                )
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
