"""Detection-level diagnostics executed outside the normal VOC evaluation loop."""
from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from .tensor_metrics import high_frequency_energy, tensor_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def boxes_to_masks(
    boxes: torch.Tensor,
    image_shape: Tuple[int, int],
    small_object_area: float = 32.0 * 32.0,
) -> Dict[str, torch.Tensor]:
    """Rasterize boxes already transformed into the detector input coordinates."""
    height, width = map(int, image_shape)
    masks = {
        "all": torch.zeros((1, 1, height, width), device=boxes.device),
        "small": torch.zeros((1, 1, height, width), device=boxes.device),
        "regular": torch.zeros((1, 1, height, width), device=boxes.device),
    }
    for box in boxes.detach():
        x1, y1, x2, y2 = box.tolist()
        left = max(0, min(width, int(np.floor(x1))))
        top = max(0, min(height, int(np.floor(y1))))
        right = max(0, min(width, int(np.ceil(x2))))
        bottom = max(0, min(height, int(np.ceil(y2))))
        if right <= left or bottom <= top:
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        group = "small" if area < small_object_area else "regular"
        masks["all"][..., top:bottom, left:right] = 1.0
        masks[group][..., top:bottom, left:right] = 1.0
    return masks


@torch.no_grad()
def image_information(value: torch.Tensor, value_range: str = "neg1_1") -> Dict[str, object]:
    image = value.detach().float()
    if value_range == "neg1_1":
        image = image * 0.5 + 0.5
    elif value_range != "zero_one":
        raise ValueError("value_range must be neg1_1 or zero_one")
    image = image.clamp(0.0, 1.0)
    luminance = (
        0.2126 * image[:, 0:1]
        + 0.7152 * image[:, 1:2]
        + 0.0722 * image[:, 2:3]
    )
    dx = (luminance[..., :, 1:] - luminance[..., :, :-1]).abs().mean()
    dy = (luminance[..., 1:, :] - luminance[..., :-1, :]).abs().mean()
    result = tensor_metrics(image)
    result.update(
        luminance_mean=float(luminance.mean().cpu()),
        edge_energy=float(((dx + dy) * 0.5).cpu()),
        high_frequency_energy=high_frequency_energy(image),
    )
    return result


def serialize_detection(detection: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    result = {}
    for key in ("boxes", "scores", "labels"):
        value = detection.get(key)
        if value is not None:
            result[key] = value.detach().cpu().tolist()
    return result


@contextmanager
def _temporary_environment(values: Mapping[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@torch.no_grad()
def run_step_sweep(
    model: torch.nn.Module,
    images: torch.Tensor,
    img_metas,
    targets,
    output_dir: str,
    sample_id: str,
    steps: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    seed: int = 20261104,
) -> Path:
    """Run extra detector forwards only from an explicit diagnostic entry.

    step=0 uses SNORM_ABLATION=norm_only; other values use the original full path
    with SNORM_STEPS set. RPN proposals/ROI logits are not exposed by FasterRCNN's
    public eval return and are therefore marked unavailable instead of fabricated.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    records = []
    try:
        for step in steps:
            seed_everything(seed)
            env = {
                "SNORM_DIAG": "1",
                "SNORM_DIAG_DIR": str(output / "ahdr"),
                "SNORM_DIAG_RUN_ID": "steps_%s" % sample_id,
                "SNORM_DIAG_SAMPLE_IDS": "%s_steps_%d" % (sample_id, step),
                "SNORM_STEPS": str(max(1, int(step))),
                "SNORM_ABLATION": "norm_only" if int(step) == 0 else "full",
            }
            with _temporary_environment(env):
                detection = model(images, img_metas, targets, moe_mode=7)[0]
            records.append(
                {
                    "sample_id": sample_id,
                    "steps": int(step),
                    "detections": serialize_detection(detection),
                    "rpn_proposals": "unavailable: not exposed by public eval return",
                    "roi_logits": "unavailable: not exposed by public eval return",
                }
            )
    finally:
        model.train(was_training)
    path = output / ("%s_step_sweep.json" % sample_id)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_sample_ids(path: str) -> Iterable[str]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            yield value
