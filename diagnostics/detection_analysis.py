"""Detection-level diagnostics executed outside the normal VOC evaluation loop."""
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from .internal_capture import DetectionInternalCapture
from .tensor_metrics import high_frequency_energy, tensor_metrics


def _progress(message):
    print("[SNORM-DIAG] %s" % message)
    sys.stdout.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def extract_single_detection(model_output):
    if not isinstance(model_output, (list, tuple)):
        raise TypeError("FasterRCNN eval output must be a list/tuple of detections")
    if len(model_output) != 1:
        raise ValueError(
            "diagnostic entry requires batch size 1, got %d detections" % len(model_output)
        )
    detection = model_output[0]
    if not isinstance(detection, dict):
        raise TypeError("single detection must be a dictionary")
    missing = [key for key in ("boxes", "scores", "labels") if key not in detection]
    if missing:
        raise KeyError("detection is missing keys: %s" % ", ".join(missing))
    return detection


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


@contextmanager
def _temporary_snorm_diagnostics(model: torch.nn.Module):
    states = []
    for module in model.modules():
        if type(module).__name__ == "SNorm" and hasattr(module, "_diag_enabled"):
            states.append((module, module._diag_enabled, module._diag_call_index))
            module._diag_enabled = True
            module._diag_call_index = 0
    try:
        yield
    finally:
        for module, enabled, call_index in states:
            module._diag_enabled = enabled
            module._diag_call_index = call_index


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
    capture_internals: bool = False,
    save_internal_tensors: bool = False,
) -> Path:
    """Run extra detector forwards only from an explicit diagnostic entry.

    step=0 uses SNORM_ABLATION=norm_only; other values use the original full path
    with SNORM_STEPS set. RPN proposals/ROI logits are not exposed by FasterRCNN's
    public eval return and are therefore marked unavailable instead of fabricated.
    """
    if images.ndim != 4 or images.size(0) != 1:
        raise ValueError("diagnostic entry requires images with batch size 1")
    if len(img_metas) != 1 or len(targets) != 1:
        raise ValueError("img_metas and targets must each contain exactly one sample")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module_modes = [(module, module.training) for module in model.modules()]
    rng_context = preserve_rng_state()
    rng_context.__enter__()
    records = []
    try:
        model.eval()
        for step in steps:
            _progress("sample=%s step=%d inference start" % (sample_id, int(step)))
            seed_everything(seed)
            env = {
                "SNORM_DIAG": "1",
                "SNORM_DIAG_DIR": str(output / "ahdr"),
                "SNORM_DIAG_RUN_ID": "steps_%s" % sample_id,
                "SNORM_DIAG_SAMPLE_IDS": "%s_steps_%d" % (sample_id, step),
                "SNORM_STEPS": str(max(1, int(step))),
                "SNORM_ABLATION": "norm_only" if int(step) == 0 else "full",
            }
            capture = None
            with _temporary_environment(env), _temporary_snorm_diagnostics(model):
                if capture_internals:
                    capture = DetectionInternalCapture(
                        model,
                        output / "internals",
                        step,
                        save_tensors=save_internal_tensors,
                    )
                    with capture:
                        detection = extract_single_detection(
                            model(images, img_metas, targets, moe_mode=7)
                        )
                    internal_summary, internal_path = capture.save()
                    rpn_value = internal_summary.get(
                        "rpn", internal_summary.get("rpn_error", "not captured")
                    )
                    roi_value = internal_summary.get(
                        "roi", internal_summary.get("roi_error", "not captured")
                    )
                    internal_file = str(internal_path)
                else:
                    detection = extract_single_detection(
                        model(images, img_metas, targets, moe_mode=7)
                    )
                    rpn_value = "unavailable: internal capture disabled"
                    roi_value = "unavailable: internal capture disabled"
                    internal_file = None
            records.append(
                {
                    "sample_id": sample_id,
                    "steps": int(step),
                    "detections": serialize_detection(detection),
                    "rpn_proposals": rpn_value,
                    "roi_logits": roi_value,
                    "internal_summary_file": internal_file,
                }
            )
            _progress("sample=%s step=%d inference end" % (sample_id, int(step)))
    finally:
        for module, training in module_modes:
            module.training = training
        rng_context.__exit__(None, None, None)
    path = output / ("%s_step_sweep.json" % sample_id)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress("saved detector sweep: %s" % path)
    return path


def load_sample_ids(path: str) -> Iterable[str]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            yield value
