"""Reversible probe for SNorm compute_norm_parameters epoch-boundary updates."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .tensor_metrics import tensor_metrics


def _snapshot(snorm) -> Dict[str, object]:
    return {
        "memory_index": int(snorm.memory_index),
        "memory_aplpha_a": np.array(snorm.memory_aplpha_a, copy=True),
        "memory_aplpha_u": np.array(snorm.memory_aplpha_u, copy=True),
        "alpha_ba": snorm.alpha_ba.detach().clone(),
        "alpha_bu": snorm.alpha_bu.detach().clone(),
    }


def _restore(snorm, state: Dict[str, object]) -> None:
    snorm.memory_index = int(state["memory_index"])
    snorm.memory_aplpha_a[...] = state["memory_aplpha_a"]
    snorm.memory_aplpha_u[...] = state["memory_aplpha_u"]
    snorm.alpha_ba.data.copy_(state["alpha_ba"].to(snorm.alpha_ba.device))
    snorm.alpha_bu.data.copy_(state["alpha_bu"].to(snorm.alpha_bu.device))


def _state_summary(state: Dict[str, object]) -> Dict[str, object]:
    index = int(state["memory_index"])
    return {
        "memory_index": index,
        "memory_empty": index == 0,
        "empty_memory_overwrites_with_zero": index == 0,
        "memory_aplpha_a_used": np.asarray(state["memory_aplpha_a"])[:index].tolist(),
        "memory_aplpha_u_used": np.asarray(state["memory_aplpha_u"])[:index].tolist(),
        "alpha_ba": state["alpha_ba"].cpu().tolist(),
        "alpha_bu": state["alpha_bu"].cpu().tolist(),
    }


@torch.no_grad()
def probe_epoch_boundary(
    snorm,
    output_dir: str,
    render_callback: Optional[Callable[[], torch.Tensor]] = None,
    detection_callback: Optional[Callable[[], object]] = None,
    apply_update: bool = False,
) -> Path:
    """Probe update effects and restore state unless apply_update is explicit.

    Callbacks must use the same fixed input. Detection callback output is serialized
    only for boxes/scores/labels when that public structure is available.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original = _snapshot(snorm)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    before_render = render_callback() if render_callback is not None else None
    before_detection = detection_callback() if detection_callback is not None else None
    try:
        snorm.compute_norm_parameters(device=snorm.alpha_ba.device)
        updated = _snapshot(snorm)
        after_render = render_callback() if render_callback is not None else None
        after_detection = detection_callback() if detection_callback is not None else None
        report = {
            "before": _state_summary(original),
            "after": _state_summary(updated),
            "render_comparison": None,
            "detection_comparison": None,
            "state_will_be_restored": not apply_update,
        }
        if before_render is not None and after_render is not None:
            report["render_comparison"] = {
                "before": tensor_metrics(before_render),
                "after": tensor_metrics(after_render, reference=before_render),
            }
        if before_detection is not None and after_detection is not None:
            report["detection_comparison"] = {
                "before": _serialize_detection(before_detection),
                "after": _serialize_detection(after_detection),
            }
        path = output / "epoch_boundary_probe.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save(
            {
                "before_alpha_ba": original["alpha_ba"].cpu(),
                "before_alpha_bu": original["alpha_bu"].cpu(),
                "after_alpha_ba": updated["alpha_ba"].cpu(),
                "after_alpha_bu": updated["alpha_bu"].cpu(),
            },
            output / "epoch_boundary_probe.pt",
        )
        return path
    finally:
        if not apply_update:
            _restore(snorm, original)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _serialize_detection(value):
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if not isinstance(value, dict):
        return {"unavailable": "callback did not return a detection dictionary"}
    result = {}
    for key in ("boxes", "scores", "labels"):
        item = value.get(key)
        if torch.is_tensor(item):
            result[key] = item.detach().cpu().tolist()
    return result
