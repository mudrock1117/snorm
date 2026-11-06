"""Checkpoint extraction and validation for legacy SNorm detector snapshots."""
from __future__ import absolute_import

from collections import OrderedDict

import torch


_STATE_KEYS = ("model", "state_dict", "model_state_dict", "net", "network", "weights")
_CRITICAL_PREFIXES = (
    "teacher.backbone_0.",
    "teacher.backbone_1.",
    "teacher.backbone_2.",
    "teacher.backbone_3.",
    "teacher.rpn_norm.",
    "teacher.box_head_norm.",
    "teacher.proj_norm.",
)


def _is_state_dict(value):
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(key, str) for key in value.keys()) and any(
        torch.is_tensor(item) for item in value.values()
    )


def extract_state_dict(checkpoint):
    """Return (state_dict, source_key) for common historical formats."""
    if _is_state_dict(checkpoint):
        state = checkpoint
        source = "<root>"
    elif isinstance(checkpoint, dict):
        state = None
        source = None
        for key in _STATE_KEYS:
            candidate = checkpoint.get(key)
            if _is_state_dict(candidate):
                state = candidate
                source = key
                break
        if state is None:
            available = sorted(str(key) for key in checkpoint.keys())
            raise ValueError(
                "checkpoint has no recognized model state; available keys: %s"
                % ", ".join(available)
            )
    else:
        raise TypeError("checkpoint must be a state-dict or a dictionary container")

    keys = list(state.keys())
    if keys and all(key.startswith("module.") for key in keys):
        state = OrderedDict((key[7:], value) for key, value in state.items())
        source = source + " (stripped module.)"
    return state, source


def load_detector_checkpoint(model, checkpoint_path, allow_critical_missing=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, source = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    critical_missing = [
        key for key in missing if any(key.startswith(prefix) for prefix in _CRITICAL_PREFIXES)
    ]
    print("checkpoint_state_source:", source)
    print("missing_keys:", missing)
    print("unexpected_keys:", unexpected)
    if critical_missing:
        message = (
            "critical detector parameters are missing; refusing diagnostic inference: %s"
            % ", ".join(critical_missing)
        )
        if not allow_critical_missing:
            raise RuntimeError(message)
        print("WARNING:", message)
    return {
        "checkpoint": checkpoint,
        "source": source,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "critical_missing_keys": critical_missing,
    }
