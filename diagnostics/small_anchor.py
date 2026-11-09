"""Auditable E2 checkpoint expansion for the A small-anchor experiment."""
from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch

from .checkpoint import extract_state_dict


RPN_HEAD_SUFFIXES = (
    "cls_logits.weight",
    "cls_logits.bias",
    "bbox_pred.weight",
    "bbox_pred.bias",
)
RPN_PREFIXES = ("teacher.rpn_norm.", "teacher.rpn_eval.")
CRITICAL_PREFIXES = (
    "teacher.backbone_0.",
    "teacher.backbone_1.",
    "teacher.backbone_2.",
    "teacher.backbone_3.",
    "teacher.proj_norm.",
    "teacher.rpn_norm.",
    "teacher.box_head_norm.",
)


def anchor_channel_index(scale_index, ratio_index, scale_count):
    """Ratio-major, scale-minor order proven by AnchorGenerator.generate_anchors."""
    return int(ratio_index) * int(scale_count) + int(scale_index)


def _copy_anchor_channels(old_tensor, new_tensor, old_sizes, new_sizes,
                          ratio_count, bbox):
    result = new_tensor.detach().cpu().clone()
    copied = []
    for ratio_index in range(int(ratio_count)):
        for old_scale_index, size in enumerate(old_sizes):
            if size not in new_sizes:
                raise RuntimeError(
                    "old anchor size %s is absent from new anchor sizes %s"
                    % (size, new_sizes)
                )
            new_scale_index = new_sizes.index(size)
            old_anchor = anchor_channel_index(
                old_scale_index, ratio_index, len(old_sizes)
            )
            new_anchor = anchor_channel_index(
                new_scale_index, ratio_index, len(new_sizes)
            )
            if bbox:
                old_slice = slice(old_anchor * 4, old_anchor * 4 + 4)
                new_slice = slice(new_anchor * 4, new_anchor * 4 + 4)
                result[new_slice].copy_(old_tensor[old_slice])
            else:
                result[new_anchor].copy_(old_tensor[old_anchor])
            copied.append((ratio_index, int(size), old_anchor, new_anchor))
    return result, copied


def _validate_rpn_module(module, new_sizes, aspect_ratios):
    actual_sizes = tuple(int(value) for value in module.anchor_generator.sizes[0])
    actual_ratios = tuple(float(value) for value in module.anchor_generator.aspect_ratios[0])
    if actual_sizes != tuple(new_sizes):
        raise RuntimeError(
            "RPN AnchorGenerator sizes %s do not match requested %s"
            % (actual_sizes, tuple(new_sizes))
        )
    if actual_ratios != tuple(float(value) for value in aspect_ratios):
        raise RuntimeError(
            "RPN AnchorGenerator ratios %s do not match config %s"
            % (actual_ratios, tuple(aspect_ratios))
        )
    expected = len(new_sizes) * len(aspect_ratios)
    if module.cls_logits.out_channels != expected:
        raise RuntimeError("RPN cls_logits channels do not match anchors")
    if module.bbox_pred.out_channels != expected * 4:
        raise RuntimeError("RPN bbox_pred channels do not match anchors")


def load_small_anchor_checkpoint(
        model,
        checkpoint_path,
        old_anchor_sizes=(64, 128, 256, 512),
        new_anchor_sizes=(32, 64, 128, 256, 512)):
    """Load E2 and expand only RPN anchor-dependent output channels.

    Existing RPN construction initializes all new head weights with normal
    std=0.01 and bias=0. This function starts from that initialized model state,
    copies every shape-compatible E2 tensor, and overwrites only the old anchor
    channel positions in the expanded heads.
    """
    old_sizes = tuple(int(value) for value in old_anchor_sizes)
    new_sizes = tuple(int(value) for value in new_anchor_sizes)
    ratios = tuple(float(value) for value in model.cfg.MODEL.RPN.ASPECT_RATIOS)
    if old_sizes != (64, 128, 256, 512):
        raise RuntimeError("A experiment requires the audited E2 anchor sizes")
    if new_sizes != tuple(int(v) for v in model.cfg.MODEL.RPN.ANCHOR_SIZES):
        raise RuntimeError("new anchor sizes do not match model config")
    if new_sizes != (32, 64, 128, 256, 512):
        raise RuntimeError("A experiment requires exactly one new 32 anchor")

    _validate_rpn_module(model.teacher.rpn_norm, new_sizes, ratios)
    _validate_rpn_module(model.teacher.rpn_eval, new_sizes, ratios)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state, source = extract_state_dict(checkpoint)
    model_state = model.state_dict()
    adapted = OrderedDict()
    copied_parameters = []
    expanded_parameters = []
    unexpected = sorted(set(checkpoint_state.keys()) - set(model_state.keys()))
    missing = []

    for key, target in model_state.items():
        if key not in checkpoint_state:
            adapted[key] = target
            missing.append(key)
            continue
        source_tensor = checkpoint_state[key]
        if tuple(source_tensor.shape) == tuple(target.shape):
            adapted[key] = source_tensor
            copied_parameters.append(key)
            continue

        is_rpn_head = any(
            key == prefix + suffix
            for prefix in RPN_PREFIXES for suffix in RPN_HEAD_SUFFIXES
        )
        if not is_rpn_head:
            raise RuntimeError(
                "unexpected checkpoint shape mismatch for %s: %s -> %s"
                % (key, tuple(source_tensor.shape), tuple(target.shape))
            )
        bbox = ".bbox_pred." in key
        old_anchor_count = len(old_sizes) * len(ratios)
        new_anchor_count = len(new_sizes) * len(ratios)
        expected_old = old_anchor_count * (4 if bbox else 1)
        expected_new = new_anchor_count * (4 if bbox else 1)
        if source_tensor.size(0) != expected_old or target.size(0) != expected_new:
            raise RuntimeError(
                "cannot audit anchor expansion for %s: first dims %d -> %d, "
                "expected %d -> %d"
                % (
                    key, source_tensor.size(0), target.size(0),
                    expected_old, expected_new,
                )
            )
        expanded, mapping = _copy_anchor_channels(
            source_tensor, target, old_sizes, new_sizes, len(ratios), bbox
        )
        adapted[key] = expanded
        expanded_parameters.append({
            "key": key,
            "old_shape": tuple(source_tensor.shape),
            "new_shape": tuple(target.shape),
            "mapping": mapping,
        })

    critical_missing = [
        key for key in missing
        if any(key.startswith(prefix) for prefix in CRITICAL_PREFIXES)
    ]
    if critical_missing:
        raise RuntimeError(
            "critical E2 parameters are missing: %s"
            % ", ".join(critical_missing)
        )
    incompatible = model.load_state_dict(adapted, strict=True)
    if getattr(incompatible, "missing_keys", None):
        raise RuntimeError("adapted state unexpectedly has missing keys")
    if getattr(incompatible, "unexpected_keys", None):
        raise RuntimeError("adapted state unexpectedly has unexpected keys")

    initialized_per_ratio = []
    if expanded_parameters:
        initialized_per_ratio = [
            anchor_channel_index(0, ratio_index, len(new_sizes))
            for ratio_index in range(len(ratios))
        ]
    print("[SMALL-ANCHOR LOAD]")
    print("checkpoint_state_source =", source)
    print("old anchors =", old_sizes)
    print("new anchors =", new_sizes)
    print("channel ordering = ratio-major, scale-minor")
    print("copied parameters =", len(copied_parameters))
    print("expanded parameters =", [
        item["key"] for item in expanded_parameters
    ])
    print("newly initialized channels =", initialized_per_ratio)
    print("new channel init =", (
        "existing RPN normal std=0.01, bias=0"
        if expanded_parameters else "not applicable (checkpoint shapes already match)"
    ))
    print("missing_keys =", missing)
    print("unexpected_keys =", unexpected)
    return {
        "checkpoint": checkpoint,
        "source": source,
        "copied_parameters": copied_parameters,
        "expanded_parameters": expanded_parameters,
        "newly_initialized_anchor_channels": initialized_per_ratio,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
