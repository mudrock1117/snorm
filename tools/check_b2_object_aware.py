#!/usr/bin/env python
"""Single-GPU audit for B2 Object-Aware Selective-AHDR."""
from __future__ import absolute_import, print_function

import argparse
import math
import random
import sys

import numpy as np
import torch

from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.modeling.build import build_detectors
from detection.modeling.norm.object_aware_ahdr import (
    object_aware_gate_losses,
)
from diagnostics.a3_protected import (
    A3_TRAINABLE_NAMES,
    BBOX_NEW_CHANNELS,
    CLS_NEW_CHANNELS,
    ProtectedSmallAnchorStep,
)
from diagnostics.b2_protected import (
    B2_GATE_LR,
    B2_OUTPUT_LR,
    audit_b2_trainable_parameters,
    configure_b2_trainable_parameters,
)
from diagnostics.small_anchor import load_small_anchor_checkpoint


EXPECTED_CHECKPOINT = (
    "/home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/"
    "s0_dbias_color_E2_zdh/E2/best.pth"
)
DETECTION_KEYS = (
    "rpn_cls_loss_w", "rpn_reg_loss_w",
    "rcnn_cls_loss_w", "rcnn_reg_loss_w",
)


def move_batch(batch, device):
    images, img_metas, targets, _, _, degraded, _, _ = batch
    return (
        images.to(device), img_metas,
        [target.to(device) for target in targets],
        degraded.to(device),
    )


def has_small_target(targets, img_metas, threshold):
    for target, meta in zip(targets, img_metas):
        width, height = meta["img_shape"]
        valid_area = max(float(width) * float(height), 1.0)
        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue
        areas = (
            (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
            * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        )
        if bool((areas / valid_area < float(threshold)).any().item()):
            return True
    return False


def snapshot(module):
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def module_delta(module, before):
    result = 0.0
    for parameter, old in zip(module.parameters(), before):
        result = max(
            result,
            float((parameter.detach().cpu() - old).abs().max().item()),
        )
    return result


def gate_gradient_norm(model):
    squared = 0.0
    for parameter in model.teacher.selective_ahdr_gate.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(squared)


def forward_b2(model, images, img_metas, targets, degraded):
    return model(
        images, img_metas, targets, mode_training=14,
        stpd_images=degraded, is_update=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.checkpoint != EXPECTED_CHECKPOINT:
        raise RuntimeError("B2 audit requires fixed E2 checkpoint")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if not bool(cfg.B2_OBJECT_AWARE.ENABLED):
        raise RuntimeError("B2_OBJECT_AWARE must be enabled")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    random.seed(20261113)
    np.random.seed(20261113)
    torch.manual_seed(20261113)

    loader = build_data_loaders(
        cfg.DATASETS.TRAINS,
        transforms=cfg.INPUT.TRANSFORMS_TRAIN,
        is_train=True, distributed=False, batch_size=1,
        num_workers=0, shuffle=False,
    )
    selected_batch = None
    for batch_index, batch in enumerate(loader):
        _, metas, batch_targets, _, _, _, _, _ = batch
        if has_small_target(
                batch_targets, metas,
                cfg.B2_OBJECT_AWARE.SMALL_AREA_RATIO):
            selected_batch = batch
            break
        if batch_index >= 999:
            break
    if selected_batch is None:
        raise RuntimeError("B2 audit could not find source small-object GT")
    images, img_metas, targets, degraded = move_batch(
        selected_batch, device
    )

    model = build_detectors(cfg).to(device)
    load_result = load_small_anchor_checkpoint(
        model, args.checkpoint,
        old_anchor_sizes=(64, 128, 256, 512),
        new_anchor_sizes=(32, 64, 128, 256, 512),
    )
    expected_missing = set(
        name for name in model.state_dict().keys()
        if name.startswith("teacher.selective_ahdr_gate.")
    )
    failures = []
    if set(load_result["missing_keys"]) != expected_missing:
        failures.append("checkpoint expansion missing-key mismatch")
    if load_result["unexpected_keys"]:
        failures.append("checkpoint expansion has unexpected keys")

    outputs, gate_parameters, gate_names = (
        configure_b2_trainable_parameters(model)
    )
    optimizer = torch.optim.Adam([
        {
            "params": outputs, "lr": B2_OUTPUT_LR,
            "name": "b2_new_32_anchor_output",
        },
        {
            "params": gate_parameters, "lr": B2_GATE_LR,
            "name": "b2_object_aware_gate",
        },
    ], lr=B2_OUTPUT_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    audit_info = audit_b2_trainable_parameters(model, optimizer)
    controller = ProtectedSmallAnchorStep(
        model, extra_trainable_names=gate_names
    )
    model.train()

    optimizer.zero_grad()
    detection_losses, _ = forward_b2(
        model, images, img_metas, targets, degraded
    )
    detection = sum(detection_losses[key] for key in DETECTION_KEYS)
    detection.backward()
    detection_gate_grad = gate_gradient_norm(model)
    if detection_gate_grad <= 0:
        failures.append("detection-only Gate gradient is not positive")

    optimizer.zero_grad()
    small_losses, small_outputs = forward_b2(
        model, images, img_metas, targets, degraded
    )
    small_losses["b2_small_floor"].backward()
    small_gate_grad = gate_gradient_norm(model)
    if small_gate_grad <= 0:
        failures.append("small-floor-only Gate gradient is not positive")

    payload = small_outputs["b2"]
    gate = payload["gate"].detach()
    coefficient = payload["coefficient"].detach()
    masks = payload["masks"]
    init_diff = payload["initialization_diff"].detach()
    max_init_diff = float(init_diff.abs().max().item())
    if abs(float(gate.mean().item()) - 0.5) > 1e-6:
        failures.append("Gate initialization is not 0.5")
    if abs(float(coefficient.mean().item()) - 0.6) > 1e-6:
        failures.append("alpha*M initialization is not 0.6")
    if max_init_diff >= 1e-6:
        failures.append("B2 initialization differs from legacy")

    object_mask = masks["object_mask"]
    small_support = masks["small_support"]
    for name in (
            "object_mask", "small_support", "small_weighted",
            "background_mask"):
        value = masks[name]
        if not bool(torch.isfinite(value).all().item()):
            failures.append("%s is not finite" % name)
    if object_mask.shape != gate.shape:
        failures.append("object mask/Gate shape mismatch")
    if float(object_mask.sum().item()) <= 0:
        failures.append("source object mask is empty")
    if float(small_support.sum().item()) <= 0:
        failures.append("source small-object support is empty")
    if (
            float(object_mask.min().item()) < 0
            or float(object_mask.max().item()) > 1):
        failures.append("object mask outside [0,1]")

    first_box = targets[0]["boxes"][0].detach()
    valid_w, valid_h = img_metas[0]["img_shape"]
    padded_h, padded_w = degraded.shape[-2:]
    feature_h, feature_w = gate.shape[-2:]
    center_x = float((first_box[0] + first_box[2]) * 0.5)
    center_y = float((first_box[1] + first_box[3]) * 0.5)
    feature_x = min(
        feature_w - 1,
        max(0, int(center_x * float(feature_w) / float(padded_w))),
    )
    feature_y = min(
        feature_h - 1,
        max(0, int(center_y * float(feature_h) / float(padded_h))),
    )
    projected_value = float(
        object_mask[0, 0, feature_y, feature_x].item()
    )
    if projected_value <= 0:
        failures.append("projected GT center is outside object mask")

    raw_losses = payload["raw_losses"]
    for name, value in raw_losses.items():
        if not bool(torch.isfinite(value).all().item()):
            failures.append("non-finite B2 loss: %s" % name)

    perturbed_gate = torch.full_like(gate, 0.48, requires_grad=True)
    perturbed = object_aware_gate_losses(
        perturbed_gate, masks,
        object_floor=cfg.B2_OBJECT_AWARE.OBJECT_FLOOR,
        small_object_floor=cfg.B2_OBJECT_AWARE.SMALL_OBJECT_FLOOR,
    )
    if float(perturbed["object_floor"].item()) <= 0:
        failures.append("perturbed object floor is not positive")
    if float(perturbed["small_floor"].item()) <= 0:
        failures.append("perturbed small floor is not positive")

    frozen_modules = {
        "backbone_0": model.teacher.backbone_0,
        "backbone_1": model.teacher.backbone_1,
        "backbone_2": model.teacher.backbone_2,
        "backbone_3": model.teacher.backbone_3,
        "proj_norm": model.teacher.proj_norm,
        "render_net": model.teacher.proj_norm.render_net,
        "rpn_norm.conv": model.teacher.rpn_norm.conv,
        "box_head_norm": model.teacher.box_head_norm,
        "box_head_eval": model.teacher.box_head_eval,
        "rpn_eval": model.teacher.rpn_eval,
    }
    frozen_before = {
        name: snapshot(module) for name, module in frozen_modules.items()
    }
    gate_before = snapshot(model.teacher.selective_ahdr_gate)

    optimizer.zero_grad()
    losses, _ = forward_b2(model, images, img_metas, targets, degraded)
    total = sum(losses.values())
    total.backward()
    named = dict(model.named_parameters())
    new_before = {}
    new_gradient = {}
    for name in A3_TRAINABLE_NAMES:
        parameter = named[name]
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        new_before[name] = parameter.detach()[list(channels)].clone()
        new_gradient[name] = float(
            parameter.grad.detach()[list(channels)].abs().max().item()
        )
    if max(new_gradient.values()) <= 0:
        failures.append("new-anchor raw gradient is not positive")

    protected = controller.before_step()
    old_gradient = {}
    for name, parameter in controller.parameters.items():
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        mask = torch.ones(
            parameter.size(0), dtype=torch.bool, device=parameter.device
        )
        mask[list(channels)] = False
        old_gradient[name] = float(
            parameter.grad.detach()[mask].abs().max().item()
        )
        if old_gradient[name] != 0:
            failures.append("old-anchor masked gradient is nonzero")

    optimizer.step()
    old_delta = controller.after_step(optimizer, protected)
    new_delta = {}
    for name, before in new_before.items():
        parameter = named[name]
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        new_delta[name] = float(
            (
                parameter.detach()[list(channels)] - before
            ).abs().max().item()
        )
    gate_delta = module_delta(model.teacher.selective_ahdr_gate, gate_before)
    frozen_delta = {
        name: module_delta(module, frozen_before[name])
        for name, module in frozen_modules.items()
    }
    if max(new_delta.values()) <= 0:
        failures.append("new-anchor parameter delta is not positive")
    if gate_delta <= 0:
        failures.append("Gate parameter delta is not positive")
    if any(value != 0 for value in frozen_delta.values()):
        failures.append("a frozen module changed")

    print("[B2 AUDIT]")
    print("checkpoint missing/unexpected =",
          load_result["missing_keys"], load_result["unexpected_keys"])
    print("trainable whitelist =", audit_info["trainable_names"])
    print("gate parameter count =", audit_info["gate_parameter_count"])
    print("gate mean =", float(gate.mean().item()))
    print("alpha*M mean =", float(coefficient.mean().item()))
    print("initialization max_abs_diff =", max_init_diff)
    print("valid image (W,H) =", (valid_w, valid_h))
    print("padded image (H,W) =", (padded_h, padded_w))
    print("Gate/mask shape =", tuple(gate.shape))
    print("GT center feature cell/value =",
          (feature_y, feature_x, projected_value))
    print("object/small coverage =",
          float(object_mask.mean().item()),
          float(small_support.mean().item()))
    print("objects/small objects =",
          masks["num_objects"], masks["num_small_objects"])
    print("raw B2 losses =", {
        name: float(value.item()) for name, value in raw_losses.items()
    })
    print("perturbed floors =", {
        name: float(value.item()) for name, value in perturbed.items()
    })
    print("detection-only Gate grad =", detection_gate_grad)
    print("small-only Gate grad =", small_gate_grad)
    print("new-anchor raw grad =", new_gradient)
    print("old-anchor masked grad =", old_gradient)
    print("old/new/Gate delta =", old_delta, new_delta, gate_delta)
    print("frozen module delta =", frozen_delta)
    if failures:
        print("[B2 AUDIT FAIL]")
        print("failure_reasons =", failures)
        return 1
    print("[B2 AUDIT PASS]")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
