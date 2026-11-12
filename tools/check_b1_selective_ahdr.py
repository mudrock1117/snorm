#!/usr/bin/env python
"""GPU audit for B1 Protected Small-Anchor + Selective-AHDR."""
from __future__ import print_function

import argparse
import math
import random

import numpy as np
import torch

from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.modeling.build import build_detectors
from diagnostics.a3_protected import (
    A3_TRAINABLE_NAMES,
    BBOX_NEW_CHANNELS,
    CLS_NEW_CHANNELS,
    ProtectedSmallAnchorStep,
)
from diagnostics.b1_protected import (
    B1_GATE_LR,
    B1_OUTPUT_LR,
    audit_b1_trainable_parameters,
    configure_b1_trainable_parameters,
)
from diagnostics.small_anchor import load_small_anchor_checkpoint


def snapshot_module(module):
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def module_max_delta(module, before):
    maximum = 0.0
    for parameter, old in zip(module.parameters(), before):
        maximum = max(
            maximum,
            float((parameter.detach().cpu() - old).abs().max().item()),
        )
    return maximum


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=False
    )
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            squared += float(gradient.detach().pow(2).sum().item())
    return math.sqrt(squared)


def selected_delta(parameter, before, channels):
    return float(
        (
            parameter.detach()[list(channels)] - before
        ).abs().max().item()
    )


def move_batch(batch, device):
    if len(batch) != 8:
        raise RuntimeError("B1 audit expects the real eight-element batch")
    images, img_metas, targets, sc_images, sc_trans, tr_images, tr_trans, g_masks = batch
    images = images.to(device)
    tr_images = tr_images.to(device)
    targets = [target.to(device) for target in targets]
    return images, img_metas, targets, tr_images


def b1_forward(model, images, img_metas, targets, degraded):
    return model(
        images,
        img_metas,
        targets,
        mode_training=13,
        stpd_images=degraded,
        is_update=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    expected_checkpoint = (
        "/home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/"
        "s0_dbias_color_E2_zdh/E2/best.pth"
    )
    if args.checkpoint != expected_checkpoint:
        raise RuntimeError(
            "B1 audit only accepts the fixed E2 baseline checkpoint: %s"
            % expected_checkpoint
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if not bool(cfg.SELECTIVE_AHDR.ENABLED):
        raise RuntimeError("B1 audit requires SELECTIVE_AHDR.ENABLED=True")
    if tuple(cfg.MODEL.RPN.ANCHOR_SIZES) != (32, 64, 128, 256, 512):
        raise RuntimeError("B1 audit requires the audited small-anchor layout")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    random.seed(20261112)
    np.random.seed(20261112)
    torch.manual_seed(20261112)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20261112)

    loader = build_data_loaders(
        cfg.DATASETS.TRAINS,
        transforms=cfg.INPUT.TRANSFORMS_TRAIN,
        is_train=True,
        distributed=False,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(loader))
    images, img_metas, targets, degraded = move_batch(batch, device)

    model = build_detectors(cfg).to(device)
    load_result = load_small_anchor_checkpoint(
        model,
        args.checkpoint,
        old_anchor_sizes=(64, 128, 256, 512),
        new_anchor_sizes=(32, 64, 128, 256, 512),
    )
    expected_missing = set(
        name for name in model.state_dict().keys()
        if name.startswith("teacher.selective_ahdr_gate.")
    )
    if set(load_result["missing_keys"]) != expected_missing:
        raise RuntimeError(
            "B1 checkpoint missing keys are not exactly the new gate: %s"
            % load_result["missing_keys"]
        )
    if load_result["unexpected_keys"]:
        raise RuntimeError(
            "B1 checkpoint has unexpected keys: %s"
            % load_result["unexpected_keys"]
        )

    outputs, gate_parameters, gate_names = (
        configure_b1_trainable_parameters(model)
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": outputs,
                "lr": B1_OUTPUT_LR,
                "name": "b1_new_32_anchor_output",
            },
            {
                "params": gate_parameters,
                "lr": B1_GATE_LR,
                "name": "b1_selective_ahdr_gate",
            },
        ],
        lr=B1_OUTPUT_LR,
        betas=(0.9, 0.999),
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    audit_info = audit_b1_trainable_parameters(model, optimizer)
    controller = ProtectedSmallAnchorStep(
        model, extra_trainable_names=gate_names
    )
    model.train()

    gate_list = list(model.teacher.selective_ahdr_gate.parameters())

    optimizer.zero_grad()
    detection_losses, _ = b1_forward(
        model, images, img_metas, targets, degraded
    )
    detection_only = sum(
        value for key, value in detection_losses.items()
        if not key.startswith("b1_")
    )
    detection_gate_gradient = gradient_norm(detection_only, gate_list)
    if detection_gate_gradient <= 0:
        raise RuntimeError("detection loss alone has no gate gradient")

    optimizer.zero_grad()
    consistency_losses, _ = b1_forward(
        model, images, img_metas, targets, degraded
    )
    consistency_only = consistency_losses["b1_consistency"]
    consistency_gate_gradient = gradient_norm(
        consistency_only, gate_list
    )
    if consistency_gate_gradient <= 0:
        raise RuntimeError("consistency loss alone has no gate gradient")

    frozen_modules = {
        "backbone_0": model.teacher.backbone_0,
        "backbone_1": model.teacher.backbone_1,
        "backbone_2": model.teacher.backbone_2,
        "backbone_3": model.teacher.backbone_3,
        "render_net": model.teacher.proj_norm.render_net,
        "proj_norm": model.teacher.proj_norm,
        "rpn_norm_shared_conv": model.teacher.rpn_norm.conv,
        "box_head_norm": model.teacher.box_head_norm,
        "rpn_eval": model.teacher.rpn_eval,
    }
    frozen_before = {
        name: snapshot_module(module) for name, module in frozen_modules.items()
    }
    gate_before = snapshot_module(model.teacher.selective_ahdr_gate)

    optimizer.zero_grad()
    losses, forward_outputs = b1_forward(
        model, images, img_metas, targets, degraded
    )
    detection_raw = sum(
        value for key, value in losses.items()
        if not key.startswith("b1_")
    )
    auxiliary = sum(
        value for key, value in losses.items()
        if key.startswith("b1_")
    )
    total = sum(losses.values())
    total.backward()

    gate_gradient_sq = 0.0
    for parameter in gate_list:
        if parameter.grad is not None:
            gate_gradient_sq += float(
                parameter.grad.detach().pow(2).sum().item()
            )
    gate_gradient = math.sqrt(gate_gradient_sq)
    if gate_gradient <= 0:
        raise RuntimeError("B1 total loss has no gate gradient")

    named = dict(model.named_parameters())
    new_before = {}
    raw_new_gradient = {}
    for name in A3_TRAINABLE_NAMES:
        parameter = named[name]
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        new_before[name] = parameter.detach()[list(channels)].clone()
        raw_new_gradient[name] = float(
            parameter.grad.detach()[list(channels)].abs().max().item()
        )
    if max(raw_new_gradient.values()) <= 0:
        raise RuntimeError("B1 total loss has no new-anchor gradient")

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
            raise RuntimeError("B1 old-anchor gradient is nonzero")

    gate = forward_outputs["b1"]["gate"].detach()
    coefficient = forward_outputs["b1"]["coefficient"].detach()
    initialization_diff = forward_outputs["b1"]["initialization_diff"].detach()
    if not torch.isfinite(gate).all():
        raise RuntimeError("B1 gate contains NaN/Inf")
    if float(gate.min().item()) < 0 or float(gate.max().item()) > 1:
        raise RuntimeError("B1 gate is outside [0,1]")
    mean_abs_diff = float(initialization_diff.abs().mean().item())
    max_abs_diff = float(initialization_diff.abs().max().item())
    legacy = forward_outputs["b1"]["legacy_feature"].detach()
    relative_l2 = float(
        initialization_diff.pow(2).sum().sqrt().item()
        / (legacy.pow(2).sum().sqrt().item() + 1e-12)
    )
    if max_abs_diff > 1e-5:
        raise RuntimeError(
            "B1 initialization does not preserve legacy output: max=%s"
            % max_abs_diff
        )

    optimizer.step()
    old_delta = controller.after_step(optimizer, protected)
    new_delta = {}
    for name, before in new_before.items():
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        new_delta[name] = selected_delta(named[name], before, channels)
    if max(new_delta.values()) <= 0:
        raise RuntimeError("B1 new-anchor outputs did not update")
    gate_delta = module_max_delta(
        model.teacher.selective_ahdr_gate, gate_before
    )
    if gate_delta <= 0:
        raise RuntimeError("B1 gate did not update")

    frozen_delta = {
        name: module_max_delta(module, frozen_before[name])
        for name, module in frozen_modules.items()
    }
    for name, delta in frozen_delta.items():
        if delta != 0:
            raise RuntimeError(
                "B1 frozen module changed: %s delta=%s" % (name, delta)
            )

    print("[B1 CHECKPOINT]")
    print("checkpoint =", args.checkpoint)
    print("old anchors =", (64, 128, 256, 512))
    print("new anchors =", (32, 64, 128, 256, 512))
    print("new cls =", list(CLS_NEW_CHANNELS))
    print("new bbox =", "[0:4,20:24,40:44]")
    print("missing_keys =", load_result["missing_keys"])
    print("unexpected_keys =", load_result["unexpected_keys"])
    print("optimizer whitelist =", audit_info["trainable_names"])
    print("optimizer group LRs =", [group["lr"] for group in optimizer.param_groups])
    print("raw detection loss =", float(detection_raw.detach().item()))
    print("raw/weighted loss entries =", {
        key: float(value.detach().item()) for key, value in losses.items()
    })
    print("weighted auxiliary total =", float(auxiliary.detach().item()))
    print("detection-only gate gradient norm =", detection_gate_gradient)
    print("consistency-only gate gradient norm =", consistency_gate_gradient)
    print("total gate gradient norm =", gate_gradient)
    print("new-anchor raw gradient max =", raw_new_gradient)
    print("old-anchor gradient after mask =", old_gradient)
    print("old-anchor parameter delta =", old_delta)
    print("new-anchor parameter delta =", new_delta)
    print("gate parameter delta =", gate_delta)
    print("frozen module delta =", frozen_delta)
    print("gate min/max/mean/std =", (
        float(gate.min().item()), float(gate.max().item()),
        float(gate.mean().item()), float(gate.std().item()),
    ))
    print("effective coefficient mean/std =", (
        float(coefficient.mean().item()), float(coefficient.std().item()),
    ))
    print("initialization mean_abs_diff =", mean_abs_diff)
    print("initialization max_abs_diff =", max_abs_diff)
    print("initialization relative_L2_diff =", relative_l2)
    if float(auxiliary.detach().item()) > float(detection_raw.detach().item()):
        print("WARNING: weighted auxiliary loss exceeds detection loss")
    print("[B1 AUDIT PASS]")
    print("PASS")


if __name__ == "__main__":
    main()
