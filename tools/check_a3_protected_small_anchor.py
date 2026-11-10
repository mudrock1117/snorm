#!/usr/bin/env python
"""Audit A3 checkpoint mapping, gradients, optimizer isolation, and updates."""
from __future__ import print_function

import argparse

import torch

from detection.config import cfg
from detection.modeling.build import build_detectors
from diagnostics.a3_protected import (
    BBOX_NEW_CHANNELS,
    CLS_NEW_CHANNELS,
    ProtectedSmallAnchorStep,
    audit_a3_trainable_parameters,
    configure_a3_trainable_parameters,
    parse_rpn_training_output,
)
from diagnostics.small_anchor import load_small_anchor_checkpoint


def snapshot_module(module):
    return [
        parameter.detach().cpu().clone()
        for parameter in module.parameters()
    ]


def module_max_delta(module, before):
    maximum = 0.0
    for parameter, old in zip(module.parameters(), before):
        delta = (parameter.detach().cpu() - old).abs().max().item()
        maximum = max(maximum, float(delta))
    return maximum


def allowed_delta(parameter, before, channels):
    current = parameter.detach()[list(channels)]
    return float((current - before).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = build_detectors(cfg).to(device)
    load_small_anchor_checkpoint(
        model,
        args.checkpoint,
        old_anchor_sizes=(64, 128, 256, 512),
        new_anchor_sizes=tuple(cfg.MODEL.RPN.ANCHOR_SIZES),
    )
    parameters = configure_a3_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        parameters,
        lr=cfg.SOLVER.LR,
        betas=(0.9, 0.999),
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    audit_a3_trainable_parameters(model, optimizer)
    controller = ProtectedSmallAnchorStep(model)

    teacher = model.teacher
    frozen_modules = {
        "backbone_0": teacher.backbone_0,
        "backbone_1": teacher.backbone_1,
        "backbone_2": teacher.backbone_2,
        "backbone_3": teacher.backbone_3,
        "box_head_norm": teacher.box_head_norm,
    }
    frozen_before = {
        name: snapshot_module(module)
        for name, module in frozen_modules.items()
    }

    rpn = teacher.rpn_norm
    rpn.train()
    feature = torch.zeros(
        1, rpn.conv.in_channels, 8, 8, device=device
    )
    img_metas = [{"img_shape": (128, 128)}]
    targets = [{
        "boxes": torch.tensor(
            [[17.0, 17.0, 49.0, 49.0]], device=device
        ),
        "labels": torch.ones(1, dtype=torch.long, device=device),
    }]
    rpn_output = rpn(
        None, feature, img_metas, targets
    )
    proposals, losses, cls_logits, bbox_pred = parse_rpn_training_output(
        rpn_output
    )
    loss = losses["rpn_cls_loss"] + losses["rpn_reg_loss"]
    optimizer.zero_grad()
    loss.backward()

    named = dict(model.named_parameters())
    raw_new_gradient = {}
    for name in (
            "teacher.rpn_norm.cls_logits.weight",
            "teacher.rpn_norm.cls_logits.bias"):
        parameter = named[name]
        raw_new_gradient[name] = float(
            parameter.grad.detach()[list(CLS_NEW_CHANNELS)].abs().max().item()
        )
    for name in (
            "teacher.rpn_norm.bbox_pred.weight",
            "teacher.rpn_norm.bbox_pred.bias"):
        parameter = named[name]
        raw_new_gradient[name] = float(
            parameter.grad.detach()[list(BBOX_NEW_CHANNELS)].abs().max().item()
        )
    if max(
            value for name, value in raw_new_gradient.items()
            if ".cls_logits." in name) <= 0:
        raise RuntimeError("A3 audit produced no new cls-channel gradient")
    if max(
            value for name, value in raw_new_gradient.items()
            if ".bbox_pred." in name) <= 0:
        raise RuntimeError("A3 audit produced no new bbox-channel gradient")

    new_before = {}
    for name, parameter in named.items():
        if name not in (
                "teacher.rpn_norm.cls_logits.weight",
                "teacher.rpn_norm.cls_logits.bias",
                "teacher.rpn_norm.bbox_pred.weight",
                "teacher.rpn_norm.bbox_pred.bias"):
            continue
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        new_before[name] = parameter.detach()[list(channels)].clone()

    protected_snapshots = controller.before_step()
    old_grad_max = {}
    for name, parameter in controller.parameters.items():
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        mask = torch.ones(
            parameter.size(0), dtype=torch.bool, device=parameter.device
        )
        mask[list(channels)] = False
        old_grad_max[name] = float(
            parameter.grad.detach()[mask].abs().max().item()
        )
        if old_grad_max[name] != 0:
            raise RuntimeError("old anchor gradient is nonzero: %s" % name)

    optimizer.step()
    old_parameter_delta = controller.after_step(
        optimizer, protected_snapshots
    )
    new_parameter_delta = {}
    for name, before in new_before.items():
        parameter = named[name]
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        new_parameter_delta[name] = allowed_delta(
            parameter, before, channels
        )
    if max(new_parameter_delta.values()) <= 0:
        raise RuntimeError("no new 32-anchor parameter changed")

    frozen_delta = {
        name: module_max_delta(module, frozen_before[name])
        for name, module in frozen_modules.items()
    }
    for name, delta in frozen_delta.items():
        if delta != 0:
            raise RuntimeError("frozen module changed: %s delta=%s" % (name, delta))

    print("[A3 AUDIT PASS]")
    print("device =", device)
    print("loss =", float(loss.detach().cpu().item()))
    print("proposal counts =", [int(value.size(0)) for value in proposals])
    print("RPN loss keys =", sorted(losses.keys()))
    print("cls shape =", list(cls_logits.shape))
    print("bbox shape =", list(bbox_pred.shape))
    print("raw new-channel gradient max =", raw_new_gradient)
    print("old-channel gradient max after mask =", old_grad_max)
    print("old-channel parameter max_abs_delta =", old_parameter_delta)
    print("new-channel parameter max_abs_delta =", new_parameter_delta)
    print("frozen module max_abs_delta =", frozen_delta)
    print("PASS")


if __name__ == "__main__":
    main()
