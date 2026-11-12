#!/usr/bin/env python
"""GPU audit for B1 Protected Small-Anchor + Selective-AHDR."""
from __future__ import print_function

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.modeling.build import build_detectors
from detection.modeling import faster_rcnn as faster_rcnn_module
from diagnostics.a3_protected import (
    A3_TRAINABLE_NAMES,
    BBOX_NEW_CHANNELS,
    CLS_NEW_CHANNELS,
    ProtectedSmallAnchorStep,
)
from diagnostics.b1_protected import (
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


def gradient_diagnostics(loss, named_parameters):
    """Return per-parameter evidence; keep None distinct from exact zero."""
    parameters = [parameter for _, parameter in named_parameters]
    gradients = torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=False
    )
    details = {}
    squared = 0.0
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            details[name] = {
                "is_none": True,
                "abs_mean": None,
                "abs_max": None,
                "norm": None,
            }
            continue
        detached = gradient.detach()
        norm = float(detached.pow(2).sum().sqrt().item())
        details[name] = {
            "is_none": False,
            "abs_mean": float(detached.abs().mean().item()),
            "abs_max": float(detached.abs().max().item()),
            "norm": norm,
        }
        squared += norm * norm
    return math.sqrt(squared), details


def tensor_autograd_info(tensor):
    return {
        "requires_grad": bool(tensor.requires_grad),
        "grad_fn": (
            type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None
        ),
    }


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


def b1_forward_with_capture(model, images, img_metas, targets, degraded):
    """Capture the real mode-13 tensors without changing its loss."""
    captured = {}
    gate_module = model.teacher.selective_ahdr_gate

    def gate_hook(module, inputs, output):
        captured["gate_base"] = inputs[0]
        captured["gate_restored"] = inputs[1]
        captured["gate_selected"] = output[0]
        captured["gate_output"] = output[1]

    original_loss = faster_rcnn_module.selective_consistency_losses

    def loss_wrapper(selected, clean_reference, gate, loss_targets,
                     image_sizes=None):
        captured["selected"] = selected
        captured["clean_reference"] = clean_reference
        captured["loss_gate"] = gate
        return original_loss(
            selected, clean_reference, gate, loss_targets,
            image_sizes=image_sizes,
        )

    handle = gate_module.register_forward_hook(gate_hook)
    faster_rcnn_module.selective_consistency_losses = loss_wrapper
    try:
        losses, outputs = b1_forward(
            model, images, img_metas, targets, degraded
        )
    finally:
        faster_rcnn_module.selective_consistency_losses = original_loss
        handle.remove()
    return losses, outputs, captured


def report_stage(stage_name, model, payload):
    print("[B1 AUDIT STAGE]", stage_name)
    print("model.training =", bool(model.training))
    print("teacher.training =", bool(model.teacher.training))
    print("selective_enabled =", bool(model.teacher.selective_ahdr_enabled))
    print("SNORM_EXPERIMENT =", os.environ.get("SNORM_EXPERIMENT"))
    print("b1_payload_present =", payload is not None)


def run_b1_diagnostic_forward(stage_name, model, images, img_metas,
                              targets, degraded, failures, capture=False):
    if capture:
        losses, outputs, captured = b1_forward_with_capture(
            model, images, img_metas, targets, degraded
        )
    else:
        losses, outputs = b1_forward(
            model, images, img_metas, targets, degraded
        )
        captured = None
    payload = outputs.get("b1") if isinstance(outputs, dict) else None
    report_stage(stage_name, model, payload)
    if payload is None:
        failures.append(
            "%s did not produce B1 diagnostic payload" % stage_name
        )
    return losses, outputs, payload, captured


def perturbation_consistency_gradient(model, captured, consistency_weight,
                                      named_gate_parameters):
    """Audit-only off-optimum check; restores the formal gate bias exactly."""
    gate_module = model.teacher.selective_ahdr_gate
    bias = gate_module.predictor.bias
    old_bias = bias.detach().clone()
    try:
        with torch.no_grad():
            bias.add_(0.2)
        selected, info = gate_module(
            captured["gate_base"].detach(),
            captured["gate_restored"].detach(),
        )
        loss = float(consistency_weight) * F.smooth_l1_loss(
            selected, captured["clean_reference"].detach()
        )
        total, details = gradient_diagnostics(loss, named_gate_parameters)
        return {
            "loss": float(loss.detach().item()),
            "gradient_norm": total,
            "gradient_details": details,
            "gate_mean": float(info["gate"].detach().mean().item()),
            "coefficient_mean": float(
                info["coefficient"].detach().mean().item()
            ),
        }
    finally:
        with torch.no_grad():
            bias.copy_(old_bias)


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
                "lr": float(cfg.SELECTIVE_AHDR.GATE_LR),
                "name": "b1_selective_ahdr_gate",
            },
        ],
        lr=B1_OUTPUT_LR,
        betas=(0.9, 0.999),
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    audit_info = audit_b1_trainable_parameters(
        model,
        optimizer,
        output_lr=B1_OUTPUT_LR,
        gate_lr=float(cfg.SELECTIVE_AHDR.GATE_LR),
    )
    controller = ProtectedSmallAnchorStep(
        model, extra_trainable_names=gate_names
    )
    model.train()

    gate_named = list(model.teacher.selective_ahdr_gate.named_parameters())
    gate_list = [parameter for _, parameter in gate_named]
    failures = []

    optimizer.zero_grad()
    detection_losses, _, detection_payload, _ = (
        run_b1_diagnostic_forward(
            "detection_only", model, images, img_metas, targets,
            degraded, failures
        )
    )
    detection_only = sum(
        value for key, value in detection_losses.items()
        if not key.startswith("b1_")
    )
    detection_gate_gradient, detection_gradient_details = (
        gradient_diagnostics(detection_only, gate_named)
    )
    if detection_gate_gradient <= 0:
        failures.append("detection loss alone has no gate gradient")

    optimizer.zero_grad()
    consistency_losses, _, consistency_payload, consistency_capture = (
        run_b1_diagnostic_forward(
            "consistency_only", model, images, img_metas, targets,
            degraded, failures, capture=True
        )
    )
    consistency_only = consistency_losses["b1_consistency"]
    consistency_gate_gradient, consistency_gradient_details = (
        gradient_diagnostics(consistency_only, gate_named)
    )
    optimizer.zero_grad()
    object_losses, _, object_payload, _ = run_b1_diagnostic_forward(
        "object_consistency_only", model, images, img_metas, targets,
        degraded, failures
    )
    object_consistency_only = object_losses["b1_object_consistency"]
    object_consistency_gate_gradient, object_gradient_details = (
        gradient_diagnostics(object_consistency_only, gate_named)
    )
    if object_consistency_gate_gradient <= 0:
        failures.append("object consistency loss alone has no gate gradient")

    base = consistency_capture["gate_base"]
    restored = consistency_capture["gate_restored"]
    selected_capture = consistency_capture["selected"]
    clean_reference = consistency_capture["clean_reference"]
    gate_capture = consistency_capture["loss_gate"]
    coefficient_capture = consistency_capture["gate_output"]["coefficient"]
    base_restored_diff = (base - restored).detach().abs()
    selected_reference_diff = (
        selected_capture - clean_reference
    ).detach().abs()
    perturbation = perturbation_consistency_gradient(
        model,
        consistency_capture,
        float(cfg.SELECTIVE_AHDR.CONS_WEIGHT),
        gate_named,
    )
    consistency_has_none = any(
        item["is_none"] for item in consistency_gradient_details.values()
    )
    selected_reference_exact = torch.equal(
        selected_capture.detach(), clean_reference.detach()
    )
    if consistency_has_none:
        failures.append(
            "consistency autograd path is disconnected from gate parameters"
        )
    elif consistency_gate_gradient <= 0:
        if not (
                selected_reference_exact
                and perturbation["gradient_norm"] > 0):
            failures.append(
                "consistency gradient is zero without exact optimum proof "
                "and a successful audit-only perturbation"
            )

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
    losses, forward_outputs, step_payload, step_capture = (
        run_b1_diagnostic_forward(
            "protected_optimizer_step", model, images, img_metas,
            targets, degraded, failures, capture=True
        )
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
        failures.append("B1 total loss has no gate gradient")

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
        failures.append("B1 total loss has no new-anchor gradient")

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
            failures.append("B1 old-anchor gradient is nonzero: %s" % name)

    gate = None
    coefficient = None
    mean_abs_diff = None
    max_abs_diff = None
    relative_l2 = None
    if step_payload is not None:
        gate = step_payload["gate"].detach()
        coefficient = step_payload["coefficient"].detach()
        initialization_diff = step_payload["initialization_diff"].detach()
        legacy = step_payload["legacy_feature"].detach()
        if not torch.isfinite(gate).all():
            failures.append("B1 gate contains NaN/Inf")
        if float(gate.min().item()) < 0 or float(gate.max().item()) > 1:
            failures.append("B1 gate is outside [0,1]")
        mean_abs_diff = float(initialization_diff.abs().mean().item())
        max_abs_diff = float(initialization_diff.abs().max().item())
        relative_l2 = float(
            initialization_diff.pow(2).sum().sqrt().item()
            / (legacy.pow(2).sum().sqrt().item() + 1e-12)
        )
        if max_abs_diff > 1e-5:
            failures.append(
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
        failures.append("B1 new-anchor outputs did not update")
    gate_delta = module_max_delta(
        model.teacher.selective_ahdr_gate, gate_before
    )
    if gate_delta <= 0:
        failures.append("B1 gate did not update")

    frozen_delta = {
        name: module_max_delta(module, frozen_before[name])
        for name, module in frozen_modules.items()
    }
    for name, delta in frozen_delta.items():
        if delta != 0:
            failures.append(
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
    print("[B1 AUTOGRAD PATH]")
    print("selected =", tensor_autograd_info(selected_capture))
    print("gate =", tensor_autograd_info(gate_capture))
    print("consistency loss =", tensor_autograd_info(consistency_only))
    print("base/restored mean absolute difference =",
          float(base_restored_diff.mean().item()))
    print("base/restored max absolute difference =",
          float(base_restored_diff.max().item()))
    print("selected/clean_reference mean absolute difference =",
          float(selected_reference_diff.mean().item()))
    print("selected/clean_reference max absolute difference =",
          float(selected_reference_diff.max().item()))
    print("consistency loss value =",
          float(consistency_only.detach().item()))
    print("selected exactly equals clean_reference =",
          selected_reference_exact)
    print("consistency gradient classification =", (
        "disconnected"
        if consistency_has_none
        else (
            "numerical_zero_at_exact_optimum"
            if consistency_gate_gradient <= 0 and selected_reference_exact
            else "nonzero"
        )
    ))
    print("gate mean/std/min/max =", (
        float(gate_capture.detach().mean().item()),
        float(gate_capture.detach().std().item()),
        float(gate_capture.detach().min().item()),
        float(gate_capture.detach().max().item()),
    ))
    print("coefficient mean/std/min/max =", (
        float(coefficient_capture.detach().mean().item()),
        float(coefficient_capture.detach().std().item()),
        float(coefficient_capture.detach().min().item()),
        float(coefficient_capture.detach().max().item()),
    ))
    print("detection-only gate gradients =", detection_gradient_details)
    print("consistency-only gate gradients =", consistency_gradient_details)
    print("object-consistency-only gate gradients =",
          object_gradient_details)
    print("audit-only perturbed consistency =", perturbation)
    print("detection-only gate gradient norm =", detection_gate_gradient)
    print("consistency-only gate gradient norm =", consistency_gate_gradient)
    print("object-consistency-only gate gradient norm =",
          object_consistency_gate_gradient)
    print("total gate gradient norm =", gate_gradient)
    print("new-anchor raw gradient max =", raw_new_gradient)
    print("old-anchor gradient after mask =", old_gradient)
    print("old-anchor parameter delta =", old_delta)
    print("new-anchor parameter delta =", new_delta)
    print("gate parameter delta =", gate_delta)
    print("frozen module delta =", frozen_delta)
    gt_example = (
        targets[0]["boxes"][0].detach().cpu().tolist()
        if len(targets) and targets[0]["boxes"].numel() else None
    )
    print("source img_shape =", [meta["img_shape"] for meta in img_metas])
    print("source GT box example =", gt_example)
    if step_payload is not None:
        object_mask = step_payload["object_mask"].detach()
        print("selected feature shape =", list(
            step_payload["legacy_feature"].shape
        ))
        print("object mask shape =", list(object_mask.shape))
        print("object mask coverage =", [
            float(mask.sum().item()) for mask in object_mask
        ])
        print("gate min/max/mean/std =", (
            float(gate.min().item()), float(gate.max().item()),
            float(gate.mean().item()), float(gate.std().item()),
        ))
        print("effective coefficient mean/std =", (
            float(coefficient.mean().item()), float(coefficient.std().item()),
        ))
    else:
        print("selected feature shape = unavailable")
        print("object mask shape = unavailable")
        print("object mask coverage = unavailable")
    print("initialization mean_abs_diff =", mean_abs_diff)
    print("initialization max_abs_diff =", max_abs_diff)
    print("initialization relative_L2_diff =", relative_l2)
    if float(auxiliary.detach().item()) > float(detection_raw.detach().item()):
        print("WARNING: weighted auxiliary loss exceeds detection loss")
    if failures:
        print("[B1 AUDIT FAIL]")
        for failure in failures:
            print("FAIL:", failure)
        print("failure_reasons =", failures)
        return 1
    print("[B1 AUDIT PASS]")
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
