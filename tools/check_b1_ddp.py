#!/usr/bin/env python
"""Three-rank DDP smoke/audit for protected B1 training."""
from __future__ import absolute_import, print_function

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.modeling.build import build_detectors
from detection.utils import dist_utils
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


EXPECTED_CHECKPOINT = (
    "/home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/"
    "s0_dbias_color_E2_zdh/E2/best.pth"
)
REQUIRED_DETECTION_LOSSES = (
    "rpn_cls_loss_w",
    "rpn_reg_loss_w",
    "rcnn_cls_loss_w",
    "rcnn_reg_loss_w",
)


def move_batch(batch, device):
    if len(batch) != 8:
        raise RuntimeError("B1 DDP audit expects the real eight-item batch")
    images, img_metas, targets, _, _, degraded, _, _ = batch
    return (
        images.to(device),
        img_metas,
        [target.to(device) for target in targets],
        degraded.to(device),
    )


def snapshot_module(module):
    return [
        parameter.detach().cpu().clone()
        for parameter in module.parameters()
    ]


def module_max_delta(module, before):
    maximum = 0.0
    for parameter, old in zip(module.parameters(), before):
        maximum = max(
            maximum,
            float((parameter.detach().cpu() - old).abs().max().item()),
        )
    return maximum


def selected_delta(parameter, before, channels):
    return float(
        (
            parameter.detach()[list(channels)] - before
        ).abs().max().item()
    )


def gradient_norm(parameters):
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(squared)


def all_rank_results(value):
    return dist_utils.all_gather(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.checkpoint != EXPECTED_CHECKPOINT:
        raise RuntimeError(
            "B1 DDP audit requires fixed E2 checkpoint: %s"
            % EXPECTED_CHECKPOINT
        )

    dist_utils.init_distributed_mode(args)
    if not args.distributed:
        raise RuntimeError("check_b1_ddp.py must run under distributed launch")
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    if world_size != 3:
        raise RuntimeError("B1 DDP audit requires world_size=3")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if not bool(cfg.SELECTIVE_AHDR.ENABLED):
        raise RuntimeError("B1 DDP audit requires SELECTIVE_AHDR.ENABLED")
    if tuple(cfg.MODEL.RPN.ANCHOR_SIZES) != (32, 64, 128, 256, 512):
        raise RuntimeError("B1 DDP audit requires protected anchor layout")

    seed = 20261112
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    device = torch.device("cuda:%d" % args.gpu)
    loader = build_data_loaders(
        cfg.DATASETS.TRAINS,
        transforms=cfg.INPUT.TRANSFORMS_TRAIN,
        is_train=True,
        distributed=True,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
    )
    sampler = loader.batch_sampler.sampler
    sampler.set_epoch(0)
    sampler_indices = list(iter(sampler))
    local_index_head = sampler_indices[:int(cfg.SOLVER.BATCH_SIZE)]
    images, img_metas, targets, degraded = move_batch(
        next(iter(loader)), device
    )
    local_batch = int(images.size(0))

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
        raise RuntimeError("B1 DDP checkpoint missing-key mismatch")
    if load_result["unexpected_keys"]:
        raise RuntimeError("B1 DDP checkpoint has unexpected keys")

    output_parameters, gate_parameters, gate_names = (
        configure_b1_trainable_parameters(model)
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": output_parameters,
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

    model = DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.gpu,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    inner = model.module
    model.train()

    frozen_modules = {
        "backbone_0": inner.teacher.backbone_0,
        "backbone_1": inner.teacher.backbone_1,
        "backbone_2": inner.teacher.backbone_2,
        "backbone_3": inner.teacher.backbone_3,
        "proj_norm": inner.teacher.proj_norm,
        "render_net": inner.teacher.proj_norm.render_net,
        "rpn_norm_shared_conv": inner.teacher.rpn_norm.conv,
        "box_head_norm": inner.teacher.box_head_norm,
        "rpn_eval": inner.teacher.rpn_eval,
    }
    frozen_before = {
        name: snapshot_module(module)
        for name, module in frozen_modules.items()
    }
    gate_before = snapshot_module(inner.teacher.selective_ahdr_gate)

    optimizer.zero_grad()
    losses, outputs = model(
        images,
        img_metas,
        targets,
        mode_training=13,
        stpd_images=degraded,
        is_update=False,
    )
    local_failures = []
    missing_losses = [
        name for name in REQUIRED_DETECTION_LOSSES if name not in losses
    ]
    if missing_losses:
        local_failures.append(
            "missing source detection losses: %s" % missing_losses
        )
    if not isinstance(outputs, dict) or outputs.get("b1") is None:
        local_failures.append("mode13 did not return B1 payload")

    total_loss = sum(losses.values())
    total_loss.backward()

    gate_gradient = gradient_norm(
        inner.teacher.selective_ahdr_gate.parameters()
    )
    if gate_gradient <= 0:
        local_failures.append("Gate DDP gradient is not positive")

    named = dict(inner.named_parameters())
    new_before = {}
    raw_new_gradient = {}
    for name in A3_TRAINABLE_NAMES:
        parameter = named[name]
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        new_before[name] = parameter.detach()[list(channels)].clone()
        if parameter.grad is None:
            raw_new_gradient[name] = 0.0
            local_failures.append("new-anchor gradient is None: %s" % name)
        else:
            raw_new_gradient[name] = float(
                parameter.grad.detach()[list(channels)].abs().max().item()
            )
    if max(raw_new_gradient.values()) <= 0:
        local_failures.append("new-anchor raw gradient is not positive")

    protected = controller.before_step()
    old_gradient = {}
    for name, parameter in controller.parameters.items():
        channels = (
            CLS_NEW_CHANNELS if ".cls_logits." in name
            else BBOX_NEW_CHANNELS
        )
        old_mask = torch.ones(
            parameter.size(0), dtype=torch.bool, device=parameter.device
        )
        old_mask[list(channels)] = False
        old_gradient[name] = float(
            parameter.grad.detach()[old_mask].abs().max().item()
        )
        if old_gradient[name] != 0:
            local_failures.append(
                "old-anchor gradient after mask is nonzero: %s" % name
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
        local_failures.append("new-anchor parameter delta is not positive")
    gate_delta = module_max_delta(
        inner.teacher.selective_ahdr_gate, gate_before
    )
    if gate_delta <= 0:
        local_failures.append("Gate parameter delta is not positive")

    frozen_delta = {
        name: module_max_delta(module, frozen_before[name])
        for name, module in frozen_modules.items()
    }
    for name, delta in frozen_delta.items():
        if delta != 0:
            local_failures.append(
                "frozen module changed: %s delta=%s" % (name, delta)
            )

    torch.cuda.synchronize(device)
    local_result = {
        "rank": rank,
        "local_rank": args.gpu,
        "device": str(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "per_rank_batch": local_batch,
        "sampler_index_head": local_index_head,
        "gate_gradient_norm": gate_gradient,
        "new_anchor_raw_gradient_max": raw_new_gradient,
        "old_anchor_gradient_after_mask": old_gradient,
        "old_anchor_parameter_delta": old_delta,
        "new_anchor_parameter_delta": new_delta,
        "gate_parameter_delta": gate_delta,
        "frozen_module_delta": frozen_delta,
        "failures": local_failures,
    }
    gathered = all_rank_results(local_result)

    if rank == 0:
        failure_reasons = []
        for result in gathered:
            for failure in result["failures"]:
                failure_reasons.append(
                    "rank %s: %s" % (result["rank"], failure)
                )
        index_heads = [
            set(result["sampler_index_head"]) for result in gathered
        ]
        for left in range(len(index_heads)):
            for right in range(left + 1, len(index_heads)):
                overlap = index_heads[left] & index_heads[right]
                if overlap:
                    failure_reasons.append(
                        "sampler overlap ranks %d/%d: %s"
                        % (left, right, sorted(overlap))
                    )
        gate_norms = [
            result["gate_gradient_norm"] for result in gathered
        ]
        if max(gate_norms) - min(gate_norms) > 1e-8:
            failure_reasons.append(
                "DDP Gate gradients differ across ranks: %s" % gate_norms
            )

        print("[B1 DDP AUDIT]")
        print("world_size =", world_size)
        print("rank/local_rank/device mapping =", [
            (item["rank"], item["local_rank"], item["device"],
             item["visible_devices"]) for item in gathered
        ])
        print("dataset size =", len(loader.dataset))
        print("optimizer steps per epoch =", len(loader))
        print("effective per-rank batch size =", local_batch)
        print("global batch size =", local_batch * world_size)
        print("sampler first-batch indices =", [
            item["sampler_index_head"] for item in gathered
        ])
        print("trainable whitelist =", audit_info["trainable_names"])
        print("optimizer groups/LRs =", [
            (group.get("name"), group["lr"])
            for group in optimizer.param_groups
        ])
        print("Gate gradient norm by rank =", gate_norms)
        print("new-anchor raw gradient max =", [
            item["new_anchor_raw_gradient_max"] for item in gathered
        ])
        print("old-anchor gradient after mask =", [
            item["old_anchor_gradient_after_mask"] for item in gathered
        ])
        print("old-anchor parameter delta =", [
            item["old_anchor_parameter_delta"] for item in gathered
        ])
        print("new-anchor parameter delta =", [
            item["new_anchor_parameter_delta"] for item in gathered
        ])
        print("Gate parameter delta =", [
            item["gate_parameter_delta"] for item in gathered
        ])
        print("frozen module delta =", [
            item["frozen_module_delta"] for item in gathered
        ])
        print("all ranks completed =", len(gathered) == world_size)
        if failure_reasons:
            print("[B1 DDP AUDIT FAIL]")
            print("failure_reasons =", failure_reasons)
        else:
            print("[B1 DDP AUDIT PASS]")
            print("PASS")

    dist.barrier()
    has_failure = torch.tensor(
        [1 if local_failures else 0], device=device, dtype=torch.int32
    )
    dist.all_reduce(has_failure, op=dist.ReduceOp.MAX)
    return int(has_failure.item())


if __name__ == "__main__":
    sys.exit(main())
