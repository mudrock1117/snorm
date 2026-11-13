#!/usr/bin/env python
"""Single/three-GPU protected-update audit for C1."""
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
from detection.data.source_diversification import SourceAdaptiveDiversification
from detection.modeling.build import build_detectors
from detection.utils import dist_utils
from diagnostics.a3_protected import (
    A3_TRAINABLE_NAMES, BBOX_NEW_CHANNELS, CLS_NEW_CHANNELS,
    ProtectedSmallAnchorStep, audit_a3_trainable_parameters,
    configure_a3_trainable_parameters,
)
from diagnostics.small_anchor import load_small_anchor_checkpoint


EXPECTED = ("/home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/"
            "s0_dbias_color_E2_zdh/E2/best.pth")


def snap(module):
    return [p.detach().cpu().clone() for p in module.parameters()]


def delta(module, before):
    value = 0.0
    for parameter, old in zip(module.parameters(), before):
        value = max(value, float((parameter.detach().cpu() - old).abs().max().item()))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist-url", default="env://")
    args = parser.parse_args()
    if args.checkpoint != EXPECTED:
        raise RuntimeError("C1 audit requires fixed E2 checkpoint")
    dist_utils.init_distributed_mode(args)
    rank, world = dist_utils.get_rank(), dist_utils.get_world_size()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    if not bool(cfg.C1_SOURCE_DIVERSIFICATION.ENABLED):
        raise RuntimeError("C1 augmentation is disabled")
    if tuple(cfg.MODEL.RPN.ANCHOR_SIZES) != (32, 64, 128, 256, 512):
        raise RuntimeError("C1 requires protected 32-anchor layout")
    if world > 1 and world != 3:
        raise RuntimeError("C1 DDP audit requires three ranks")
    random.seed(20261113 + rank)
    np.random.seed(20261113 + rank)
    torch.manual_seed(20261113 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261113 + rank)
    formal_batch = int(cfg.SOLVER.BATCH_SIZE)
    execution_batch = formal_batch if args.config_only else 1
    loader = build_data_loaders(
        cfg.DATASETS.TRAINS, transforms=cfg.INPUT.TRANSFORMS_TRAIN,
        is_train=True, distributed=world > 1, batch_size=execution_batch,
        num_workers=0, shuffle=False,
    )
    sampler = loader.batch_sampler.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    batch = next(iter(loader))
    config_result = {
        "rank": rank, "formal_batch": formal_batch,
        "loader_batch": int(loader.batch_sampler.batch_size),
        "input_batch": int(batch[0].size(0)),
        "samples_per_rank": len(sampler), "steps": int(math.ceil(float(len(sampler)) / formal_batch)),
    }
    if args.config_only:
        gathered = dist_utils.all_gather(config_result)
        if rank == 0:
            failures = []
            for item in gathered:
                if item["formal_batch"] != 4 or item["loader_batch"] != 4 or item["input_batch"] != 4:
                    failures.append("rank %s batch mismatch" % item["rank"])
                if world == 3 and (item["samples_per_rank"] != 6440 or item["steps"] != 1610):
                    failures.append("rank %s sampler/steps mismatch" % item["rank"])
            print("[C1 CONFIG AUDIT]", gathered)
            print("[C1 CONFIG AUDIT PASS]" if not failures else "[C1 CONFIG AUDIT FAIL]", failures)
        return 0

    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    images, metas, targets = batch[0].to(device), batch[1], [v.to(device) for v in batch[2]]
    boxes_before = [target["boxes"].clone() for target in targets]
    augmented, aug_stats = SourceAdaptiveDiversification(cfg.C1_SOURCE_DIVERSIFICATION)(images, metas, targets)
    failures = []
    if augmented.shape != images.shape or not bool(torch.isfinite(augmented).all().item()):
        failures.append("augmentation shape/finite failure")
    if any(not torch.equal(a, b["boxes"]) for a, b in zip(boxes_before, targets)):
        failures.append("augmentation changed GT boxes")

    model = build_detectors(cfg).to(device)
    loaded = load_small_anchor_checkpoint(
        model, args.checkpoint, old_anchor_sizes=(64, 128, 256, 512),
        new_anchor_sizes=(32, 64, 128, 256, 512),
    )
    params = configure_a3_trainable_parameters(model)
    optimizer = torch.optim.Adam(params, lr=5e-5, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    audit = audit_a3_trainable_parameters(model, optimizer)
    controller = ProtectedSmallAnchorStep(model)
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu,
            find_unused_parameters=True, broadcast_buffers=False,
        )
    inner = model.module if hasattr(model, "module") else model
    frozen = {
        "backbone_0": inner.teacher.backbone_0, "backbone_1": inner.teacher.backbone_1,
        "backbone_2": inner.teacher.backbone_2, "backbone_3": inner.teacher.backbone_3,
        "proj_norm": inner.teacher.proj_norm, "rpn_conv": inner.teacher.rpn_norm.conv,
        "box_head_norm": inner.teacher.box_head_norm, "rpn_eval": inner.teacher.rpn_eval,
    }
    frozen_before = dict((name, snap(module)) for name, module in frozen.items())
    named = dict(inner.named_parameters())
    new_before = {}
    optimizer.zero_grad()
    model.train()
    losses, _ = model(augmented, metas, targets, mode_training=0, is_update=False)
    required = ("rpn_cls_loss_w", "rpn_reg_loss_w", "rcnn_cls_loss_w", "rcnn_reg_loss_w")
    missing = [key for key in required if key not in losses]
    if missing:
        failures.append("missing detection losses %s" % missing)
    sum(losses.values()).backward()
    raw_new, old_grad = {}, {}
    for name in A3_TRAINABLE_NAMES:
        p = named[name]
        channels = CLS_NEW_CHANNELS if ".cls_logits." in name else BBOX_NEW_CHANNELS
        new_before[name] = p.detach()[list(channels)].clone()
        raw_new[name] = float(p.grad.detach()[list(channels)].abs().max().item())
    protected = controller.before_step()
    for name, p in controller.parameters.items():
        channels = CLS_NEW_CHANNELS if ".cls_logits." in name else BBOX_NEW_CHANNELS
        mask = torch.ones(p.size(0), dtype=torch.bool, device=p.device)
        mask[list(channels)] = False
        old_grad[name] = float(p.grad.detach()[mask].abs().max().item())
    optimizer.step()
    old_delta = controller.after_step(optimizer, protected)
    new_delta = {}
    for name, old in new_before.items():
        channels = CLS_NEW_CHANNELS if ".cls_logits." in name else BBOX_NEW_CHANNELS
        new_delta[name] = float((named[name].detach()[list(channels)] - old).abs().max().item())
    frozen_delta = dict((name, delta(module, frozen_before[name])) for name, module in frozen.items())
    if max(raw_new.values()) <= 0 or max(new_delta.values()) <= 0:
        failures.append("new 32-anchor did not receive/update")
    if max(old_grad.values()) != 0 or max(old_delta.values()) != 0:
        failures.append("old anchor protection failed")
    if max(frozen_delta.values()) != 0:
        failures.append("frozen module changed")
    result = {
        "rank": rank, "augmentation": dict((k, v) for k, v in aug_stats.items()
                                            if k not in ("object_mask", "small_mask")),
        "trainable": audit["trainable_names"], "new_raw_grad": raw_new,
        "old_grad": old_grad, "old_delta": old_delta, "new_delta": new_delta,
        "frozen_delta": frozen_delta, "failures": failures,
    }
    gathered = dist_utils.all_gather(result)
    if rank == 0:
        all_failures = ["rank %s: %s" % (r["rank"], f) for r in gathered for f in r["failures"]]
        print("[C1 TRAINING AUDIT]", gathered)
        print("[C1 TRAINING AUDIT PASS]" if not all_failures else "[C1 TRAINING AUDIT FAIL]")
        print("failure_reasons =", all_failures)
    if world > 1:
        dist.barrier()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
