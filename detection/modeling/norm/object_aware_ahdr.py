"""Source-GT object-aware masks and losses for the B2 experiment."""
from __future__ import absolute_import, print_function

import math

import torch
import torch.nn.functional as F


def _soft_box_mask(height, width, x0, y0, x1, y1, device, dtype):
    mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    ix0 = max(0, min(width - 1, int(math.floor(x0))))
    iy0 = max(0, min(height - 1, int(math.floor(y0))))
    ix1 = max(ix0 + 1, min(width, int(math.ceil(x1))))
    iy1 = max(iy0 + 1, min(height, int(math.ceil(y1))))
    mask[:, :, iy0:iy1, ix0:ix1] = 1.0
    return F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1).clamp_(0, 1)


def build_object_aware_masks(
        targets, valid_image_sizes, padded_image_size, feature_size,
        box_expand_ratio=1.15, small_area_ratio=0.02,
        small_weight_beta=1.0, device=None, dtype=None):
    """Project resized source GT from padded-image coordinates to Gate space.

    valid_image_sizes follow project img_shape=(width, height). Targets are
    resized-image coordinates in the top-left of the padded batch tensor.
    padded_image_size and feature_size follow (height, width).
    """
    feature_h, feature_w = int(feature_size[0]), int(feature_size[1])
    padded_h, padded_w = int(padded_image_size[0]), int(padded_image_size[1])
    if feature_h <= 0 or feature_w <= 0 or padded_h <= 0 or padded_w <= 0:
        raise ValueError("B2 mask spatial sizes must be positive")
    batch = len(targets)
    if device is None:
        device = targets[0]["boxes"].device
    if dtype is None:
        dtype = torch.float32
    object_mask = torch.zeros(
        batch, 1, feature_h, feature_w, device=device, dtype=dtype
    )
    small_weighted = torch.zeros_like(object_mask)
    small_support = torch.zeros_like(object_mask)
    num_objects = []
    num_small_objects = []
    scale_x = float(feature_w) / float(padded_w)
    scale_y = float(feature_h) / float(padded_h)

    for batch_index, target in enumerate(targets):
        boxes = target["boxes"]
        valid_w, valid_h = valid_image_sizes[batch_index]
        valid_w, valid_h = float(valid_w), float(valid_h)
        valid_area = max(valid_w * valid_h, 1.0)
        object_count = 0
        small_count = 0
        for box in boxes.detach():
            x0 = max(0.0, min(valid_w, float(box[0])))
            y0 = max(0.0, min(valid_h, float(box[1])))
            x1 = max(0.0, min(valid_w, float(box[2])))
            y1 = max(0.0, min(valid_h, float(box[3])))
            if x1 <= x0 or y1 <= y0:
                continue
            object_count += 1
            area_ratio = ((x1 - x0) * (y1 - y0)) / valid_area
            center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            half_w = 0.5 * (x1 - x0) * float(box_expand_ratio)
            half_h = 0.5 * (y1 - y0) * float(box_expand_ratio)
            ex0 = max(0.0, center_x - half_w)
            ey0 = max(0.0, center_y - half_h)
            ex1 = min(valid_w, center_x + half_w)
            ey1 = min(valid_h, center_y + half_h)
            box_mask = _soft_box_mask(
                feature_h, feature_w,
                ex0 * scale_x, ey0 * scale_y,
                ex1 * scale_x, ey1 * scale_y,
                device, dtype,
            )
            object_mask[batch_index:batch_index + 1] = torch.max(
                object_mask[batch_index:batch_index + 1], box_mask
            )
            relative = min(
                area_ratio / max(float(small_area_ratio), 1e-12), 1.0
            )
            importance = 1.0 + float(small_weight_beta) * (1.0 - relative)
            small_weighted[batch_index:batch_index + 1] = torch.max(
                small_weighted[batch_index:batch_index + 1],
                box_mask * importance,
            )
            if area_ratio < float(small_area_ratio):
                small_count += 1
                small_support[batch_index:batch_index + 1] = torch.max(
                    small_support[batch_index:batch_index + 1], box_mask
                )
        num_objects.append(object_count)
        num_small_objects.append(small_count)

    return {
        "object_mask": object_mask.clamp_(0, 1),
        "small_weighted": small_weighted,
        "small_support": small_support.clamp_(0, 1),
        "background_mask": (1.0 - object_mask).clamp_(0, 1),
        "num_objects": num_objects,
        "num_small_objects": num_small_objects,
    }


def object_aware_gate_losses(
        gate, masks, object_floor=0.5, small_object_floor=0.525,
        eps=1e-6):
    object_mask = masks["object_mask"]
    small_support = masks["small_support"]
    small_weighted = masks["small_weighted"]
    background_mask = masks["background_mask"]

    object_penalty = F.relu(float(object_floor) - gate).pow(2)
    object_loss = (
        object_penalty * object_mask
    ).sum() / (object_mask.sum() + eps)

    small_importance = small_support * small_weighted
    small_penalty = F.relu(float(small_object_floor) - gate).pow(2)
    small_loss = (
        small_penalty * small_importance
    ).sum() / (small_importance.sum() + eps)

    background_loss = (
        (gate - 0.5).pow(2) * background_mask
    ).sum() / (background_mask.sum() + eps)

    tv_h = (
        (gate[:, :, 1:, :] - gate[:, :, :-1, :]).abs().mean()
        if gate.size(-2) > 1 else gate.sum() * 0.0
    )
    tv_w = (
        (gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().mean()
        if gate.size(-1) > 1 else gate.sum() * 0.0
    )
    tv_loss = tv_h + tv_w
    anchor_loss = (gate.mean() - 0.5).pow(2)
    return {
        "object_floor": object_loss,
        "small_floor": small_loss,
        "background": background_loss,
        "tv": tv_loss,
        "global_anchor": anchor_loss,
    }


def masked_gate_statistics(gate, mask, eps=1e-6):
    weight = mask.expand_as(gate)
    count = weight.sum()
    mean = (gate * weight).sum() / (count + eps)
    variance = ((gate - mean).pow(2) * weight).sum() / (count + eps)
    return mean, variance.clamp_min(0).sqrt(), count
