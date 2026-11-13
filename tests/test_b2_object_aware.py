from __future__ import absolute_import, print_function

import torch

from detection.modeling.norm.object_aware_ahdr import (
    build_object_aware_masks,
    object_aware_gate_losses,
)


def test_b2_mask_uses_padded_canvas_and_valid_image_area():
    targets = [{
        "boxes": torch.tensor([
            [50.0, 20.0, 70.0, 40.0],
            [199.0, 99.0, 199.2, 99.2],
        ])
    }]
    masks = build_object_aware_masks(
        targets,
        valid_image_sizes=[(200, 100)],
        padded_image_size=(120, 240),
        feature_size=(12, 24),
        box_expand_ratio=1.0,
        small_area_ratio=0.02,
        small_weight_beta=1.0,
    )
    assert masks["object_mask"].shape == (1, 1, 12, 24)
    # x scales with feature width/padded width; y with height/padded height.
    assert masks["object_mask"][0, 0, 2:4, 5:7].max().item() > 0
    assert masks["object_mask"][0, 0, 9, 19].item() > 0
    assert masks["num_objects"] == [2]
    assert masks["num_small_objects"] == [2]
    assert masks["small_support"].sum().item() > 0
    assert masks["small_weighted"].max().item() <= 2.0
    assert masks["object_mask"].min().item() >= 0
    assert masks["object_mask"].max().item() <= 1


def test_b2_losses_are_finite_and_small_floor_has_gradient():
    gate_logit = torch.nn.Parameter(torch.tensor(0.0))
    gate = torch.sigmoid(gate_logit).view(1, 1, 1, 1).expand(1, 1, 5, 7)
    object_mask = torch.ones_like(gate)
    masks = {
        "object_mask": object_mask,
        "small_support": object_mask,
        "small_weighted": object_mask * 2.0,
        "background_mask": torch.zeros_like(gate),
    }
    losses = object_aware_gate_losses(gate, masks)
    assert all(torch.isfinite(value).item() for value in losses.values())
    assert losses["object_floor"].item() == 0
    assert losses["small_floor"].item() > 0
    gradient = torch.autograd.grad(losses["small_floor"], gate_logit)[0]
    assert gradient.abs().item() > 0


def test_b2_empty_small_support_returns_zero_without_nan():
    gate = torch.full((2, 1, 4, 6), 0.5, requires_grad=True)
    object_mask = torch.ones_like(gate)
    masks = {
        "object_mask": object_mask,
        "small_support": torch.zeros_like(gate),
        "small_weighted": torch.ones_like(gate),
        "background_mask": torch.zeros_like(gate),
    }
    losses = object_aware_gate_losses(gate, masks)
    assert losses["small_floor"].item() == 0
    assert torch.isfinite(losses["small_floor"]).item()


def test_b2_gate_048_activates_object_and_small_floors():
    gate = torch.full((1, 1, 3, 5), 0.48, requires_grad=True)
    mask = torch.ones_like(gate)
    losses = object_aware_gate_losses(
        gate,
        {
            "object_mask": mask,
            "small_support": mask,
            "small_weighted": mask * 1.5,
            "background_mask": torch.zeros_like(mask),
        },
    )
    assert losses["object_floor"].item() > 0
    assert losses["small_floor"].item() > 0
