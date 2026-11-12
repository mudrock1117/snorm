"""Lightweight baseline-preserving Selective-AHDR gate and source GT masks."""
from __future__ import absolute_import, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_ahdr_render_tensor(result):
    """Parse OHDR.forward without changing its train/eval return contract."""
    if torch.is_tensor(result):
        return result
    if (
            isinstance(result, (tuple, list))
            and len(result) == 2
            and torch.is_tensor(result[0])
            and isinstance(result[1], dict)):
        return result[0]
    raise TypeError(
        "unexpected AHDR return contract: %s" % type(result).__name__
    )


class SelectiveAHDRGate(nn.Module):
    """Spatial gate over base and restoration delta.

    coefficient = alpha_max * sigmoid(G(base, restored - base)).
    The final layer is zero initialized, so gate=0.5 everywhere and an
    alpha_max of 1.2 exactly reproduces the legacy 0.6 residual blend.
    """

    def __init__(self, channels=3, hidden_channels=8, alpha_max=1.2,
                 gate_init=0.5):
        super(SelectiveAHDRGate, self).__init__()
        if not 0.0 < float(gate_init) < 1.0:
            raise ValueError("gate_init must be strictly between zero and one")
        self.alpha_max = float(alpha_max)
        self.gate_init = float(gate_init)
        self.encoder = nn.Conv2d(
            int(channels) * 2, int(hidden_channels), 3, padding=1, bias=True
        )
        self.predictor = nn.Conv2d(
            int(hidden_channels), 1, 1, padding=0, bias=True
        )
        nn.init.kaiming_normal_(self.encoder.weight, mode="fan_out")
        nn.init.constant_(self.encoder.bias, 0.0)
        nn.init.constant_(self.predictor.weight, 0.0)
        initial_logit = math.log(self.gate_init / (1.0 - self.gate_init))
        nn.init.constant_(self.predictor.bias, initial_logit)

    def forward(self, base, restored):
        if tuple(base.shape) != tuple(restored.shape):
            raise RuntimeError(
                "Selective-AHDR base/restored shape mismatch: %s vs %s"
                % (tuple(base.shape), tuple(restored.shape))
            )
        if not torch.isfinite(base).all() or not torch.isfinite(restored).all():
            raise FloatingPointError("Selective-AHDR input contains NaN/Inf")
        delta = restored - base
        hidden = F.relu(self.encoder(torch.cat((base, delta), dim=1)))
        gate = torch.sigmoid(self.predictor(hidden))
        coefficient = gate * self.alpha_max
        output = base + coefficient * delta
        if not torch.isfinite(output).all():
            raise FloatingPointError("Selective-AHDR output contains NaN/Inf")
        return output, {
            "gate": gate,
            "coefficient": coefficient,
            "base": base,
            "restored": restored,
            "delta": delta,
        }

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def project_source_boxes_to_mask(targets, feature_size, image_sizes=None,
                                 device=None, dtype=None):
    """Project source GT; image_sizes follow project img_shape=(width,height).\n\n    Every non-empty tiny box receives at least one feature cell.\n    """
    height, width = int(feature_size[0]), int(feature_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("feature_size must be positive")
    batch = len(targets)
    if device is None:
        device = targets[0]["boxes"].device if batch else torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    mask = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"]
        if boxes is None or boxes.numel() == 0:
            continue
        if image_sizes is None:
            image_height, image_width = height, width
        else:
            image_width, image_height = image_sizes[batch_index]
        scale_x = float(width) / float(image_width)
        scale_y = float(height) / float(image_height)
        for box in boxes.detach():
            x0 = max(0, min(width - 1, int(math.floor(float(box[0]) * scale_x))))
            y0 = max(0, min(height - 1, int(math.floor(float(box[1]) * scale_y))))
            x1 = max(x0 + 1, min(width, int(math.ceil(float(box[2]) * scale_x))))
            y1 = max(y0 + 1, min(height, int(math.ceil(float(box[3]) * scale_y))))
            mask[batch_index, 0, y0:y1, x0:x1] = 1
    return mask


def selective_consistency_losses(selected, clean_reference, gate, targets,
                                  image_sizes=None):
    if tuple(selected.shape) != tuple(clean_reference.shape):
        raise RuntimeError("B1 consistency feature shape mismatch")
    distance = F.smooth_l1_loss(
        selected, clean_reference.detach(), reduction="none"
    )
    consistency = distance.mean()
    object_mask = project_source_boxes_to_mask(
        targets,
        (selected.size(-2), selected.size(-1)),
        image_sizes=image_sizes,
        device=selected.device,
        dtype=selected.dtype,
    )
    expanded_mask = object_mask.expand_as(distance)
    object_consistency = (
        distance * expanded_mask
    ).sum() / (expanded_mask.sum() + 1e-6)
    gate_regularization = (gate - 0.5).pow(2).mean()
    return consistency, object_consistency, gate_regularization, object_mask
