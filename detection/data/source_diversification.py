"""C1 source-only, geometry-preserving appearance diversification."""
from __future__ import absolute_import, print_function

import math
import random

import torch
import torch.nn.functional as F


def _box_masks(targets, valid_sizes, padded_size, device, dtype,
               expand=1.10, small_ratio=0.02):
    ph, pw = padded_size
    obj = torch.zeros(len(targets), 1, ph, pw, device=device, dtype=dtype)
    small = torch.zeros_like(obj)
    for bi, (target, size) in enumerate(zip(targets, valid_sizes)):
        vw, vh = float(size[0]), float(size[1])
        area = max(vw * vh, 1.0)
        for box in target["boxes"].detach():
            x0, y0, x1, y1 = [float(v) for v in box]
            x0, x1 = max(0.0, x0), min(vw, x1)
            y0, y1 = max(0.0, y0), min(vh, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            cx, cy = (x0 + x1) * .5, (y0 + y1) * .5
            hw, hh = (x1 - x0) * .5 * expand, (y1 - y0) * .5 * expand
            ix0, ix1 = int(max(0, math.floor(cx - hw))), int(min(pw, math.ceil(cx + hw)))
            iy0, iy1 = int(max(0, math.floor(cy - hh))), int(min(ph, math.ceil(cy + hh)))
            ix1, iy1 = max(ix0 + 1, ix1), max(iy0 + 1, iy1)
            obj[bi, 0, iy0:iy1, ix0:ix1] = 1
            if ((x1 - x0) * (y1 - y0)) / area < small_ratio:
                small[bi, 0, iy0:iy1, ix0:ix1] = 1
    obj = F.avg_pool2d(obj, 9, 1, 4).clamp_(0, 1)
    small = F.avg_pool2d(small, 9, 1, 4).clamp_(0, 1)
    return obj, small


def _blur(x, sigma):
    if sigma <= 1e-6:
        return x
    radius = max(1, int(math.ceil(2.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = x.size(1)
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, radius), groups=channels)
    return F.conv2d(x, ky, padding=(radius, 0), groups=channels)


def _gradient_energy(x):
    lum = x.mean(1, keepdim=True)
    dx = (lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs().mean()
    dy = (lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs().mean()
    return dx + dy


class SourceAdaptiveDiversification(object):
    """Operate in [0,1] image space; input/output use configured normalization."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, images, img_metas, targets):
        if not bool(self.cfg.ENABLED):
            return images, {"applied": False}
        if random.random() >= float(self.cfg.APPLY_PROB):
            return images, {"applied": False}
        x = (images * 0.5 + 0.5).clamp(0, 1)
        original = x
        b, _, ph, pw = x.shape
        valid_sizes = [meta["img_shape"] for meta in img_metas]
        obj, small = _box_masks(
            targets, valid_sizes, (ph, pw), x.device, x.dtype,
            float(self.cfg.OBJECT_PROTECTION.BOX_EXPAND_RATIO),
            float(self.cfg.OBJECT_PROTECTION.SMALL_AREA_RATIO),
        )
        lum = x.mean(1, keepdim=True)
        mean = float(lum.mean().item())
        contrast = float(lum.std().item())
        grad = float(_gradient_energy(x).item())
        # Continuous source-content safety factors; no target statistics.
        dark_safety = min(1.0, max(0.25, mean / float(self.cfg.SOURCE_ADAPTIVE.DARK_REFERENCE)))
        contrast_safety = min(1.0, max(0.25, contrast / float(self.cfg.SOURCE_ADAPTIVE.CONTRAST_REFERENCE)))
        detail_safety = min(1.0, max(0.25, grad / float(self.cfg.SOURCE_ADAPTIVE.GRADIENT_REFERENCE)))

        brightness = random.uniform(*self.cfg.PHOTOMETRIC.BRIGHTNESS)
        if brightness < 1:
            brightness = 1 - (1 - brightness) * dark_safety
        contrast_factor = random.uniform(*self.cfg.PHOTOMETRIC.CONTRAST)
        if contrast_factor < 1:
            contrast_factor = 1 - (1 - contrast_factor) * contrast_safety
        gamma = random.uniform(*self.cfg.PHOTOMETRIC.GAMMA)
        if gamma > 1:
            gamma = 1 + (gamma - 1) * dark_safety
        saturation = random.uniform(*self.cfg.PHOTOMETRIC.SATURATION)

        if bool(self.cfg.PHOTOMETRIC.ENABLED):
            x = (x * brightness).clamp(0, 1)
            channel_mean = x.mean((2, 3), keepdim=True)
            x = ((x - channel_mean) * contrast_factor + channel_mean).clamp(0, 1)
            gray = x.mean(1, keepdim=True)
            x = (gray + saturation * (x - gray)).clamp(0, 1)
            x = x.clamp_min(1e-6).pow(gamma).clamp(0, 1)

        sigma = random.uniform(*self.cfg.DEGRADATION.BLUR_SIGMA)
        noise_std = random.uniform(*self.cfg.DEGRADATION.NOISE_STD)
        attenuation = random.uniform(*self.cfg.DEGRADATION.ATTENUATION)
        protection = 1.0
        if bool(self.cfg.OBJECT_PROTECTION.ENABLED):
            protection = (1.0 - float(self.cfg.OBJECT_PROTECTION.LAMBDA_OBJECT) * obj
                          - float(self.cfg.OBJECT_PROTECTION.LAMBDA_SMALL) * small).clamp(0.15, 1.0)
        if bool(self.cfg.DEGRADATION.ENABLED):
            blurred = _blur(x, sigma)
            x = x + protection * (blurred - x)
            noise = torch.randn_like(x) * noise_std
            x = x + protection * noise
            airlight = x.mean((2, 3), keepdim=True)
            x = x + protection * attenuation * (airlight - x)

        detail_k = random.uniform(*self.cfg.DETAIL.K)
        if detail_k < 1:
            detail_k = 1 - (1 - detail_k) * detail_safety
        if bool(self.cfg.DETAIL.ENABLED):
            low = _blur(x, float(self.cfg.DETAIL.SIGMA))
            detailed = low + detail_k * (x - low)
            x = x + protection * (detailed - x)

        local_coverage = 0.0
        if bool(self.cfg.LOCAL.ENABLED):
            gh, gw = int(self.cfg.LOCAL.GRID_H), int(self.cfg.LOCAL.GRID_W)
            coarse = torch.rand(b, 1, gh, gw, device=x.device, dtype=x.dtype)
            local = F.interpolate(coarse, size=(ph, pw), mode="bilinear", align_corners=False)
            local = _blur(local, float(self.cfg.LOCAL.SMOOTH_SIGMA)).clamp(0, 1)
            local_strength = random.uniform(*self.cfg.LOCAL.STRENGTH)
            local_coverage = float((local > .5).float().mean().item())
            local_variant = (x * (1.0 + local_strength * (local - .5))).clamp(0, 1)
            local_low = _blur(local_variant, float(self.cfg.DETAIL.SIGMA))
            local_variant = (local_low + (1.0 + local_strength * (local - .5)) *
                             (local_variant - local_low)).clamp(0, 1)
            x = x + protection * (local_variant - x)

        x = x.clamp(0, 1)
        if not bool(torch.isfinite(x).all().item()):
            raise FloatingPointError("C1 augmentation produced NaN/Inf")
        stats = {
            "applied": True, "brightness": brightness, "contrast": contrast_factor,
            "gamma": gamma, "saturation": saturation, "blur_sigma": sigma,
            "noise_std": noise_std, "attenuation": attenuation, "detail_k": detail_k,
            "local_coverage": local_coverage,
            "object_coverage": float(obj.mean().item()),
            "small_coverage": float(small.mean().item()),
            "input_mean": float(original.mean().item()),
            "input_std": float(original.std().item()),
            "augmented_mean": float(x.mean().item()),
            "augmented_std": float(x.std().item()),
            "gradient_before": grad,
            "gradient_after": float(_gradient_energy(x).item()),
            "object_mask": obj, "small_mask": small,
        }
        return (x - 0.5) / 0.5, stats
