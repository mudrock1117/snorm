"""Lightweight local adaptive image enhancement.

Input/output use the detector's normalized RGB convention [-1, 1].
This module is intended for test-time ablation and is disabled by default.

Unified enhancement form:
    Y' = mu(x) + [Y - mu(x)] * G(x)
    G(x) = 1 + alpha * C(x)
where C(x) = clip((T_l - sigma_l(x)) / T_l, 0, 1) is the local contrast
deficiency, and alpha is determined by the selected mode.

Supported modes:
    off                  - identity
    contrast             - fixed alpha (original behaviour)
    brightness           - brightness-only (original)
    combined             - brightness + fixed contrast (original)
    global_adaptive      - A1: single image-level alpha
    local_adaptive       - A2: spatially varying alpha_l(x,y)
    global_local_adaptive - A3: global alpha_g modulating local alpha_l
"""
import os

import torch
import torch.nn.functional as F


def _local_mean(x, kernel_size):
    pad = kernel_size // 2
    return F.avg_pool2d(
        x, kernel_size=kernel_size, stride=1, padding=pad,
        count_include_pad=False,
    )


def _smooth_map(alpha_map, kernel_size):
    if kernel_size <= 1:
        return alpha_map
    pad = kernel_size // 2
    return F.avg_pool2d(
        alpha_map, kernel_size=kernel_size, stride=1, padding=pad,
        count_include_pad=False,
    )


def _global_stats(y):
    mu_g = y.mean(dim=(1, 2, 3), keepdim=True)
    var_g = ((y - mu_g) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    sigma_g = var_g.sqrt()
    return mu_g, sigma_g


def _compute_global_alpha(mu_g, sigma_g, cfg):
    T_b = cfg["T_b"]
    T_c = cfg["T_c"]
    alpha_min = cfg["alpha_min"]
    alpha_max = cfg["alpha_max"]
    lambda_b = cfg["lambda_b"]

    D_b = ((T_b - mu_g) / max(T_b, 1e-4)).clamp(0.0, 1.0)
    D_c = ((T_c - sigma_g) / max(T_c, 1e-4)).clamp(0.0, 1.0)

    alpha_g = alpha_min + (alpha_max - alpha_min) * D_c * (1.0 - lambda_b * D_b)
    return alpha_g.clamp(alpha_min, alpha_max)


def _compute_local_alpha(local_std, cfg):
    T_l = cfg["T_l"]
    alpha_min = cfg["alpha_min"]
    alpha_max = cfg["alpha_max"]

    D_l = ((T_l - local_std) / max(T_l, 1e-4)).clamp(0.0, 1.0)
    alpha_l = alpha_min + (alpha_max - alpha_min) * D_l
    return alpha_l.clamp(alpha_min, alpha_max)


def _compute_fused_alpha(alpha_g, D_l, cfg):
    alpha_min = cfg["alpha_min"]
    alpha_max = cfg["alpha_max"]
    lambda_l = cfg["lambda_l"]

    alpha_f = alpha_g * (1.0 + lambda_l * (2.0 * D_l - 1.0))
    return alpha_f.clamp(alpha_min, alpha_max)


def enhance_image(images, mode=None):
    """Enhance normalized RGB images using local luminance statistics.

    Args:
        images: Tensor [B, 3, H, W], normalized to [-1, 1].
        mode: enhancement mode string. If omitted, reads SNORM_IMAGE_ENHANCE.
              'off' returns the input unchanged.
    """
    if mode is None:
        mode = os.environ.get("SNORM_IMAGE_ENHANCE", "off")
    mode = str(mode).strip().lower()
    if mode in ("", "off", "none", "0", "false"):
        return images
    if mode == "clahe":
        mode = "contrast"

    valid_modes = (
        "contrast", "brightness", "combined",
        "global_adaptive", "local_adaptive", "global_local_adaptive",
    )
    if mode not in valid_modes:
        raise ValueError(
            f"SNORM_IMAGE_ENHANCE must be one of {valid_modes}, got '{mode}'"
        )
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, H, W] RGB tensor, got {tuple(images.shape)}")

    kernel_size = int(os.environ.get("SNORM_ENHANCE_KERNEL", "31"))
    kernel_size = max(3, min(kernel_size, 101))
    if kernel_size % 2 == 0:
        kernel_size += 1

    cfg = {
        "T_b": float(os.environ.get("SNORM_ENHANCE_T_B", "0.5")),
        "T_c": float(os.environ.get("SNORM_ENHANCE_T_C", "0.2")),
        "T_l": float(os.environ.get("SNORM_ENHANCE_TARGET_STD", "0.12")),
        "alpha_min": float(os.environ.get("SNORM_ENHANCE_ALPHA_MIN", "0.20")),
        "alpha_max": float(os.environ.get("SNORM_ENHANCE_ALPHA_MAX", "0.50")),
        "lambda_b": float(os.environ.get("SNORM_ENHANCE_LAMBDA_B", "0.5")),
        "lambda_l": float(os.environ.get("SNORM_ENHANCE_LAMBDA_L", "0.5")),
        "smooth_k": int(os.environ.get("SNORM_ENHANCE_SMOOTH_K", "7")),
    }

    rgb = ((images.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    luminance = (
        0.299 * rgb[:, 0:1] +
        0.587 * rgb[:, 1:2] +
        0.114 * rgb[:, 2:3]
    )
    local_mean = _local_mean(luminance, kernel_size)
    local_sq_mean = _local_mean(luminance * luminance, kernel_size)
    local_std = (local_sq_mean - local_mean.square()).clamp_min(1e-6).sqrt()

    enhanced_y = luminance
    if mode in ("brightness", "combined"):
        target_mean = float(os.environ.get("SNORM_ENHANCE_TARGET_LUMA", "0.38"))
        brightness_strength = float(os.environ.get("SNORM_ENHANCE_BRIGHTNESS", "0.45"))
        dark_region = (target_mean - local_mean).clamp_min(0.0)
        enhanced_y = enhanced_y + brightness_strength * dark_region * (1.0 - enhanced_y)

    if mode in ("contrast", "combined",
                "global_adaptive", "local_adaptive", "global_local_adaptive"):
        T_l = cfg["T_l"]
        C_x = ((T_l - local_std) / max(T_l, 1e-4)).clamp(0.0, 1.0)

        if mode == "contrast":
            alpha = float(os.environ.get("SNORM_ENHANCE_CONTRAST", "0.8"))
        elif mode == "global_adaptive":
            mu_g, sigma_g = _global_stats(luminance)
            alpha = _compute_global_alpha(mu_g, sigma_g, cfg)
            alpha = alpha.view(alpha.shape[0], 1, 1, 1)
        elif mode == "local_adaptive":
            alpha = _compute_local_alpha(local_std, cfg)
            alpha = _smooth_map(alpha, cfg["smooth_k"])
        elif mode == "global_local_adaptive":
            mu_g, sigma_g = _global_stats(luminance)
            alpha_g = _compute_global_alpha(mu_g, sigma_g, cfg)
            D_l = C_x
            alpha_f = _compute_fused_alpha(alpha_g, D_l, cfg)
            alpha = _smooth_map(alpha_f, cfg["smooth_k"])

        gain = 1.0 + alpha * C_x
        enhanced_y = enhanced_y + (luminance - local_mean) * (gain - 1.0)

    enhanced_y = enhanced_y.clamp(0.0, 1.0)
    scale = (enhanced_y + 1e-4) / (luminance + 1e-4)
    enhanced_rgb = (rgb * scale).clamp(0.0, 1.0)
    return enhanced_rgb.mul(2.0).sub(1.0).to(dtype=images.dtype)