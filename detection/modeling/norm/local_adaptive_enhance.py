"""Lightweight, parameter-free local adaptive image enhancement.

Input/output use the detector's normalized RGB convention [-1, 1].
This module is intended for test-time ablation and is disabled by default.
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


def enhance_image(images, mode=None):
    """Enhance normalized RGB images using local luminance statistics.

    Args:
        images: Tensor [B, 3, H, W], normalized to [-1, 1].
        mode: 'contrast', 'brightness', or 'combined'. If omitted, reads
              SNORM_IMAGE_ENHANCE. 'off' returns the input unchanged.
    """
    if mode is None:
        mode = os.environ.get("SNORM_IMAGE_ENHANCE", "off")
    mode = str(mode).strip().lower()
    if mode in ("", "off", "none", "0", "false"):
        return images
    if mode == "clahe":
        # This is a GPU-friendly local-statistics contrast approximation,
        # not histogram-based OpenCV CLAHE.
        mode = "contrast"
    if mode not in ("contrast", "brightness", "combined"):
        raise ValueError(
            "SNORM_IMAGE_ENHANCE must be off, contrast, brightness, or combined"
        )
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, H, W] RGB tensor, got {tuple(images.shape)}")

    kernel_size = int(os.environ.get("SNORM_ENHANCE_KERNEL", "31"))
    kernel_size = max(3, min(kernel_size, 101))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Convert the detector's normalized tensor to RGB in [0, 1].
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
        # Lift only locally under-exposed regions; protect highlights.
        target_mean = float(os.environ.get("SNORM_ENHANCE_TARGET_LUMA", "0.38"))
        brightness_strength = float(os.environ.get("SNORM_ENHANCE_BRIGHTNESS", "0.45"))
        dark_region = (target_mean - local_mean).clamp_min(0.0)
        enhanced_y = enhanced_y + brightness_strength * dark_region * (1.0 - enhanced_y)

    if mode in ("contrast", "combined"):
        target_std = float(os.environ.get("SNORM_ENHANCE_TARGET_STD", "0.12"))
        contrast_strength = float(os.environ.get("SNORM_ENHANCE_CONTRAST", "0.8"))
        low_contrast = ((target_std - local_std) / max(target_std, 1e-4)).clamp(0.0, 1.0)
        gain = 1.0 + contrast_strength * low_contrast
        enhanced_y = enhanced_y + (luminance - local_mean) * (gain - 1.0)

    enhanced_y = enhanced_y.clamp(0.0, 1.0)
    # Scale RGB by luminance ratio to approximately preserve chromaticity.
    scale = (enhanced_y + 1e-4) / (luminance + 1e-4)
    enhanced_rgb = (rgb * scale).clamp(0.0, 1.0)
    return enhanced_rgb.mul(2.0).sub(1.0).to(dtype=images.dtype)
