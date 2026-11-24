"""CD-FKD utilities isolated behind ``SNORM_EXPERIMENT=CDFKD``."""
import math
import os
import random

import torch
from torch import nn
import torch.nn.functional as F


class SpatialEnhancementAdapter(nn.Module):
    """Small spatial residual adapter with exact identity initialization."""

    def __init__(self, hidden_channels=64, residual_scale=0.10):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.project_in = nn.Conv2d(3, hidden_channels, 1, bias=True)
        self.spatial = nn.Conv2d(
            hidden_channels, hidden_channels, 3, padding=1,
            groups=hidden_channels, bias=True,
        )
        self.mix = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=True)
        self.project_out = nn.Conv2d(hidden_channels, 3, 1, bias=True)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, image):
        residual = self.project_in(image)
        residual = F.gelu(self.spatial(residual))
        residual = F.gelu(self.mix(residual))
        residual = self.residual_scale * torch.tanh(self.project_out(residual))
        return image + residual

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class ConservativeSourceDiversification(object):
    """Source-only appearance corruption with geometry-preserving downscale."""

    def __init__(self):
        self.apply_prob = float(os.environ.get('SNORM_CDFKD_AUG_PROB', '0.8'))
        self.down_min = float(os.environ.get('SNORM_CDFKD_DOWNSCALE_MIN', '0.8'))
        self.down_max = float(os.environ.get('SNORM_CDFKD_DOWNSCALE_MAX', '1.0'))
        if not (0.0 < self.down_min <= self.down_max <= 1.0):
            raise ValueError('CDFKD downscale range must satisfy 0 < min <= max <= 1')
        default_ops = (
            'gaussian_noise,shot_noise,defocus_blur,motion_blur,'
            'contrast,brightness,pixelate'
        )
        self.operations = tuple(
            value.strip() for value in
            os.environ.get('SNORM_CDFKD_AUGS', default_ops).split(',')
            if value.strip()
        )

    @staticmethod
    def _gaussian_blur(image, sigma):
        radius = max(1, int(math.ceil(2.0 * sigma)))
        axis = torch.arange(-radius, radius + 1, device=image.device,
                            dtype=image.dtype)
        kernel = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        channels = image.size(1)
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        image = F.conv2d(image, horizontal, padding=(0, radius), groups=channels)
        return F.conv2d(image, vertical, padding=(radius, 0), groups=channels)

    @staticmethod
    def _motion_blur(image):
        size = random.choice((3, 5))
        kernel = image.new_zeros(size, size)
        if random.random() < 0.5:
            kernel[size // 2, :] = 1.0 / size
        else:
            kernel[:, size // 2] = 1.0 / size
        channels = image.size(1)
        kernel = kernel.view(1, 1, size, size).expand(channels, 1, size, size)
        return F.conv2d(image, kernel, padding=size // 2, groups=channels)

    def __call__(self, images):
        image = (images * 0.5 + 0.5).clamp(0.0, 1.0)
        operation = 'none'
        if self.operations and random.random() < self.apply_prob:
            operation = random.choice(self.operations)
            if operation == 'gaussian_noise':
                image = image + torch.randn_like(image) * random.uniform(0.005, 0.02)
            elif operation == 'shot_noise':
                levels = random.uniform(80.0, 160.0)
                image = torch.poisson(image * levels) / levels
            elif operation == 'defocus_blur':
                image = self._gaussian_blur(image, random.uniform(0.3, 0.8))
            elif operation == 'motion_blur':
                image = self._motion_blur(image)
            elif operation == 'contrast':
                mean = image.mean((2, 3), keepdim=True)
                image = mean + random.uniform(0.85, 1.15) * (image - mean)
            elif operation == 'brightness':
                image = image * random.uniform(0.85, 1.15)
            elif operation == 'pixelate':
                height, width = image.shape[-2:]
                ratio = random.uniform(0.75, 0.90)
                small = F.interpolate(
                    image, size=(max(1, int(height * ratio)), max(1, int(width * ratio))),
                    mode='bilinear', align_corners=False,
                )
                image = F.interpolate(small, size=(height, width), mode='nearest')
            else:
                raise ValueError('Unknown CDFKD augmentation: %s' % operation)

        height, width = image.shape[-2:]
        ratio = random.uniform(self.down_min, self.down_max)
        if ratio < 0.999999:
            small = F.interpolate(
                image, size=(max(1, int(height * ratio)), max(1, int(width * ratio))),
                mode='bilinear', align_corners=False,
            )
            image = F.interpolate(
                small, size=(height, width), mode='bilinear', align_corners=False,
            )
        image = image.clamp(0.0, 1.0)
        return (image - 0.5) / 0.5, {'operation': operation, 'downscale_ratio': ratio}


def global_feature_distillation(teacher_feature, student_feature):
    if teacher_feature.shape[-2:] != student_feature.shape[-2:]:
        student_feature = F.interpolate(
            student_feature, size=teacher_feature.shape[-2:],
            mode='bilinear', align_corners=False,
        )
    teacher_vector = teacher_feature.detach().flatten(1)
    student_vector = student_feature.flatten(1)
    return (1.0 - F.cosine_similarity(
        teacher_vector, student_vector, dim=1, eps=1e-8,
    )).mean()


def valid_gt_boxes(targets, img_metas):
    boxes_per_image = []
    for target, meta in zip(targets, img_metas):
        boxes = target['boxes']
        width, height = meta['img_shape']
        boxes = boxes.clone()
        boxes[:, 0::2].clamp_(0, float(width))
        boxes[:, 1::2].clamp_(0, float(height))
        valid = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & (
            (boxes[:, 3] - boxes[:, 1]) >= 1.0
        )
        boxes_per_image.append(boxes[valid])
    return boxes_per_image


def instance_feature_distillation(teacher_feature, student_feature, pooler,
                                   targets, img_metas):
    boxes = valid_gt_boxes(targets, img_metas)
    if sum(box.size(0) for box in boxes) == 0:
        return student_feature.sum() * 0.0
    teacher_roi = pooler(teacher_feature.detach(), boxes).flatten(1)
    student_roi = pooler(student_feature, boxes).flatten(1)
    return (1.0 - F.cosine_similarity(
        teacher_roi, student_roi, dim=1, eps=1e-8,
    )).mean()
