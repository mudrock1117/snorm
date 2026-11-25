"""CD-FKD utilities isolated behind ``SNORM_EXPERIMENT=CDFKD``."""
import math
import os
import random

import torch
from torch import nn
import torch.nn.functional as F


_PRESETS = {
    'baseline': dict(adapter_type='basic', rgb_adapter=True,
                     feature_adapter=False, div_strength='conservative',
                     global_weight=0.5, instance_weight=1.0),
    'det_dominant': dict(adapter_type='basic', rgb_adapter=True,
                         feature_adapter=False, div_strength='conservative',
                         global_weight=0.1, instance_weight=0.25),
    'multiscale': dict(adapter_type='multiscale', rgb_adapter=True,
                       feature_adapter=False, div_strength='conservative',
                       global_weight=0.1, instance_weight=0.25),
    'feature': dict(adapter_type='basic', rgb_adapter=False,
                    feature_adapter=True, div_strength='conservative',
                    global_weight=0.1, instance_weight=0.25),
    'combined': dict(adapter_type='multiscale', rgb_adapter=True,
                     feature_adapter=True, div_strength='medium',
                     global_weight=0.1, instance_weight=0.25),
    'feature_original_only': dict(
        adapter_type='basic', rgb_adapter=False, feature_adapter=True,
        div_strength='conservative', source_diversification=False,
        global_weight=0.0, instance_weight=0.0, instance_mode='cosine',
        hard_object=False, hard_object_weight=0.0,
    ),
    'feature_hard_object': dict(
        adapter_type='basic', rgb_adapter=False, feature_adapter=True,
        div_strength='conservative', global_weight=0.0,
        instance_weight=0.0, instance_mode='cosine', hard_object=True,
        hard_object_weight=0.1, hard_min_weight=0.5, hard_max_weight=2.0,
    ),
}


def _env_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def resolve_cdfkd_config(environ=None):
    """Resolve presets first, then explicit environment overrides."""
    env = os.environ if environ is None else environ
    preset = env.get('SNORM_CDFKD_PRESET', 'baseline').strip().lower()
    if preset not in _PRESETS:
        raise ValueError('Unknown SNORM_CDFKD_PRESET: %s' % preset)
    config = dict(_PRESETS[preset])
    if _env_bool(env.get('SNORM_CDFKD_ACCURACY_MODE', '0')):
        if 'SNORM_CDFKD_GLOBAL_W' not in env:
            config['global_weight'] = 0.1
        if 'SNORM_CDFKD_INSTANCE_W' not in env:
            config['instance_weight'] = 0.25
    overrides = {
        'SNORM_CDFKD_ADAPTER_TYPE': ('adapter_type', str),
        'SNORM_CDFKD_RGB_ADAPTER': ('rgb_adapter', _env_bool),
        'SNORM_CDFKD_FEATURE_ADAPTER': ('feature_adapter', _env_bool),
        'SNORM_CDFKD_DIV_STRENGTH': ('div_strength', str),
        'SNORM_CDFKD_SOURCE_DIVERSIFICATION': ('source_diversification', _env_bool),
        'SNORM_CDFKD_GLOBAL_W': ('global_weight', float),
        'SNORM_CDFKD_INSTANCE_W': ('instance_weight', float),
        'SNORM_CDFKD_HARD_OBJECT': ('hard_object', _env_bool),
        'SNORM_CDFKD_HARD_OBJECT_W': ('hard_object_weight', float),
        'SNORM_CDFKD_HARD_MIN_W': ('hard_min_weight', float),
        'SNORM_CDFKD_HARD_MAX_W': ('hard_max_weight', float),
    }
    for name, (key, parser) in overrides.items():
        if name in env:
            config[key] = parser(env[name])
    config.update({
        'preset': preset,
        'det_weight': float(env.get('SNORM_CDFKD_DET_W', '1.0')),
        'instance_mode': env.get(
            'SNORM_CDFKD_INSTANCE_MODE',
            config.get('instance_mode', 'cosine')).lower(),
        'rgb_residual_scale': float(env.get('SNORM_CDFKD_RESIDUAL_SCALE', '0.10')),
        'feature_residual_scale': float(env.get(
            'SNORM_CDFKD_FEATURE_RESIDUAL_SCALE', '0.10')),
        'rgb_channels': int(env.get('SNORM_CDFKD_ADAPTER_CHANNELS', '64')),
        'feature_channels': int(env.get('SNORM_CDFKD_FEATURE_CHANNELS', '32')),
    })
    config.setdefault('source_diversification', True)
    config.setdefault('hard_object', False)
    config.setdefault('hard_object_weight', 0.0)
    config.setdefault('hard_min_weight', 0.5)
    config.setdefault('hard_max_weight', 2.0)
    if config['adapter_type'] not in ('basic', 'multiscale'):
        raise ValueError('CDFKD adapter type must be basic or multiscale')
    if config['div_strength'] not in ('conservative', 'medium', 'strong'):
        raise ValueError('CDFKD diversification strength is invalid')
    if config['instance_mode'] not in ('cosine', 'weighted'):
        raise ValueError('CDFKD instance mode must be cosine or weighted')
    if not config['rgb_adapter'] and not config['feature_adapter']:
        raise ValueError('CDFKD requires at least one enabled adapter')
    for key in ('det_weight', 'global_weight', 'instance_weight',
                'hard_object_weight'):
        if config[key] < 0.0:
            raise ValueError('CDFKD %s must be non-negative' % key)
    if not (0.0 < config['hard_min_weight'] <= config['hard_max_weight']):
        raise ValueError('CDFKD hard weights require 0 < min <= max')
    return config


def hard_positive_detection_loss(class_logits, labels, min_weight=0.5,
                                 max_weight=2.0):
    """Reweight positive source GT ROIs by detached classification hardness."""
    if isinstance(labels, (list, tuple)):
        labels = torch.cat(labels, dim=0)
    positive_mask = labels > 0
    positive_rois = int(positive_mask.sum().item())
    if positive_rois == 0:
        zero = class_logits.sum() * 0.0
        stats = {
            'positive_rois': 0,
            'mean_gt_prob': zero.detach(),
            'mean_hardness': zero.detach(),
            'weight_mean': zero.detach(),
            'weight_min': zero.detach(),
            'weight_max': zero.detach(),
            'hard_loss': zero.detach(),
        }
        return zero, stats
    positive_logits = class_logits[positive_mask]
    positive_labels = labels[positive_mask]
    ce = F.cross_entropy(positive_logits, positive_labels, reduction='none')
    probs = F.softmax(positive_logits, dim=1)
    gt_prob = probs.gather(1, positive_labels.unsqueeze(1)).squeeze(1)
    hardness = -torch.log(gt_prob.clamp_min(1e-8))
    raw_weight = hardness.detach()
    weight = raw_weight / raw_weight.mean().clamp_min(1e-8)
    weight = weight.clamp(min_weight, max_weight)
    weight = weight / weight.mean().clamp_min(1e-8)
    hard_loss = (weight * ce).mean()
    stats = {
        'positive_rois': positive_rois,
        'mean_gt_prob': gt_prob.mean().detach(),
        'mean_hardness': hardness.mean().detach(),
        'weight_mean': weight.mean().detach(),
        'weight_min': weight.min().detach(),
        'weight_max': weight.max().detach(),
        'hard_loss': hard_loss.detach(),
    }
    return hard_loss, stats


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
        self.last_residual = None

    def forward(self, image):
        residual = self.project_in(image)
        residual = F.gelu(self.spatial(residual))
        residual = F.gelu(self.mix(residual))
        residual = self.residual_scale * torch.tanh(self.project_out(residual))
        self.last_residual = residual.detach()
        return image + residual

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class MultiScaleSpatialEnhancementAdapter(nn.Module):
    """Three receptive fields with a zero-initialized RGB residual head."""

    def __init__(self, hidden_channels=64, residual_scale=0.10):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.stem = nn.Conv2d(3, hidden_channels, 1, bias=True)
        self.branches = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=dilation,
                      dilation=dilation, groups=hidden_channels, bias=True)
            for dilation in (1, 2, 4)
        ])
        self.fusion = nn.Conv2d(hidden_channels * 3, hidden_channels, 1,
                                bias=True)
        self.project_out = nn.Conv2d(hidden_channels, 3, 1, bias=True)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)
        self.last_residual = None

    def forward(self, image):
        stem = F.gelu(self.stem(image))
        spatial = torch.cat([F.gelu(branch(stem))
                             for branch in self.branches], dim=1)
        residual = F.gelu(self.fusion(spatial))
        residual = self.residual_scale * torch.tanh(self.project_out(residual))
        self.last_residual = residual.detach()
        return image + residual

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class FeatureResidualAdapter(nn.Module):
    """Shape-preserving bottleneck correction for the single C4 tensor."""

    def __init__(self, in_channels=1024, hidden_channels=32,
                 residual_scale=0.10):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.reduce = nn.Conv2d(in_channels, hidden_channels, 1, bias=True)
        self.spatial = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                                 groups=hidden_channels, bias=True)
        self.project = nn.Conv2d(hidden_channels, in_channels, 1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.last_residual = None

    def forward(self, feature):
        residual = F.gelu(self.reduce(feature))
        residual = F.gelu(self.spatial(residual))
        residual = self.residual_scale * torch.tanh(self.project(residual))
        self.last_residual = residual.detach()
        return feature + residual

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class ConservativeSourceDiversification(object):
    """Source-only appearance corruption with geometry-preserving downscale."""

    _RANGES = {
        'conservative': dict(down=(0.80, 1.00), noise=(0.005, 0.020),
                             shot=(80.0, 160.0), blur=(0.3, 0.8),
                             contrast=(0.85, 1.15), brightness=(0.85, 1.15),
                             pixelate=(0.75, 0.90)),
        'medium': dict(down=(0.70, 1.00), noise=(0.008, 0.030),
                       shot=(60.0, 140.0), blur=(0.4, 1.0),
                       contrast=(0.75, 1.25), brightness=(0.75, 1.25),
                       pixelate=(0.65, 0.88)),
        'strong': dict(down=(0.60, 1.00), noise=(0.010, 0.040),
                       shot=(45.0, 120.0), blur=(0.5, 1.2),
                       contrast=(0.65, 1.35), brightness=(0.65, 1.35),
                       pixelate=(0.55, 0.85)),
    }

    def __init__(self, strength='conservative'):
        self.strength = strength
        ranges = self._RANGES[strength]
        self.apply_prob = float(os.environ.get('SNORM_CDFKD_AUG_PROB', '0.8'))
        self.down_min = float(os.environ.get('SNORM_CDFKD_DOWNSCALE_MIN', ranges['down'][0]))
        self.down_max = float(os.environ.get('SNORM_CDFKD_DOWNSCALE_MAX', ranges['down'][1]))
        self.ranges = ranges
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
                image = image + torch.randn_like(image) * random.uniform(*self.ranges['noise'])
            elif operation == 'shot_noise':
                levels = random.uniform(*self.ranges['shot'])
                image = torch.poisson(image * levels) / levels
            elif operation == 'defocus_blur':
                image = self._gaussian_blur(image, random.uniform(*self.ranges['blur']))
            elif operation == 'motion_blur':
                image = self._motion_blur(image)
            elif operation == 'contrast':
                mean = image.mean((2, 3), keepdim=True)
                image = mean + random.uniform(*self.ranges['contrast']) * (image - mean)
            elif operation == 'brightness':
                image = image * random.uniform(*self.ranges['brightness'])
            elif operation == 'pixelate':
                height, width = image.shape[-2:]
                ratio = random.uniform(*self.ranges['pixelate'])
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
        return (image - 0.5) / 0.5, {
            'operation': operation, 'downscale_ratio': ratio,
            'strength': self.strength,
        }


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
                                   targets, img_metas, mode='cosine'):
    boxes = valid_gt_boxes(targets, img_metas)
    if sum(box.size(0) for box in boxes) == 0:
        return student_feature.sum() * 0.0
    teacher_roi = pooler(teacher_feature.detach(), boxes).flatten(1)
    student_roi = pooler(student_feature, boxes).flatten(1)
    losses = 1.0 - F.cosine_similarity(
        teacher_roi, student_roi, dim=1, eps=1e-8,
    )
    if mode == 'cosine':
        return losses.mean()
    # Frozen-reference ROI activation is a stable source-only proxy for
    # detection salience; detach prevents weighting from training the teacher.
    salience = teacher_roi.detach().pow(2).mean(dim=1).sqrt()
    weights = salience / salience.mean().clamp_min(1e-8)
    weights = weights.clamp(0.25, 4.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)
