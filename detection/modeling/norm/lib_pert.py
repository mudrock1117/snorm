import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import io

def local_color_jitter(x, strength=0.3):
    x = x * 0.5 + 0.5
    B, C, H, W = x.shape
    mask = torch.rand(B, 1, H // 128, W // 128, device=x.device)
    mask = (mask > 0.7).float()
    mask = F.interpolate(mask, size=(x.size(-2), x.size(-1)), mode='bilinear')

    brightness = 1 + (torch.rand(B,1,1,1, device=x.device)-0.5)*strength
    contrast = 1 + (torch.rand(B,1,1,1, device=x.device)-0.5)*strength

    mean = x.mean(dim=(2,3), keepdim=True)
    x_jitter = (x - mean) * contrast + mean
    x_jitter = x_jitter * brightness

    out = x * (1-mask) + x_jitter * mask
    out = (out - 0.5) / 0.5

    return out


def local_color_jitter_v2(x, patch_size=128):
     x = x * 0.5 + 0.5
     b, c, h, w = x.shape
     x_new = x.clone()

     for i in range(b):
         y, x_c = torch.randint(0, h-patch_size, (1,)), torch.randint(0, w-patch_size, (1,))
         shift = (torch.rand(c, 1, 1).to(x.device) - 0.5) * 0.4
         x_new[i, :, y:y+patch_size, x_c:x_c+patch_size] += shift
     out = (torch.clamp(x_new, 0, 1) - 0.5) / 0.5
     return out


def calc_mean_std(feat, eps=1e-8):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def user_adain(feats, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, bias=0.1, eps=1e-8, ema=0.1):
    w_a = cv2.resize(w_a, (feats.size(-1), feats.size(-2)), interpolation=cv2.INTER_LINEAR)
    w_u = cv2.resize(w_u, (feats.size(-1), feats.size(-2)), interpolation=cv2.INTER_LINEAR)
    bias = cv2.resize(bias, (feats.size(-1), feats.size(-2)), interpolation=cv2.INTER_LINEAR)

    w_a = torch.from_numpy(w_a).to(feats.device)
    w_u = torch.from_numpy(w_u).to(feats.device)
    bias = torch.from_numpy(bias).to(feats.device)

    pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set)
    feats_pool = pool_func(feats)
    feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
    pfeats = feats - feats_pool
    sfeats = pfeats ** 2
    sfeats = (pool_func(sfeats) + eps).sqrt()
    sfeats = F.interpolate(sfeats, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
    feats_mean, feats_std = calc_mean_std(feats)
    ba_ema_rate = ema
    sfeats = sfeats * ba_ema_rate + (1 - ba_ema_rate) * feats_std
    nfeats =   w_a * pfeats * sfeats / (sfeats + 1e-3)  + w_u * feats_pool + bias
    nfeats = torch.clip(nfeats, min=-1.0, max=1.0)

    return nfeats.float()

def user_domain_shift(feats):
    kernel_x = random.randint(1, 10)
    kernel_y = random.randint(1, 10)
    kernel_sz_set = [kernel_x, kernel_y]

    w_a = np.random.uniform(0.5, 1.5, (kernel_x, kernel_y))
    w_u = np.random.uniform(0.4, 2.0, (kernel_x, kernel_y))
    bias = np.random.uniform(0.8, 0.9, (kernel_x, kernel_y))
    ema = 0.9
    feats = user_adain(feats, kernel_sz_set=kernel_sz_set, w_a=w_a, w_u=w_u, bias=bias, ema=ema)
    feats = user_noise_v2(feats)
    return feats

def user_noise(x,kernel_sz_set=[5,3]):
    if random.random() > 0.5:
        return x
    b, c, h, w = x.size()
    mask = torch.ones((h, w)).to(x.device)
    h_start, w_start = h//4, w//4
    mask[h_start:h_start*2, w_start:w_start*2] = 0

    noise = torch.randn_like(x) * x.std() * 0.1
    return x + noise * (1 - mask)


def user_noise_v2(x, kernel_sz_set=[5,3]):
    x = x * 0.5 + 0.5
    B, C, H, W = x.shape
    mask = torch.rand(B, 1, kernel_sz_set[1], kernel_sz_set[0], device=x.device)
    mask = (mask > 0.7).float()
    mask = F.interpolate(mask, size=(x.size(-2), x.size(-1)), mode='bilinear')

    noise = torch.randn_like(x) * x.std() * 0.1
    out = x + noise * (1 - mask)
    out = (out - 0.5) / 0.5

    return out

def random_block_mask(
    x,
    p=0.5,
    min_ratio=0.02,
    max_ratio=0.4,
    value=0.0,
    threshold=0.5,
    kernel_sz=[5,5]
):
    B, C, H, W = x.shape
    import copy

    area_f = torch.rand((B, 1, kernel_sz[0], kernel_sz[1])).to(x.device)
    area_v = torch.rand(x.size()).to(x.device)

    area_f = F.interpolate(area_f, size=(x.size(-2), x.size(-1)), mode='bilinear')

    area_m = (area_f > threshold).float()

    x_0 = area_m * x + (1 - area_m) * area_v
    x_1 = (1 - area_m) * x + area_m * area_v

    return x_0, x_1, area_m

def random_block_noise(
    x,
    p=0.5,
    min_ratio=0.02,
    max_ratio=0.4,
    value=0.0,
    threshold=0.5,
    kernel_sz=[5,5]
):
    B, C, H, W = x.shape
    import copy

    area_f = torch.rand((B, 1, kernel_sz[0], kernel_sz[1])).to(x.device)
    area_v = torch.rand(x.size()).to(x.device)

    area_f = F.interpolate(area_f, size=(x.size(-2), x.size(-1)), mode='bilinear')

    area_m = (area_f > threshold).float()

    x_0 = area_m * x + (1 - area_m) * area_v
    x_0 = x * 0.9 + x_0 * 0.1

    x_1 = (1 - area_m) * x + area_m * area_v

    return x_0, x_1, area_m

def night_to_day_simple(x, gamma=0.4, gain=1.2):
    x = torch.clamp(x, 0, 1)
    x = x ** gamma
    x = x * gain
    return torch.clamp(x, 0, 1)

def local_brightness_contrast(
    x,
    alpha_range=(0.7, 1.3),
    beta_range=(-0.2, 0.2),
    patch_size=(30, 30)
):
    B, C, H, W = x.shape

    out = x.clone()

    for b in range(B):
        ph, pw = patch_size
        top = torch.randint(0, H - ph + 1, (1,)).item()
        left = torch.randint(0, W - pw + 1, (1,)).item()

        patch = out[b, :, top:top+ph, left:left+pw]

        alpha = torch.empty(1).uniform_(*alpha_range).to(x.device)
        beta  = torch.empty(1).uniform_(*beta_range).to(x.device)

        mean = patch.mean(dim=(1,2), keepdim=True)
        patch = alpha * (patch - mean) + mean + beta
        out[b, :, top:top+ph, left:left+pw] = patch

    out = out.clamp(0, 1)
    return out

def local_gaussian_noise(
    x,
    patch_size=(30, 30),
    sigma_range=(0.02, 0.1)
):
    B, C, H, W = x.shape
    out = x.clone()

    for b in range(B):
        ph, pw = patch_size
        top = torch.randint(0, H - ph + 1, (1,)).item()
        left = torch.randint(0, W - pw + 1, (1,)).item()

        sigma = torch.empty(1).uniform_(*sigma_range).to(x.device)
        noise = torch.randn(C, ph, pw, device=x.device) * sigma

        out[b, :, top:top+ph, left:left+pw] += noise

    return out.clamp(0, 1)


def local_transform(x, kernel_sz=[1,1]):
    x = x * 0.5 + 0.5
    if random.random() < 0.7:
        x = local_bright_contrast(x, kernel_sz=kernel_sz)
    if random.random() < 0.5:
        x = local_gaussian_noise(x)
    x = (x - 0.5) / 0.5
    return x


def local_bright_contrast(feats, kernel_sz_set=[7,7], w_a=(0.2, 1.8), w_u=(0.5, 2.0), kernel_sz=None):
        kernel_sz_set = kernel_sz
        B,C, H, W = feats.size()
        device = feats.device
        pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set)
        feats_pool = pool_func(feats)

        feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        pfeats = feats - feats_pool

        brightness = torch.empty(B, 1, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*w_u)
        contrast = torch.empty(B, 1, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*w_a)
        brightness = brightness.expand(B, C, kernel_sz_set[0], kernel_sz_set[1])
        contrast = contrast.expand(B, C, kernel_sz_set[0], kernel_sz_set[1])

        brightness = F.interpolate(brightness, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        contrast = F.interpolate(contrast, size=(feats.size(-2), feats.size(-1)), mode='bicubic')

        nfeats = contrast * pfeats  + brightness * feats_pool
        nfeats = torch.clip(nfeats, min=0.0, max=1.0)
        return nfeats.float()


def gaussian_noise(x, sigma_range=(0.01, 0.1)):
    sigma = torch.empty(1).uniform_(*sigma_range).to(x.device)
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0, 1)


def impulse_noise(x, amount_range=(0.01, 0.1)):
    amount = torch.empty(1).uniform_(*amount_range).to(x.device)
    B, C, H, W = x.shape
    noise = torch.rand(B, C, H, W, device=x.device)
    x_noisy = x.clone()
    x_noisy[noise < amount / 2] = 0
    x_noisy[noise > 1 - amount / 2] = 1
    return x_noisy


def shot_noise(x, intensity_range=(50, 200)):
    intensity = torch.empty(1).uniform_(*intensity_range).to(x.device)
    noise = torch.poisson(x * intensity) / intensity
    return torch.clamp(noise, 0, 1)


def gaussian_blur(x, kernel_size_range=(3, 11)):
    kernel_size = random.randint(kernel_size_range[0], kernel_size_range[1])
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    sigma = torch.tensor(random.uniform(0.5, 2.0), device=x.device, dtype=x.dtype)
    gaussian = torch.tensor([[torch.exp(-((i - kernel_size//2)**2 + (j - kernel_size//2)**2) / (2*sigma**2)) 
                            for j in range(kernel_size)] for i in range(kernel_size)], 
                           device=x.device, dtype=x.dtype)
    gaussian = gaussian / gaussian.sum()
    gaussian = gaussian.unsqueeze(0).unsqueeze(0).repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, gaussian, padding=padding, groups=x.size(1))


def motion_blur(x, kernel_size_range=(5, 15)):
    kernel_size = random.randint(kernel_size_range[0], kernel_size_range[1])
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    angle = random.uniform(0, 180)
    rad = angle * np.pi / 180
    center = kernel_size // 2
    kernel = torch.zeros(kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    for i in range(kernel_size):
        for j in range(kernel_size):
            if np.abs(np.cos(rad) * (j - center) + np.sin(rad) * (i - center)) < 1:
                kernel[i, j] = 1
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=padding, groups=x.size(1))


def zoom_blur(x, strength_range=(0.1, 0.5)):
    strength = random.uniform(*strength_range)
    B, C, H, W = x.shape
    blurred = x.clone()
    for _ in range(3):
        scale = 1 + strength * random.uniform(0, 1)
        h, w = int(H / scale), int(W / scale)
        zoomed = F.interpolate(x, size=(h, w), mode='bicubic')
        zoomed = F.interpolate(zoomed, size=(H, W), mode='bicubic')
        blurred = blurred * 0.5 + zoomed * 0.5
    return blurred


def fog(x, density_range=(0.1, 0.5)):
    density = random.uniform(*density_range)
    B, C, H, W = x.shape
    fog_color = torch.ones(B, C, H, W, device=x.device) * 0.7
    return torch.clamp(x * (1 - density) + fog_color * density, 0, 1)


def frost(x, intensity_range=(0.1, 0.3)):
    intensity = random.uniform(*intensity_range)
    B, C, H, W = x.shape
    frost_pattern = torch.rand(B, C, H, W, device=x.device)
    frost_pattern = gaussian_blur(frost_pattern, kernel_size_range=(3, 7))
    return torch.clamp(x * (1 - intensity) + frost_pattern * intensity, 0, 1)


def snow(x, intensity_range=(0.01, 0.05)):
    intensity = random.uniform(*intensity_range)
    B, C, H, W = x.shape
    snowflakes = torch.rand(B, C, H, W, device=x.device)
    snow_mask = (snowflakes > 1 - intensity).float()
    return torch.clamp(x + snow_mask * intensity, 0, 1)


def contrast_change(x, factor_range=(0.3, 1.8)):
    factor = random.uniform(*factor_range)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return torch.clamp((x - mean) * factor + mean, 0, 1)


def brightness_change(x, factor_range=(0.3, 1.5)):
    factor = random.uniform(*factor_range)
    return torch.clamp(x * factor, 0, 1)


def saturation_change(x, factor_range=(0.0, 1.5)):
    factor = random.uniform(*factor_range)
    gray = x.mean(dim=1, keepdim=True)
    return torch.clamp(x * factor + gray * (1 - factor), 0, 1)


def pixelate(x, block_size_range=(2, 8)):
    block_size = random.randint(*block_size_range)
    B, C, H, W = x.shape
    downscaled = F.interpolate(x, size=(H // block_size, W // block_size), mode='nearest')
    return F.interpolate(downscaled, size=(H, W), mode='nearest')


def jpeg_compression(x, quality_range=(10, 80)):
    quality = random.randint(*quality_range)
    import PIL
    from PIL import Image
    B, C, H, W = x.shape
    result = []
    for i in range(B):
        img = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_decompressed = Image.open(buffer)
        result.append(torch.from_numpy(np.array(img_decompressed)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(result).to(x.device)


def elastic_transform(x, alpha_range=(10, 50), sigma_range=(3, 10)):
    alpha = random.uniform(*alpha_range)
    sigma = random.uniform(*sigma_range)
    B, C, H, W = x.shape
    grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1).to(x.device), 
                         size=(B, C, H, W), align_corners=False)
    noise = gaussian_blur(torch.randn(B, 2, H, W, device=x.device), kernel_size_range=(3, 7)) * alpha
    grid = grid + noise.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, align_corners=False)


def _gen_region_mask(B, H, W, n_regions=4, device='cuda'):
    """生成空间分区域掩码，将H×W分为n_regions个区域"""
    gh = max(H // n_regions, 2)
    gw = max(W // n_regions, 2)
    raw = torch.rand(B, 1, gh, gw, device=device)
    raw = (raw * n_regions).int().float() / n_regions
    mask = F.interpolate(raw, size=(H, W), mode='bilinear', align_corners=False)
    mask = (mask * n_regions).round() / n_regions
    return mask


def spatial_fog_night_degrade(feats):
    """空间分区域退化：将特征图分为多个区域，分别施加雾天/夜晚/无退化

    返回两个不同退化版本，让CrossAttentionFusion学习从不同质量输入中选取最优部分。

    输入: feats (B,C,H,W), 范围[-1,1]
    输出: feats_deg0, feats_deg1, feats_guide (均为B,C,H,W, 范围[-1,1])
    """
    B, C, H, W = feats.shape
    device = feats.device

    # 转到[0,1]方便退化操作
    x = feats * 0.5 + 0.5

    # 生成两个不同的空间区域掩码
    n_regions = random.choice([2, 3, 4])
    mask_0 = _gen_region_mask(B, H, W, n_regions, device)
    mask_1 = _gen_region_mask(B, H, W, n_regions, device)

    # ---- 退化方式池 ----
    # 雾天退化：高斯模糊、雾、对比度降低
    # 夜晚退化：高斯噪声、散粒噪声、亮度降低
    # 保留：无退化

    # 退化版本0：偏向雾天
    x_fog = fog(gaussian_blur(x.clone()), density_range=(0.15, 0.35))
    x_fog = contrast_change(x_fog, factor_range=(0.6, 0.9))

    # 退化版本1：偏夜晚
    x_night = gaussian_noise(x.clone(), sigma_range=(0.03, 0.12))
    x_night = brightness_change(x_night, factor_range=(0.3, 0.7))

    # 干净版本
    x_clean = x.clone()

    # 按掩码混合：版本0 = 部分雾天退化 + 部分夜晚退化 + 部分干净
    # 用mask_0区分区域
    deg0 = mask_0 * x_fog + (1 - mask_0) * x_clean

    # 版本1 = 部分夜晚退化 + 部分雾天退化 + 部分干净
    # 用mask_1区分区域（与mask_0不同，确保两个版本退化分布不同）
    deg1 = mask_1 * x_night + (1 - mask_1) * x_clean

    # 引导特征：混合退化（全图轻微退化）
    guide = 0.5 * x_fog + 0.5 * x_night

    # 转回[-1,1]
    deg0 = (deg0 - 0.5) / 0.5
    deg1 = (deg1 - 0.5) / 0.5
    guide = (guide - 0.5) / 0.5

    return deg0, deg1, guide