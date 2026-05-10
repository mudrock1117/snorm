import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2

def local_color_jitter(x, strength=0.3):
    """
    x: [B,3,H,W], range [0,1]
    """
    x = x * 0.5 + 0.5
    B, C, H, W = x.shape
    mask = torch.rand(B, 1, H // 128, W // 128, device=x.device)
    mask = (mask > 0.7).float()   # 局部区域
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
    # eps is a small value added to the variance to avoid divide-by-zero.
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
    
    #w_a = F.interpolate(w_a, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
    #w_u = F.interpolate(w_u, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
    
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
    
    #w_a = random.uniform(0.5, 1.5, size=(kernel_x, kernel_y))
    #w_u = random.uniform(0.4, 2.0, size=(kernel_x, kernel_y))
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
    """
    x: [B,3,H,W], range [0,1]
    """
    x = x * 0.5 + 0.5
    B, C, H, W = x.shape
    mask = torch.rand(B, 1, kernel_sz_set[1], kernel_sz_set[0], device=x.device)
    mask = (mask > 0.7).float()   # 局部区域
    mask = F.interpolate(mask, size=(x.size(-2), x.size(-1)), mode='bilinear')

    noise = torch.randn_like(x) * x.std() * 0.1
    out = x + noise * (1 - mask)
    out = (out - 0.5) / 0.5

    return out

###########################################################################
'''
def random_block_mask(
    x,
    p=0.5,                # 每张图被 mask 的概率
    min_ratio=0.02,       # mask 最小面积比例
    max_ratio=0.4,        # mask 最大面积比例
    value=0.0             # mask 填充值: 0 / "random"
):
    
    B, C, H, W = x.shape
    assert C == 3

    x = copy.deepcopy(x)
    #x_mask = torch.zeros_like(x[:, 0:1])

    for i in range(B):
        if torch.rand(1, device=x.device) > p:
            continue

        area = H * W
        target_area = torch.empty(1, device=x.device).uniform_(min_ratio, max_ratio) * area
        aspect_ratio = torch.empty(1, device=x.device).uniform_(0.3, 3.3)

        h = int(torch.sqrt(target_area * aspect_ratio))
        w = int(torch.sqrt(target_area / aspect_ratio))

        if h >= H or w >= W:
            continue

        y = torch.randint(0, H - h + 1, (1,), device=x.device)
        x0 = torch.randint(0, W - w + 1, (1,), device=x.device)

        if value == "random_m":
            x[i, :, y:y+h, x0:x0+w] = torch.rand(
                (C, h, w), device=x.device
            )
        elif value == "random_v":
            #print('-----random_v')
            ############random value############
            #v = torch.rand(
            #    (1, 1, 1), device=x.device
            #)
            #v = v.expand((C, h, w))
            #x[i, :, y:y+h, x0:x0+w] = v
            
            ############random noise############
            v = torch.rand(
                (C, h, w), device=x.device
            )
            x[i, :, y:y+h, x0:x0+w] = v
            #x_mask[i, :, y:y+h, x0:x0+w] = 1
            
        else:
            x[i, :, y:y+h, x0:x0+w] = value

    return x
'''

def random_block_mask(
    x,
    p=0.5,                # 每张图被 mask 的概率
    min_ratio=0.02,       # mask 最小面积比例
    max_ratio=0.4,        # mask 最大面积比例
    value=0.0,             # mask 填充值: 0 / "random"
    threshold=0.5,
    kernel_sz=[5,5]
):
    """
    x: Tensor [B, 3, H, W]
    """
    B, C, H, W = x.shape
    #assert C == 3
    import copy
    #x = copy.deepcopy(x)
    #x_mask = torch.zeros_like(x[:, 0:1])
    
    #area_f = torch.rand((B, 1, 30, 30)).to(x.device)
    area_f = torch.rand((B, 1, kernel_sz[0], kernel_sz[1])).to(x.device)
    area_v = torch.rand(x.size()).to(x.device)
    
    area_f = F.interpolate(area_f, size=(x.size(-2), x.size(-1)), mode='bilinear')
    
    area_m = (area_f > threshold).float()
    
    x_0 = area_m * x + (1 - area_m) * area_v
    x_1 = (1 - area_m) * x + area_m * area_v

    return x_0, x_1, area_m

def random_block_noise(
    x,
    p=0.5,                # 每张图被 mask 的概率
    min_ratio=0.02,       # mask 最小面积比例
    max_ratio=0.4,        # mask 最大面积比例
    value=0.0,             # mask 填充值: 0 / "random"
    threshold=0.5,
    kernel_sz=[5,5]
):
    """
    x: Tensor [B, 3, H, W]
    """
    B, C, H, W = x.shape
    #assert C == 3
    import copy
    #x = copy.deepcopy(x)
    #x_mask = torch.zeros_like(x[:, 0:1])
    
    #area_f = torch.rand((B, 1, 30, 30)).to(x.device)
    area_f = torch.rand((B, 1, kernel_sz[0], kernel_sz[1])).to(x.device)
    area_v = torch.rand(x.size()).to(x.device)
    
    area_f = F.interpolate(area_f, size=(x.size(-2), x.size(-1)), mode='bilinear')
    
    area_m = (area_f > threshold).float()
    
    x_0 = area_m * x + (1 - area_m) * area_v
    x_0 = x * 0.9 + x_0 * 0.1
    
    x_1 = (1 - area_m) * x + area_m * area_v

    return x_0, x_1, area_m
    
#######################night-to-day##########################
def night_to_day_simple(x, gamma=0.4, gain=1.2):
    """
    x: [B,3,H,W] in [0,1]
    """
    # Gamma 校正（提亮暗部）
    x = torch.clamp(x, 0, 1)
    x = x ** gamma

    # 增强整体亮度
    x = x * gain

    return torch.clamp(x, 0, 1)

#############################################################
def local_brightness_contrast(
    x,
    #alpha_range=(0.7, 1.3),
    alpha_range=(0.1, 0.2),
    beta_range=(-0.2, 0.2),
    patch_size=(30, 30)
):
    """
    x: [B, 3, H, W]
    """
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
        print('---')
        patch = alpha * (patch - mean) + mean + beta
        out[b, :, top:top+ph, left:left+pw] = patch
        
    out = out.clamp(0, 1)
    return out

def local_gaussian_noise(
    x,
    patch_size=(30, 30),
    sigma_range=(0.02, 0.1)
):
    """
    x: [B, 3, H, W], assume in [0,1]
    """
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
        #kernel_sz = random.randint(5, 13)
        #kernel_sz_set = [kernel_sz, kernel_sz]
        kernel_sz_set = kernel_sz
        B,C, H, W = feats.size()
        device = feats.device
        pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set)
        feats_pool = pool_func(feats)

        feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        pfeats = feats - feats_pool

        ###
        brightness = torch.empty(B, 1, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*w_u)
        contrast = torch.empty(B, 1, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*w_a)
        brightness = brightness.expand(B, C, kernel_sz_set[0], kernel_sz_set[1])
        contrast = contrast.expand(B, C, kernel_sz_set[0], kernel_sz_set[1])
        
        brightness = F.interpolate(brightness, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        contrast = F.interpolate(contrast, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        ###
        
        nfeats = contrast * pfeats  + brightness * feats_pool
        #nfeats = contrast * pfeats  + feats_pool
        #nfeats = 0.0 * pfeats  + feats_pool
        nfeats = torch.clip(nfeats, min=0.0, max=1.0)
        return nfeats.float()
    
