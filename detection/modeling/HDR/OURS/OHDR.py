import torch
import torch.nn as nn
import torch.nn.functional as F
from detection.modeling.norm.lib_pert import *


def box_blur(x, kernel_size):
    padding = kernel_size // 2
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    C = x.shape[1]
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype) / (kernel_size * kernel_size)
    kernel = kernel.repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, groups=C)


# ============================================================
# 1. Multi-Scale Encoder
# ============================================================
class MultiScaleEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


# ============================================================
# 2. Local Window Attention (Window-size aware)
# ============================================================
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, ws * ws, C)
        return x

    def window_reverse(self, windows, H, W):
        ws = self.window_size
        B = int(windows.shape[0] / (H * W / ws / ws))
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, -1, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[-2:]

        x_windows = self.window_partition(x)

        qkv = self.qkv(x_windows)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / (self.dim ** 0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = self.proj(out)

        x = self.window_reverse(out, Hp, Wp)

        x = x[:, :, :H, :W]
        return x


# ============================================================
# 3. Cross-Attention Fusion (Simplified for detection task)
#    - 移除了不适用的有效性/置信度/一致性三重门控（搬自LuckyHDR但语义相反）
#    - 简化多频段融合：保留高低频分离，移除无理论依据的高频增强卷积
# ============================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        c = in_channels // 2
        self.in_channels = in_channels

        # 特征预处理
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.PReLU()
        )

        # 多尺度上下文建模
        self.ctx1 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.PReLU()
        )

        self.ctx2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.PReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=True),
            nn.PReLU(),
        )

        # 注意力权重估计 (3通道输出)
        self.fusion_f = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(c, 3, 3, padding=1, bias=True),
            nn.PReLU(),
        )

        # 特征精炼
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
            nn.PReLU(),
        )

        self.weight_min = 1e-4
        self.mb_kernel = 5  # 多频段融合核大小

    def _multiband_fusion(self, f0, f1, w0, w1):
        """多频段融合
        低频用注意力权重融合（整体结构/色调）
        高频同样用注意力权重加权（不对高频做额外卷积增强，避免引入伪影）
        """
        kernel_size = self.mb_kernel
        f0_low = box_blur(f0, kernel_size)
        f1_low = box_blur(f1, kernel_size)
        f0_high = f0 - f0_low
        f1_high = f1 - f1_low

        # 低频融合（使用注意力权重）
        fused_low = w0 * f0_low + w1 * f1_low

        # 高频融合（使用注意力权重，直接加权）
        fused_high = w0 * f0_high + w1 * f1_high

        # 组合（低频决定整体，高频补充细节）
        fused = fused_low + fused_high

        return fused

    def forward(self, f0, f1):
        # 1. 特征预处理
        x_0 = self.shared(f0)
        x_1 = self.shared(f1)

        # 2. 多尺度上下文建模
        x_0_1 = self.ctx1(x_0)
        x_0_2 = self.ctx2(x_0_1)
        x_0_2 = F.interpolate(x_0_2, size=(x_0_1.size(-2), x_0_1.size(-1)), mode='bilinear', align_corners=False)
        x_0_m = torch.cat([x_0_1, x_0_2], dim=1)
        x_0_f = self.fusion(x_0_m)

        x_1_1 = self.ctx1(x_1)
        x_1_2 = self.ctx2(x_1_1)
        x_1_2 = F.interpolate(x_1_2, size=(x_1_1.size(-2), x_1_1.size(-1)), mode='bilinear', align_corners=False)
        x_1_m = torch.cat([x_1_1, x_1_2], dim=1)
        x_1_f = self.fusion(x_1_m)

        # 3. 注意力权重估计
        x_m = torch.cat([x_0_f, x_1_f], dim=1)
        att_raw = self.fusion_f(x_m)
        att_w = F.softmax(att_raw, dim=1)

        w0 = att_w[:, 0:1]
        w1 = att_w[:, 1:2]

        # 4. 权重平滑化（空间平滑，消除高频孤立权重）
        w0 = box_blur(w0, 3)
        w1 = box_blur(w1, 3)
        denom = (w0 + w1).clamp_min(self.weight_min)
        w0 = w0 / denom
        w1 = w1 / denom

        # 5. 多频段融合（高低频分别加权）
        x_fused = self._multiband_fusion(f0, f1, w0, w1)

        # 6. 输出精炼
        att_w = torch.cat([w0, w1], dim=1)
        return self.extractor(x_fused), att_w


# ============================================================
# 4. Multi-Scale Feature Interaction
# ============================================================
class MultiScaleInteraction(nn.Module):
    def __init__(self, base_ch):
        super().__init__()
        self.scale_proj1 = nn.Conv2d(base_ch * 4, base_ch * 2, 1, bias=False)
        self.scale_proj2 = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, base_ch * 2)
        self.norm2 = nn.GroupNorm(8, base_ch)
        self.act = nn.PReLU()

    def forward(self, feats):
        f1, f2, f3 = feats
        
        f3_down = F.interpolate(self.scale_proj1(f3), size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f2_enhanced = self.act(self.norm1(f2 + f3_down))
        
        f2_down = F.interpolate(self.scale_proj2(f2_enhanced), size=f1.shape[-2:], mode='bilinear', align_corners=False)
        f1_enhanced = self.act(self.norm2(f1 + f2_down))
        
        return [f1_enhanced, f2_enhanced, f3]


# ============================================================
# 5. Decoder (Multi-scale Fusion)
# ============================================================
class Decoder(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.out = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_ch // 2),
            nn.PReLU(),
            nn.Conv2d(base_ch // 2, base_ch // 2, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(base_ch // 2, 3, 3, padding=1, bias=False),
            nn.PReLU()
        )

    def forward(self, feats):
        f1, f2, f3 = feats

        f2_x = self.up1(f3)
        f2_x = F.interpolate(f2_x, size=(f2.size(-2), f2.size(-1)), mode='bilinear', align_corners=False)
        f1_x = self.up2(f2_x + f2)
        f1_x = F.interpolate(f1_x, size=(f1.size(-2), f1.size(-1)), mode='bilinear', align_corners=False)
        x = f1_x + f1

        return self.out(x)


# ============================================================
# 5. Full Network
# ============================================================
class AHDR(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        self.encoder = MultiScaleEncoder(in_ch, base_ch)
        self.input_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

        self.local_attn = nn.ModuleList([
            LocalWindowAttention(base_ch, window_size=8),
            LocalWindowAttention(base_ch * 2, window_size=4),
            LocalWindowAttention(base_ch * 4, window_size=2),
        ])

        self.cross_attn = nn.ModuleList([
            CrossAttentionFusion(base_ch),
            CrossAttentionFusion(base_ch * 2),
            CrossAttentionFusion(base_ch * 4),
        ])

        self.scale_interact = MultiScaleInteraction(base_ch)

        self.decoder = Decoder(base_ch)

    def forward(self, x0, x1=None, x2=None):
        if x1 is None:
            x1 = torch.clamp(x0 + 0.25 * torch.tanh(self.input_branch(x0)), -1.0, 1.0)

        f0_scales = self.encoder(x0)
        f1_scales = self.encoder(x1)

        fused_feats = []
        attn = []
        for i in range(3):
            f0 = self.local_attn[i](f0_scales[i])
            f1 = self.local_attn[i](f1_scales[i])
            fused, w_ = self.cross_attn[i](f0, f1)
            w_ = F.interpolate(w_, size=(f0_scales[0].size(-2), f0_scales[0].size(-1)), mode='bicubic')
            attn.append(w_)
            fused_feats.append(fused)

        out = self.decoder(fused_feats)
        if x2 is None:
            out = torch.tanh(out)
            restored = torch.clamp(x0 + out, -1.0, 1.0)
            restored = (restored + 1.0) * 0.5
            if self.training:
                return restored, {'attn': attn, 'residual': out, 'fused_feats': fused_feats}
            return restored

        if not self.training:
            out = torch.clamp(out, 0.0, 1.0)
            return out

        out_dict = {'attn': attn}
        return out, out_dict