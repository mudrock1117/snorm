import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== Basic Blocks =====================

class DoubleConv(nn.Module):
    """(Conv => PReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with AvgPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            #nn.AvgPool2d(2),
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(channels, channels)

    def forward(self, x):
        return self.conv(self.up(x))


# ===================== Pixel-wise Validity Attention =====================

class PixelValidityAttention(nn.Module):
    """
    Predict pixel-wise validity weights for two HDR inputs.
    Output: (B, 2, H, W), softmax normalized
    """
    def __init__(self, in_channels):
        super().__init__()
        c = in_channels // 2

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.PReLU()
        )

        self.ctx1 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.PReLU()
        )

        self.ctx2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.PReLU()
        )

        self.ctx3 = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.PReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(c * 3, c, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(c, 2, 1, bias=True)
        )

    def forward(self, x):
        x = self.shared(x)
        f1 = self.ctx1(x)
        f2 = F.interpolate(self.ctx2(x), size=f1.shape[-2:], mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.ctx3(x), size=f1.shape[-2:], mode='bilinear', align_corners=False)

        fusion = torch.cat([f1, f2, f3], dim=1)
        weights = self.fusion(fusion)
        return F.softmax(weights, dim=1)


# ===================== AHDR Network =====================

class AHDR(nn.Module):
    """
    Pixel-wise Validity Guided HDR Network
    """
    def __init__(self, nChannel=3, nFeat=64):
        super().__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(nChannel, nFeat, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Encoder
        self.down0 = Down(nFeat, nFeat)
        self.down1 = Down(nFeat, nFeat)
        self.down2 = Down(nFeat, nFeat)
        self.down3 = Down(nFeat, nFeat)

        self.down0_b = Down(nFeat, nFeat)
        self.down1_b = Down(nFeat, nFeat)
        self.down2_b = Down(nFeat, nFeat)
        self.down3_b = Down(nFeat, nFeat)

        # Pixel-wise attention
        self.attn1 = PixelValidityAttention(nFeat * 2)
        self.attn2 = PixelValidityAttention(nFeat * 2)
        self.attn3 = PixelValidityAttention(nFeat * 2)
        self.attn4 = PixelValidityAttention(nFeat * 2)

        self.fuse1 = DoubleConv(nFeat, nFeat)
        self.fuse2 = DoubleConv(nFeat, nFeat)
        self.fuse3 = DoubleConv(nFeat, nFeat)
        self.fuse4 = DoubleConv(nFeat, nFeat)

        # Decoder
        self.up4 = Up(nFeat)
        self.up3 = Up(nFeat)
        self.up2 = Up(nFeat)
        self.up1 = Up(nFeat)

        # Output
        self.out = nn.Sequential(
            nn.Conv2d(nFeat, nFeat // 2, 3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(nFeat // 2, 3, 3, padding=1, bias=False)
        )

    def fuse(self, f0, f1, attn):
        w = attn(torch.cat([f0, f1], dim=1))
        return w[:, 0:1] * f0 + w[:, 1:2] * f1, w

    def forward(self, x0, x1, x2=None):
        # Shallow features
        f0 = self.conv_in(x0)
        f1 = self.conv_in(x1)

        # Encoder
        f0_1 = self.down0(f0)
        f0_2 = self.down1(f0_1)
        f0_3 = self.down2(f0_2)
        f0_4 = self.down3(f0_3)

        f1_1 = self.down0_b(f1)
        f1_2 = self.down1_b(f1_1)
        f1_3 = self.down2_b(f1_2)
        f1_4 = self.down3_b(f1_3)

        # Pixel-wise fusion
        fm_1, wm_1 = self.fuse(f0_1, f1_1, self.attn1)
        fm_2, wm_2 = self.fuse(f0_2, f1_2, self.attn2)
        fm_3, wm_3 = self.fuse(f0_3, f1_3, self.attn3)
        fm_4, wm_4 = self.fuse(f0_4, f1_4, self.attn4)
        
        Fx_1 = self.fuse1(fm_1)
        Fx_2 = self.fuse2(fm_2)
        Fx_3 = self.fuse3(fm_3)
        Fx_4 = self.fuse4(fm_4)

        # Decoder
        Fu_3 = self.up4(Fx_4)
        Fu_3 = F.interpolate(Fu_3, size=(Fx_3.size(-2), Fx_3.size(-1)), mode='bicubic')
        Fc_3 =  Fu_3 + Fx_3
        
        Fu_2 = self.up3(Fc_3)
        Fu_2 = F.interpolate(Fu_2, size=(Fx_2.size(-2), Fx_2.size(-1)), mode='bicubic')
        Fc_2 = Fu_2 + Fx_2
        
        
        Fu_1 = self.up2(Fc_2)
        Fu_1 = F.interpolate(Fu_1, size=(Fx_1.size(-2), Fx_1.size(-1)), mode='bicubic')
        Fc_1 = Fu_1 + Fx_1
        d0 = self.up1(Fc_1)

        out = self.out(d0)
        out = F.interpolate(out, size=x0.shape[-2:], mode='bilinear', align_corners=False)

        if not self.training:
            out = torch.clamp(out, 0.0, 1.0)
            return out
        
        wm_2 = F.interpolate(wm_2, size=(wm_1.size(-2), wm_1.size(-1)), mode='bicubic')
        wm_3 = F.interpolate(wm_3, size=(wm_1.size(-2), wm_1.size(-1)), mode='bicubic')
        wm_4 = F.interpolate(wm_4, size=(wm_1.size(-2), wm_1.size(-1)), mode='bicubic')
        out_dict = {'attn': [wm_1, wm_2, wm_3, wm_4]}
        return out, out_dict
