import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import ssim


# =========================================================
# 1. 前景加权工具
# =========================================================
def apply_foreground_weight(loss_map, mask, fg_weight=2.0):
    """
    loss_map: (B, C, H, W) or (B, H, W)
    mask:     (B, 1, H, W)
    """
    if loss_map.dim() == 4:
        weight = 1.0 + (fg_weight - 1.0) * mask
    else:
        weight = 1.0 + (fg_weight - 1.0) * mask.squeeze(1)
    return (loss_map * weight).mean()


# =========================================================
# 2. 重建损失（L1）
# =========================================================
class WeightedL1Loss(nn.Module):
    def __init__(self, fg_weight=2.0):
        super().__init__()
        self.fg_weight = fg_weight

    def forward(self, x1, x0, mask):
        loss_map = torch.abs(x1 - x0)
        return apply_foreground_weight(loss_map, mask, self.fg_weight)


# =========================================================
# 3. SSIM 损失（前景加权）
# =========================================================
class WeightedSSIMLoss(nn.Module):
    def __init__(self, fg_weight=2.0):
        super().__init__()
        self.fg_weight = fg_weight

    def forward(self, x1, x0, mask):
        ssim_map = ssim(x1, x0, data_range=1.0, size_average=False)
        loss_map = 1.0 - ssim_map
        return apply_foreground_weight(loss_map, mask, self.fg_weight)


# =========================================================
# 4. 感知损失（VGG）
# =========================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, fg_weight=2.0):
        super().__init__()
        self.fg_weight = fg_weight

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:16],  # relu3_3
        ])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x1, x0, mask):
        loss = 0.0
        for block in self.blocks:
            x1 = block(x1)
            x0 = block(x0)
            m = F.interpolate(mask, size=x1.shape[-2:], mode="nearest")
            loss += apply_foreground_weight(torch.abs(x1 - x0), m, self.fg_weight)
        return loss


# =========================================================
# 5. 判别器 D(x)
# =========================================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            block(in_channels, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 6. 对抗损失（前景加权）
# =========================================================
class WeightedGANLoss(nn.Module):
    def __init__(self, fg_weight=2.0):
        super().__init__()
        self.fg_weight = fg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, mask):
        """
        pred: D(x1)
        target: 1 for G loss, 0/1 for D loss
        """
        target_tensor = torch.full_like(pred, target)
        loss_map = self.bce(pred, target_tensor)
        m = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
        return apply_foreground_weight(loss_map, m, self.fg_weight)


# =========================================================
# 7. 总损失封装
# =========================================================
class ImageGenerationLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = WeightedL1Loss(fg_weight=3.0)
        self.ssim = WeightedSSIMLoss(fg_weight=2.0)
        #self.perc = VGGPerceptualLoss(fg_weight=2.0)
        self.gan = WeightedGANLoss(fg_weight=3.0)

        self.lambda_l1 = 1.0
        self.lambda_ssim = 0.5
        self.lambda_perc = 0.1
        self.lambda_gan = 0.01

    def forward(self, x1, x0, mask):
        # 判别器输出
        #d_fake = D(x1)

        loss = (
            self.lambda_l1 * self.l1(x1, x0, mask) +
            self.lambda_ssim * self.ssim(x1, x0, mask) +
            #self.lambda_perc * self.perc(x1, x0, mask) +
            self.lambda_gan * self.gan(x1, 1.0, mask)
        )
        return loss
        
def nll_loss_manual(x, y, reduction="mean", eps=1e-8):
    """
    x: [B, C, H, W]  softmax 后的概率
    y: [B, 1, H, W]  类别索引 (0 ~ C-1)
    """
    B, C, H, W = x.shape

    # [B, H, W]
    y = y.squeeze(1).long()

    # 1. 取 log
    log_x = torch.log(x.clamp(min=eps))  # 防止 log(0)

    # 2. 根据 GT 取对应类别概率
    # [B, 1, H, W]
    log_p = torch.gather(
        log_x, 
        dim=1, 
        index=y.unsqueeze(1)
    )

    # 3. NLL
    loss = -log_p.squeeze(1)  # [B, H, W]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
