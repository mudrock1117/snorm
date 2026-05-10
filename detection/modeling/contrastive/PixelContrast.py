from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops
from .layer import GradientReversal

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class PixelContrastive(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        self.loss_f = nn.MSELoss(reduce=True, size_average=True)

    def forward(self, features_v0, features_v1, proposals):
        #print(dis_features.size())
        feat_v0_pts, feat_v1_pts = self.select_points(features_v0, features_v1, proposals)
        loss_pixel = self.loss_consis(feat_v0_pts, feat_v1_pts)
        return loss_pixel

    def select_points(self, features_v0, features_v1, proposals):

        return

    def loss_consis(self, feat_v0_pts, feat_v1_pts):
        loss = self.loss_f(feat_v0_pts, feat_v1_pts)
        return loss.mean()
