from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops
from .layer import GradientReversal

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss
from detection.modeling.contrastive.PixelContrast import PixelContrastive
from detection.modeling.contrastive.PixelContrast import PixelContrastive

class ContrastModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        if cfg.MODEL.CONTRAST.PIXEL_LEVEL:



    def forward(self, dis_features, target, domain='source'):
        self.getGraphInformation

        pos_feats, neg_feats = self.select_samples(features_v0, features_v1, proposals)

        pos_max = self.dist_metric(pos_feats, pos_feats)
        neg_max = self.dist_metric(pos_feats, neg_feats)

        pos_max = pos_max.sum(dim=1)
        neg_max = neg_max.sum(dim=1)

        total_max = pos_max + neg_max

        loss = -torch.log(pos_max.div(total_max) + 1e-5)

        return loss.mean()

    def select_samples(self, features_v0, features_v1, proposals):
        return

    def dist_metric(self, pos_graph, neg_graph):

        return
