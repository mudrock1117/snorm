from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops
from .layer import GradientReversal

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class InstanceContrastive(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()



    def forward(self, features_v0, features_v1, proposals):
        #select pos and neg sample based computing the similarity of the whole object

        sim_max = torch.enusin('bnd, bmd->bnm', features_v0, features_v1)
        pos_max, neg_max = self.select_instances(sim_max)

        pos_max = pos_max.sum(dim=1)
        neg_max = neg_max.sum(dim=1)

        total_max = pos_max + neg_max

        loss = -torch.log(pos_max.div(total_max) + 1e-5)

        return loss.mean()

    def select_instances(self, sim_max):
        return
