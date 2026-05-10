""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
import torch.nn.functional as F



class GatedScore(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, n_channels))
        self.down1 = (Down(n_channels, n_channels))
        self.down2 = (Down(n_channels, n_channels))
        self.down3 = (Down(n_channels, n_channels))
        factor = 2 if bilinear else 1
        
        self.up1 = (Up(512, 256, bilinear))
        self.up2 = (Up(256, 128, bilinear))
        self.outc = (OutConv(128, n_classes))

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        logits = self.outc(x)
        return logits.sigmoid()



