""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
import torch.nn.functional as F


class tconV(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(tconV, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (cDoubleConv(n_channels, n_channels))
        
        self.down1 = (cDown(n_channels, hidden_num))
        self.down2 = (cDown(hidden_num, hidden_num))
        #self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        
        self.up1 = (cUp(hidden_num, hidden_num, bilinear))
        self.up2 = (cUp(hidden_num, hidden_num, bilinear))
        #self.up3 = (Up(128, 64, bilinear))
        self.outc = (cOutConv(n_channels, n_channels))

        #self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        #print(x.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        #x = self.up3(x, x1)
        logits = self.outc(x)
        
        return logits
