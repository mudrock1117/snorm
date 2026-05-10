""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
import torch.nn.functional as F

'''
class GatedScore(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        
        self.up1 = (Up(512, 256, bilinear))
        self.up2 = (Up(256, 128, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits.sigmoid()
'''

'''
class GatedScore(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
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
'''

#resize 1/4.0
class GatedScore(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, 256))
        self.down1 = (Down(256, 128))
        self.down2 = (Down(128, 64))
        #self.down3 = (Down(256, 512))

        self.outc = (OutConv(64, n_classes))

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        logits = self.outc(x3)
        #return logits.sigmoid()
        return logits


class GatedScore_S8(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore_S8, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, 256))
        self.down1 = (Down(256, 128))
        self.down2 = (Down(128, 64))
        self.down3 = (Down(64, 64))

        self.outc = (OutConv(64, n_classes))

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        logits = self.outc(x4)
        return logits



#s16
class GatedScore_S16(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore_S16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 128))
        self.down4 = (Down(128, 64))

        self.outc = (OutConv(64, n_classes))

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        logits = self.outc(x5)
        return logits

######################################################
class GatedScore_SV4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore_S16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 128))
        self.down3 = (Down(128, 128))
        self.down4 = (Down(128, 256))

        self.up1 = (Up(256, 128, bilinear))
        self.up2 = (Up(128, 64, bilinear))

        self.outc = (OutConv(64, n_classes))

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)

        logits = self.outc(x5)
        return logits
    
class GatedScore_SV8(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(GatedScore_SV8, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        hidden_num = n_channels
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 128))
        self.down3 = (Down(128, 128))
        self.down4 = (Down(128, 256))

        self.up1 = (Up(256, 128, bilinear))
        self.outc = (OutConv(128, n_classes))

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, targets=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        logits = self.outc(x)
        return logits
