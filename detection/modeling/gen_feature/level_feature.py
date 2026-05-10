import math
import torch
import torch.nn.functional as F
from torch import nn

#simple global feature
class getTwoStageFeature(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(getTwoStageFeature, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.zero_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, \
            stride=1, bias=False)
        )
        

        self.conv_two = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
        
        self.conv_three = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        
        
    def forward(self, xs):
        #zero stage
        znx = self.zero_conv(xs)
        
        #one stage
        onx_s = self.conv_two(xs)
        onx = F.interpolate(onx_s, size=[xs.size(-2), xs.size(-1)], mode="bilinear")
        
        tnx_s = self.conv_three(xs)
        tnx = F.interpolate(tnx_s, size=[xs.size(-2), xs.size(-1)], mode="bilinear")
            
        return znx, onx, tnx



class getGlobal(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(getGlobal, self).__init__()
        # TODO: Implement the sigmoid version first.
        levels = len(cfg.MODEL.GENFEATURE.FPN_STRIDES)
        self.fpn_strides = cfg.MODEL.GENFEATURE.FPN_STRIDES
        
        self.msf = getTwoStageFeature(cfg, in_channels)
        
        self.in_channels = in_channels
        
        out_channels_new = in_channels // 5
        out_channels_old = in_channels - out_channels_new
        
        self.upsample_new = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    out_channels_new,
                    kernel_size=1,
                    stride=1
                )
        )
        
        self.upsample_old = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    out_channels_old,
                    kernel_size=1,
                    stride=1
                )
        )
        
        
    def forward(self, x):
        #two stage
        znx, onx, tnx = self.msf(x)
        nx = torch.cat((znx, onx, tnx), 1)
        return nx
    
    def merge_feature(self, features, proposals):
        nb, nc, nh, nw = features.size()
        features = features.view(nb, nc, -1)
        
        #print(features.size())
        #zero
        lv = 0
        hv = self.in_channels
        zero_ft = features[:, lv:hv, :]
        
        #one
        lv = self.in_channels
        hv = self.in_channels * 2
        one_ft = features[:, lv:hv, :]
        
        #two
        lv = self.in_channels * 2
        hv = self.in_channels * 3
        two_ft = features[:, lv:hv, :]
        
        gl_ft = zero_ft + one_ft + two_ft
        gl_ft = gl_ft.view(nb, -1, nh, nw)
        return gl_ft
        
    def fusion(self, ofeatures, nfeatures):
        lofeatures = self.upsample_old(ofeatures)
        lnfeatures = self.upsample_new(nfeatures)
        features = torch.cat((lofeatures, lnfeatures), dim=1)
        return features
        

def build_featuregen(cfg, in_channels):
    return getGlobal(cfg, in_channels)
