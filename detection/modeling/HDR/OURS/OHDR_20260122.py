import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
            nn.PReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, in_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class attnC(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.share = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True)
            nn.PReLU()
            #nn.Sigmoid()
        )
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True)
            nn.PReLU()
            #nn.Sigmoid()
        )
        
        self.layer_2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True)
            nn.PReLU()
            #nn.Sigmoid()
        )
        
        self.layer_3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            #nn.Sigmoid(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True)
            nn.PReLU()
            #nn.Sigmoid()
        )
        
        self.layer_out = nn.Sequential(
            nn.Conv2d(in_channels * 3 // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            #nn.Sigmoid(),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True)
            nn.PReLU()
            #nn.Sigmoid()
        )

    def forward(self, x1):
        x_s = self.share(x1)
        
        x_l1 = self.layer_1(x_s)
        x_l2 = self.layer_2(x_s)
        x_l3 = self.layer_3(x_s)
        
        x_l2 = F.interpolate(x_l2, size=(x_l1.size(-2), x_l1.size(-1)), mode='bicubic')
        x_l3 = F.interpolate(x_l3, size=(x_l1.size(-2), x_l1.size(-1)), mode='bicubic')
        
        x_cat = torch.cat([x_l1, x_l2, x_l3], dim=1)
        x_cat = self.layer_out(x_cat)
        x_fin = nn.functional.sigmoid(x_cat)
        return x_fin


        
class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(DRDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dilation_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

# Attention Guided HDR, AHDR-Net
class AHDR(nn.Module):
    def __init__(self, nChannel, nDenselayer, nFeat, growthRate):
        super(AHDR, self).__init__()
        nChannel = nChannel #args.nChannel
        nDenselayer = nDenselayer #args.nDenselayer
        nFeat = nFeat #args.nFeat
        growthRate = growthRate #args.growthRate
        #self.args = args
        
        '''
        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        #self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)

        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv 
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        '''
        
        
        
        ########################################
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        
        self.down0_1 = (Down(nFeat, nFeat))
        self.down0_2 = (Down(nFeat, nFeat))
        self.down0_3 = (Down(nFeat, nFeat))
        self.down0_4 = (Down(nFeat, nFeat))
        
        self.down1_1 = (Down(nFeat, nFeat))
        self.down1_2 = (Down(nFeat, nFeat))
        self.down1_3 = (Down(nFeat, nFeat))
        self.down1_4 = (Down(nFeat, nFeat))
        
        #fusion
        
        self.up4 = Up(nFeat, nFeat, bilinear=True)
        self.up3 = Up(nFeat, nFeat, bilinear=True)
        self.up2 = Up(nFeat, nFeat, bilinear=True)
        self.up1 = Up(nFeat, nFeat, bilinear=True)
        
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(nFeat),
            #nn.ReLU(inplace=True)
            nn.PReLU()
        )
        
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(nFeat),
            #nn.ReLU(inplace=True)
            nn.PReLU()
        )
        
        self.fusion_3 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(nFeat),
            #nn.ReLU(inplace=True)
            nn.PReLU()
        )
        
        
        self.fusion_4 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(nFeat),
            #nn.ReLU(inplace=True)
            nn.PReLU()
        )
        
        #self.att21 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        #self.att22 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        
        #self.att41 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        #self.att42 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        
        self.out_layer = nn.PReLU() #nn.ReLU()
        #self.out_layer = nn.functional.sigmoid(output)
        
        
        self.attn_1 = attnC(nFeat*2)
        self.attn_2 = attnC(nFeat*2)
        self.attn_3 = attnC(nFeat*2)
        self.attn_4 = attnC(nFeat*2)
        
        self.out_f = nn.Sequential(
            nn.Conv2d(nFeat, nFeat // 2, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Conv2d(nFeat // 2, 3, kernel_size=3, padding=1, bias=False),
            #nn.ReLU(inplace=True)
        )


    def forward(self, x0, x1, x2):
        F0_ = self.relu(self.conv1(x0))
        F1_ = self.relu(self.conv1(x1))
        
        ############encoder############
        ####
        F0_1 = self.down0_1(F0_)
        F0_2 = self.down0_2(F0_1)
        F0_3 = self.down0_3(F0_2)
        F0_4 = self.down0_4(F0_3)
        #print(F0_.size(), F0_1.size())
        
        
        F1_1 = self.down1_1(F1_)
        F1_2 = self.down1_2(F1_1)
        F1_3 = self.down1_3(F1_2)
        F1_4 = self.down1_4(F1_3)
        
        
        
        ##s1##
        Fm_1 = torch.cat((F0_1, F1_1), 1)
        F1_A = self.attn_1(Fm_1)
        Fx_1 = F0_1 * F1_A + (1 - F1_A) * F1_1
        Fx_1 = self.fusion_1(Fx_1)
        
        
        ##s2##
        Fm_2 = torch.cat((F0_2, F1_2), 1)
        F2_A = self.attn_2(Fm_2)
        Fx_2 = F0_2 * F2_A + (1 - F2_A) * F1_2
        Fx_2 = self.fusion_2(Fx_2)
        
        ##s3##
        Fm_3 = torch.cat((F0_3, F1_3), 1)
        F3_A = self.attn_3(Fm_3)
        Fx_3 = F0_3 * F3_A + (1 - F3_A) * F1_3
        Fx_3 = self.fusion_3(Fx_3)
        
        
        ##s4##
        Fm_4 = torch.cat((F0_4, F1_4), 1)
        F4_A = self.attn_4(Fm_4)
        Fx_4 = F0_4 * F4_A + (1 - F4_A) * F1_4
        Fx_4 = self.fusion_4(Fx_4)
        
        
        ##
        #print(self.up4(Fx_4).size(), Fx_3.size())
        Fu_3 = self.up4(Fx_4)
        Fu_3 = F.interpolate(Fu_3, size=(Fx_3.size(-2), Fx_3.size(-1)), mode='bicubic')
        Fc_3 =  Fu_3 + Fx_3
        
        Fu_2 = self.up3(Fc_3)
        Fu_2 = F.interpolate(Fu_2, size=(Fx_2.size(-2), Fx_2.size(-1)), mode='bicubic')
        Fc_2 = Fu_2 + Fx_2
        
        
        Fu_1 = self.up2(Fc_2)
        Fu_1 = F.interpolate(Fu_1, size=(Fx_1.size(-2), Fx_1.size(-1)), mode='bicubic')
        Fc_1 = Fu_1 + Fx_1
        
        Fu_0 = self.up1(Fc_1)
        
        
        output = self.out_f(Fu_0)
        #output = nn.functional.sigmoid(output)
        output = self.out_layer(output)
        
        output = F.interpolate(output, size=(x0.size(-2), x0.size(-1)), mode='bicubic')
        #print(x0.size(), output.size())
        if not self.training:
             output = output.clip(max=1.0, min=0.0)

        return output
