import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import random
#from kmeans_pytorch import kmeans

from torchvision import transforms

class SamplerAugmentation(nn.Module):
    def __init__(self, num_classes, im_szs, is_translate=False):
        super(SamplerAugmentation, self).__init__()
        
    
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        
        feat = feat.permute(1,0,2,3).contiguous()
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(1, C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(1, C, 1, 1)
        return feat_mean, feat_std

    #AadIN
    def adaptive_instance_normalization(self, content_feat, gamma=1.0, beta=0.01):
        size = content_feat.size()
        gamma = gamma.view(size[0],1,1,1)
        beta = beta.view(size[0],1,1,1)
        
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean) / content_std
        normalized_feat = normalized_feat * gamma + beta
        
        return normalized_feat
    
        
    def forward(self, images, target):
        B, C, H, W = images.size()
        
        #cross
        gamma = 1 - torch.rand((B)).to(images.device) * 0.5
        beta = torch.rand((B)).to(images.device) * 0.1
        images = self.adaptive_instance_normalization(images, gamma, beta)
        
        return images, target



