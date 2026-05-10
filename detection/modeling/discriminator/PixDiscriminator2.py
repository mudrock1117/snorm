from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class PixDiscriminator(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_channels        = cfg.MODEL.RPN.NUM_CHANNELS
        num_anchors         = len(anchor_scales) * len(anchor_ratios)
        
        ##source domain
        self.source_style_domain_num = 2
        for idx in range(self.source_style_domain_num):
            self.add_module('src_classifier'+str(idx),
                    nn.Sequential(
                        nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(num_channels, num_anchors, kernel_size=1, stride=1)
                    )
                )
        
        ##target domain
        self.target_style_domain_num = 2
        for idx in range(self.target_style_domain_num):
             self.add_module('tgt_classifier'+str(idx),
                 nn.Sequential(
                     nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1),
                     nn.Conv2d(num_channels, num_anchors, kernel_size=1, stride=1)
                 )
             )
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.mseloss = torch.nn.MSELoss(reduce=True)

    def forward(self, feats, labels=None, probs=None, style_domain=None, domain='s', sVec=None, mFeats=None, adv=0):
        n, dm, fh, fw = feats.size()
        _, pb, _, _ = probs
        
        if probs is not None:
            probs = probs.sigmoid()#F.softmax(probs, dim=1)
            probs = probs.detach()
        
        if labels is None:
            labels = probs
        else:
            labels = labels.detach()
        
        loss = 0.0
        total_style_domain_num = self.source_style_domain_num + self.target_style_domain_num
        for idx in range(total_style_domain_num):
            #source
            if idx == 0 and style_domain == 0:
                loss += self.compute_forward(feats, labels.detach(), None, adv=adv, mark=idx)
            elif idx == 1 and style_domain == 1:
                loss += self.compute_forward(feats, labels.detach(), None, adv=adv, mark=idx)
            #target
            elif idx == 2 and style_domain == 2:
                loss += self.compute_forward(feats, labels.detach(), None, adv=adv, mark=idx)
            elif idx == 3 and style_domain == 3:
                loss += self.compute_forward(feats, labels.detach(), None, adv=adv, mark=idx)
            
        return loss
    
    def compute_forward(self, feats, labels, cos_mask, adv=0, mark=0):
        loss = 0.0
        
        if adv == 0:
            loss = self.forward_d(feats, labels, mark=mark)
        elif adv == 1:
            loss = 1.0 * self.forward_adv(feats, labels, mark=mark)
            #testing 0.5
        
        return loss
    
    def forward_d(self, feats, labels=None, domain=0, mark=0):
        cm_probs = []
        total_iter = 0
        loss = 0.0
        for idx in range(self.source_style_domain_num):
             src_classifier = getattr(self, 'src_classifier'+str(idx))
             lprobs = src_classifier(feats)
             if total_iter == mark:
                 #loss += self.criterion(lprobs, labels)
                 loss += self.mseloss(lprobs.sigmoid(), labels)
             else:
                 loss += 0.0 * (lprobs.sigmoid()).mean()
             total_iter += 1
        
        for idx in range(self.target_style_domain_num):
            tgt_classifier = getattr(self, 'tgt_classifier'+str(idx))
            lprobs = tgt_classifier(feats)
            if total_iter == mark:
                loss += self.mseloss(lprobs.sigmoid(), labels)
            else:
                loss += 0.0 * (lprobs.sigmoid()).mean()
            total_iter += 1
        
        return loss
    
    def forward_adv(self, feats, labels=None, domain=0, mark=0):
        bn = labels.size()
        
        cm_probs = []
        total_iter = 0
        #criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        labels[labels==0] = 255
        
        loss = 0.0
        for idx in range(self.source_style_domain_num):
             src_classifier = getattr(self, 'src_classifier'+str(idx))
             lprobs = src_classifier(feats)
             loss += self.mseloss(lprobs.sigmoid(), labels)
             #cm_probs.append(lprobs)
             total_iter += 1
        
        for idx in range(self.target_style_domain_num):
            tgt_classifier = getattr(self, 'tgt_classifier'+str(idx))
            lprobs = tgt_classifier(feats)
            loss += self.mseloss(lprobs.sigmoid(), labels)
            #cm_probs.append(lprobs)
            total_iter += 1
        
        return loss / (self.source_style_domain_num + self.target_style_domain_num)
    
    def generate_labels_d(self, labels, mark=0):
        total_index = 0
        labels_ = []
        for idx in range(self.source_style_domain_num):
             if total_index == mark:
                  labels_.append(labels)
             else:
                  labels_.append(torch.zeros_like(labels))
             total_index += 1
        
        for idx in range(self.target_style_domain_num):
            if total_index == mark:
                 labels_.append(labels)
            else:
                 labels_.append(torch.zeros_like(labels))
                 
            total_index += 1
        
        labels_ = torch.cat(labels_, dim=1)
        return labels_
    
    def generate_labels_adv(self, labels, mark=0):
        total_index = 0
        labels_ = []
        for idx in range(self.source_style_domain_num):
             if total_index == mark:
                  labels_.append(labels*0.0)
             else:
                  labels_.append(labels)
             
             total_index += 1
        
        for idx in range(self.target_style_domain_num):
            if total_index == mark:
                 labels_.append(labels*0.0)
            else:
                 labels_.append(labels)
            
            total_index += 1
        
        labels_ = torch.cat(labels_, dim=1)
        return labels_
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                #trunc_normal_(m.weight, std=.02)
                m.bias.data.zero_()
             
    
