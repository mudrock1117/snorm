from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from .layer import GradientReversal
from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class InstanceDiscriminator(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        
        out_channels = 64
        self.out_channels = out_channels
        
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        ##source domain
        self.source_style_domain_num = 2
        '''
        for idx in range(self.source_style_domain_num):
            self.add_module('src_classifier'+str(idx),
                    nn.Sequential(
                        nn.Dropout(),
                        nn.Linear(in_channels, in_channels),
                        nn.ReLU(True),
                        nn.Linear(in_channels, self.num_classes)
                    )
                )
        '''
        
        ##target domain
        self.target_style_domain_num = 2
        '''
        for idx in range(self.target_style_domain_num):
             self.add_module('tgt_classifier'+str(idx),
                 nn.Sequential(
                     nn.Dropout(),
                     nn.Linear(in_channels, in_channels),
                     nn.ReLU(True),
                     nn.Linear(in_channels, self.num_classes)
                 )
             )
        '''
        for idx in range(4):
            self.add_module('classifier'+str(idx),
                nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_channels, in_channels),
                    nn.ReLU(True),
                    nn.Linear(in_channels, self.num_classes)
                )
            )
        
        self.grad_reverse = GradientReversal(0.01)
        
        

    def forward(self, feats, labels=None, probs=None, style_domain=None, domain='s', sVec=None, mFeats=None, adv=0):
        ns, dm =  feats.size()
        if probs is not None:
            probs = F.softmax(probs, dim=1)
            probs = probs.detach()
        
        if labels is None:
            _, labels = probs.max(dim=1)
        else:
            labels = labels.detach()
        
        bn, pb = feats.size()
        mask = labels >= pb
        #mask = (labels >= pb) | (labels == 0)
        vlabel = labels.mul(1-mask.long()).unsqueeze(1)
        label_vt = torch.zeros_like(probs).scatter_(1, vlabel, 1)
        pixel_weights = 1 - mask.float()
        label_vt = label_vt.mul(pixel_weights.unsqueeze(1))
        
        _, probsi = probs.max(dim=1)
        
        loss = 0.0
        total_style_domain_num = self.source_style_domain_num + self.target_style_domain_num
        for idx in range(total_style_domain_num):
            sm_feats = None
            #source
            if idx == 0 and style_domain == 0:
                sm_mask = labels.eq(labels)
                sm_feats = feats[sm_mask, :]
            elif idx == 1 and style_domain == 1:
                sm_mask = labels.eq(labels)
                sm_feats = feats[sm_mask, :]
            #target
            elif idx == 2 and style_domain == 2:
                sm_mask = labels.eq(labels)
                sm_feats = feats[sm_mask, :]
            elif idx == 3 and style_domain == 3:
                sm_mask = labels.eq(labels)
                sm_feats = feats[sm_mask, :]
            
            if sm_feats is not None:
                loss += self.compute_forward(sm_feats, label_vt, sm_mask, adv=adv, mark=idx)
    
        return loss * 100
    
    def compute_forward(self, feats, label_vt, cos_mask, adv=0, mark=0):
        loss = 0.0
        if feats.size(0) > 0 and len(feats.size()) > 1:
            label_lvt = label_vt[cos_mask,:]
            if mark == 0:
                loss_adv0 = self.forward_adv(feats, label_vt, domain_labels=0, mark=0)
                loss_adv1 = self.forward_adv(feats, label_vt, domain_labels=0, mark=1)
            elif mark == 1:
                loss_adv0 = self.forward_adv(feats, label_vt, domain_labels=1, mark=0)
                loss_adv1 = self.forward_adv(feats, label_vt, domain_labels=0, mark=2)
            elif mark == 2:
                loss_adv0 = self.forward_adv(feats, label_vt, domain_labels=0, mark=3)
                loss_adv1 = self.forward_adv(feats, label_vt, domain_labels=1, mark=1)
            elif mark == 3:
                loss_adv0 = self.forward_adv(feats, label_vt, domain_labels=1, mark=3)
                loss_adv1 = self.forward_adv(feats, label_vt, domain_labels=1, mark=2)
        
        loss = loss_adv0 + loss_adv1
        return loss
    
    def forward_adv(self, feats, labels=None, domain_labels=0, mark=0):
        bn, pb = labels.size()
        
        labels[:, 0] *= 0.0
        feats = self.grad_reverse(feats)
        
        classifier = getattr(self, 'classifier'+str(mark))
        probs = classifier(feats)
        probs = F.softmax(probs, dim=1)
        loss_log = (1 - domain_labels) * torch.log(probs+1e-5) + domain_labels * torch.log(1 - probs + 1e-5)
        
        loss = -labels * loss_log
        return loss.mean()
    
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
             
    
