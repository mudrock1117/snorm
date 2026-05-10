from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class InstanceDiscriminator(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        
        out_channels = 64
        self.out_channels = out_channels
        
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        ##source domain
        self.gl_src_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, self.num_classes)
        )
        
        self.hard_src_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, self.num_classes)
        )
        '''
        self.sr_src_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, num_classes)
        )
        '''
        
        ##target domain
        self.gl_tgt_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, self.num_classes)
        )
        
        '''
        self.hard_tgt_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, num_classes)
        )
        '''
        
        

    def forward(self, feats, labels=None, probs=None, domain='s', sVec=None, mFeats=None, adv=0):
        #print(feats.size())
        ns, dm =  feats.size()
        if probs is not None:
            #print(probs.size())
            probs = F.softmax(probs, dim=1)
            probs = probs.detach()
        
        if labels is None:
            #print(probs.size())
            _, labels = probs.max(dim=1)
        else:
           labels = labels.detach()
        
        loss_dis = 0.0
        if domain == 's':
            loss_dis = self.source_forward(feats, labels=labels, probs=probs, sVec=sVec, mFeats=mFeats, adv=adv)
        elif domain == 't':
            loss_dis = self.target_forward(feats, labels=labels, probs=probs, sVec=sVec, mFeats=mFeats, adv=adv)
            
        return loss_dis
    
    def source_forward(self, feats, labels=None, probs=None, sVec=None, mFeats=None, adv=0):
        bn, pb = feats.size()
        mask = labels >= pb
        #mask = (labels >= pb) | (labels == 0)
        vlabel = labels.mul(1-mask.long()).unsqueeze(1)
        label_vt = torch.zeros_like(probs).scatter_(1, vlabel, 1)
        pixel_weights = 1 - mask.float()
        label_vt = label_vt.mul(pixel_weights.unsqueeze(1))
        
        _, probsi = probs.max(dim=1)
        cos_mask = labels.eq(probsi)
        
        gl_feats = feats[cos_mask, :]
        hd_feats = feats[~cos_mask, :]
        
        gl_loss = 0.0
        if gl_feats.size(0) > 0 and len(gl_feats.size()) > 1:
             label_lvt = label_vt[cos_mask,:]
             if adv == 0:
                 label_gl =  torch.cat((label_lvt, torch.zeros_like(label_lvt), torch.zeros_like(label_lvt)), dim=1)
                 gl_loss = self.forward_d(gl_feats, label_gl)
             elif adv == 1:
                 gl_loss = self.forward_adv(gl_feats, label_lvt, mark=0)
        
        hd_loss = 0.0
        if hd_feats.size(0) > 0 and len(hd_feats.size()) > 1:
             label_lvt = label_vt[~cos_mask,:]
             if adv == 0:
                 label_gl = torch.cat((torch.zeros_like(label_lvt),label_lvt, torch.zeros_like(label_lvt)), dim=1)
                 hd_loss = self.forward_d(hd_feats, label_gl)
             elif adv == 1:
                 hd_loss = self.forward_adv(hd_feats, label_lvt, mark=1)
        
        loss = gl_loss + hd_loss
        return loss
    
    def target_forward(self, feats, labels=None, probs=None, sVec=None, mFeats=None, adv=0):
        bn, pb = feats.size()
        #mask = (labels >= pb) | (labels == 0)
        mask = labels >= pb
        vlabel = labels.mul(1-mask.long()).unsqueeze(1)
        label_vt = torch.zeros_like(probs).scatter_(1, vlabel, 1)
        pixel_weights = 1 - mask.float()
        label_vt = label_vt.mul(pixel_weights.unsqueeze(1))
        
        _, probsi = probs.max(dim=1)
        cos_mask = labels.eq(probsi)
        
        gl_feats = feats[cos_mask, :]
        hd_feats = feats[~cos_mask, :]
        #print(feats.size(), cos_mask.size(), gl_feats.size(), hd_feats.size())
        
        gl_loss = 0.0
        if gl_feats.size(0) > 0 and len(gl_feats.size()) > 1:
             label_lvt = label_vt[cos_mask,:]
             if adv == 0:
                 label_gl = torch.cat((torch.zeros_like(label_lvt), torch.zeros_like(label_lvt), label_lvt), dim=1)
                 gl_loss = self.forward_d(gl_feats, label_gl)
             elif adv == 1:
                 gl_loss = self.forward_adv(gl_feats, label_lvt, mark=2)
        
        return gl_loss
    
    def forward_d(self, feats, labels=None, domain=0):
        gl_src_probs = self.gl_src_classifier(feats)
        hd_src_probs = self.hard_src_classifier(feats)
        gl_tgt_probs = self.gl_tgt_classifier(feats)
        
        #print(gl_src_probs.size(), hd_src_probs.size(), gl_tgt_probs.size())
        cm_probs = torch.cat((gl_src_probs, hd_src_probs, gl_tgt_probs), dim=1)
        cm_probs = F.softmax(cm_probs, dim=1)
        #loss_d = self.loss_func_d(cm_feats, labels)
        loss_d = -labels * torch.log(cm_probs+1e-5)
        loss_d = torch.mean(loss_d.sum(dim=1))
        return loss_d
    
    def forward_adv(self, feats, labels=None, domain=0, mark=0):
        bn, pb = labels.size()
        gl_src_probs = self.gl_src_classifier(feats)
        hd_src_probs = self.hard_src_classifier(feats)
        gl_tgt_probs = self.gl_tgt_classifier(feats)
        
        labels[:,0] *= 0
        
        cm_probs = torch.cat((gl_src_probs, hd_src_probs, gl_tgt_probs), dim=1)
        cm_probs = torch.exp(cm_probs)
        cm_labels = torch.cat((labels, labels, labels), dim=1)
        
        ##compute relation
        loss_0 = 0.0
        if mark != 0:
            cm_labels_0 = torch.cat((labels * 0, labels, labels), dim=1)
            cm_labels_0 = 1 - cm_labels_0
            cm_probs_0 = cm_probs.mul(cm_labels_0)
            cm_probs_0m = cm_probs_0.sum(dim=1, keepdim=True)
            cm_probs_0 = cm_probs_0.div(cm_probs_0m + 1e-6)
            #print(cm_probs_0.size(),labels.size())
            loss_0 = -labels * torch.log(cm_probs_0[:, 0:pb]+1e-5)
            loss_0 = torch.mean(loss_0.sum(dim=1))
        
        
        loss_1 = 0.0
        if mark != 1:
            cm_labels_1 = torch.cat((labels, labels*0.0, labels), dim=1)
            cm_labels_1 = 1 - cm_labels_1
            cm_probs_1 = cm_probs.mul(cm_labels_1)
            cm_probs_1m = cm_probs_1.sum(dim=1, keepdim=True)
            cm_probs_1 = cm_probs_1.div(cm_probs_1m + 1e-6)
            loss_1 = -labels * torch.log(cm_probs_1[:, pb:2*pb]+1e-5)
            loss_1 = torch.mean(loss_1.sum(dim=1))
        
        loss_2 = 0.0
        if mark != 2:
            cm_labels_2 = torch.cat((labels, labels, labels*0.0), dim=1)
            cm_labels_2 = 1 - cm_labels_2
            cm_probs_2 = cm_probs.mul(cm_labels_2)
            cm_probs_2m = cm_probs_2.sum(dim=1, keepdim=True)
            cm_probs_2 = cm_probs_2.div(cm_probs_2m + 1e-6)
            loss_2 = -labels * torch.log(cm_probs_2[:, 2*pb:3*pb]+1e-5)
            loss_2 = torch.mean(loss_2.sum(dim=1))
        
        loss_adv = loss_0 + loss_1 + loss_2
        return loss_adv
    
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
             
    
