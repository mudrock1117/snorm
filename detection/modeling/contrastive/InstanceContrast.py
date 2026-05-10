from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class InstanceContrastive(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        
        out_channels = 64
        self.out_channels = out_channels
        '''
        self.project_heads = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
        '''
        
        self.project_heads2 = nn.Sequential(
            nn.AdaptiveMaxPool1d(64)
        )
        
        
        
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        ###source information
        # create the queue
        self.num_general_samples = 30
        self.register_buffer("sfeature_general_memory", torch.randn(self.num_classes, self.num_general_samples, self.out_channels))
        
        self.num_hard_samples = 30
        self.register_buffer("sfeature_hard_memory", torch.randn(self.num_classes, self.num_hard_samples, self.out_channels))
        
        self.num_similar_samples = 30
        self.register_buffer("sfeature_similar_memory", torch.randn(self.num_classes, self.num_similar_samples, self.out_channels))
        #self.queue = nn.functional.normalize(self.queue, dim=0)
        
        self.num_style_samples = 30
        self.register_buffer("sfeature_style_memory", torch.randn(self.num_classes, self.num_style_samples, self.out_channels))
        
        self.register_buffer("smemory_ptr", torch.zeros((self.num_classes, 4), dtype=torch.long))
        self.register_buffer("smemory_index", torch.zeros((self.num_classes, 4), dtype=torch.long))
        
        ###target information
        self.register_buffer("tfeature_general_memory", torch.randn(self.num_classes, self.num_general_samples, self.out_channels))
        self.register_buffer("tfeature_hard_memory", torch.randn(self.num_classes, self.num_hard_samples, self.out_channels))
        self.register_buffer("tfeature_similar_memory", torch.randn(self.num_classes, self.num_similar_samples, self.out_channels))
        self.register_buffer("tfeature_style_memory", torch.randn(self.num_classes, self.num_style_samples, self.out_channels))
        self.register_buffer("tmemory_ptr", torch.zeros((self.num_classes, 4), dtype=torch.long))
        self.register_buffer("tmemory_index", torch.zeros((self.num_classes, 4), dtype=torch.long))

    def forward(self, feats, labels=None, probs=None, domain='s', sVec=None, mFeats=None):
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
        '''
        else:
            print(probs.size(1), labels.max())
        '''
        
        feats = self.project_heads2(feats)
        if mFeats is not None:
           mFeats = self.project_heads2(mFeats)
        
        loss = 0.0
        for idx in range(self.num_classes):
            '''
            mark = labels == idx
            cfeats = feats[mark, :] 
            clabels = labels[mark]
            cprobs = probs[mark,:]
            
            _, probsi = probs.max(dim=1)
            prob_mark = probsi == idx
            similar_mask = (~mark) & prob_mark
            sfeats = feats[similar_mask, :]  
            #cu, cs, cv = torch.svd(feats_cls)
            '''
            
            if domain == 's':
                loss += self.source_forward(feats, idx, labels=labels, probs=probs, sVec=sVec, mFeats=mFeats) / self.num_classes
            elif domain == 't':
                loss += self.target_forward(feats, idx, labels=labels, probs=probs, sVec=sVec, mFeats=mFeats) / self.num_classes
            
        return loss.mean() * 0.1
    
    def source_forward(self, feats, idx, labels=None, probs=None, sVec=None, mFeats=None):
        device = feats.device
        
        mark = labels == idx
        cfeats = feats[mark, :] 
        clabels = labels[mark]
        cprobs = probs[mark,:]
        #print(mark[0], (~mark)[0])
        
        _, probsi = probs.max(dim=1)
        prob_mark = probsi == idx
        similar_mask = (~mark) & prob_mark
        if mFeats is not None:
            sfeats = mFeats[similar_mask, :]
        else:
            sfeats = feats[similar_mask, :]
        
        if sVec is None:
            csVec = torch.zeros_like(labels)
        else:
            csVec = sVec[mark]
                    
        run_condition = False
        if self.smemory_index[:,0].min() ==1 and self.smemory_index[:,1].min() ==1 and \
        self.smemory_index[:,2].min() ==1 and idx > 0:
             run_condition = True
        
        if cfeats.size(0) <= 0:
             run_condition = False        
        
        if run_condition:
            #select general pos samples
            dist_gl_pos_in = self.jaccard_distance(cfeats, self.sfeature_general_memory[idx].clone().detach())
            
            #select hard pos samples
            dist_hd_pos_in = self.jaccard_distance(cfeats, self.sfeature_hard_memory[idx].clone().detach())
            
            #select style pos samples
            dist_sl_pos_in = self.jaccard_distance(cfeats, self.sfeature_style_memory[idx].clone().detach())
            
            #select pos samples from another domain
            dist_gl_pos_et = self.jaccard_distance(cfeats, self.tfeature_general_memory[idx].clone().detach())
            
            #select general neg samples
            g_neg_feats_mark = torch.rand((self.num_classes, self.num_general_samples)).to(device)
            g_neg_feats_markv, g_neg_feats_marki = g_neg_feats_mark.topk(10, dim=1)
            g_neg_feats_marki = g_neg_feats_marki.unsqueeze(2).expand(g_neg_feats_marki.size(0), g_neg_feats_marki.size(1), self.out_channels)
            #print(self.sfeature_general_memory.size(), g_neg_feats_marki.size())
            g_neg_feats = torch.gather(self.sfeature_general_memory.clone().detach(), 1, g_neg_feats_marki)
            g_neg_feats_mark = torch.ones(self.num_classes, 10).to(device)
            g_neg_feats_mark[idx] *= 0.0
            g_neg_feats_mask = g_neg_feats_mark > 0.5
            g_neg_nfeats = g_neg_feats[g_neg_feats_mask, :]
            
            #print(g_neg_nfeats.size())
            
            dist_gl_neg_in = self.jaccard_distance(cfeats, g_neg_nfeats)
            
            loss_gl_gl_in = self.loss_similary(dist_gl_pos_in, dist_gl_neg_in)
            loss_gl_hd_in = 0.0#self.loss_similary(dist_hd_pos_in, dist_gl_neg_in)
            loss_gl_sl_in = 0.0##self.loss_similary(dist_sl_pos_in, dist_gl_neg_in)
            loss_gl_et_tr = 0.0#self.loss_similary(dist_gl_pos_et, dist_gl_neg_in)
            
            
            #select similar neg samples
            dist_sr_neg_in = self.jaccard_distance(cfeats, self.sfeature_similar_memory[idx].clone().detach())
            
            loss_sr_gl_in = self.loss_similary(dist_gl_pos_in, dist_sr_neg_in)
            loss_sr_hd_in = self.loss_similary(dist_hd_pos_in, dist_sr_neg_in)
            loss_sr_sl_in = 0.0#self.loss_similary(dist_sl_pos_in, dist_sr_neg_in)
            loss_sr_et_tr = self.loss_similary(dist_gl_pos_et, dist_sr_neg_in)
            
            
            #select neg samples from another domain
            e_neg_feats_mark = torch.rand((self.num_classes, self.num_general_samples)).to(device)
            e_neg_feats_markv, e_neg_feats_marki = e_neg_feats_mark.topk(10, dim=1)
            e_neg_feats_marki = e_neg_feats_marki.unsqueeze(2).expand(e_neg_feats_marki.size(0), e_neg_feats_marki.size(1), self.out_channels)
            e_neg_feats = torch.gather(self.tfeature_general_memory.clone().detach(), 1, e_neg_feats_marki)
            e_neg_feats_mark = torch.ones(self.num_classes, 10).to(device)
            e_neg_feats_mark[idx] *= 0.0
            e_neg_feats_mask = e_neg_feats_mark>0.5
            e_neg_nfeats = e_neg_feats[e_neg_feats_mask, :]
            
            dist_et_neg_in = self.jaccard_distance(cfeats, e_neg_nfeats)
            
            loss_et_gl_in = self.loss_similary(dist_gl_pos_in, dist_et_neg_in)
            loss_et_hd_in = 0.0#self.loss_similary(dist_hd_pos_in, dist_et_neg_in)
            loss_et_sl_in = 0.0##self.loss_similary(dist_sl_pos_in, dist_et_neg_in)
            loss_et_et_tr = 0.0#self.loss_similary(dist_gl_pos_et, dist_et_neg_in)
            
            
            ###compute loss
            loss_gl = loss_gl_gl_in + loss_gl_hd_in + loss_gl_sl_in + loss_gl_et_tr
            loss_sr = loss_sr_gl_in + loss_sr_hd_in + loss_sr_sl_in + loss_sr_et_tr
            loss_et = loss_et_gl_in + loss_et_hd_in + loss_et_sl_in + loss_et_et_tr
            
            loss = loss_gl + loss_sr + loss_et
            #print('---source:', loss)
        else:
            loss = 0.0 * feats.mean()
        
        ###update memory
        if cfeats.size(0) > 0:
            '''
            prob_v, prob_i = cprobs.max(dim=1)
            label_mask = clabels == idx
            prob_mask = prob_i == idx
            sVec_mask = csVec > 0.5
            
            gerenal_mask = label_mask & prob_mask & (~sVec_mask)
            hard_mask = label_mask & (~prob_mask) & (~sVec_mask)
            similar_mask = (~label_mask) & prob_mask & (~sVec_mask)
            style_mask = label_mask & sVec_mask
            
            self.update_memory(cfeats[gerenal_mask, :], idx, tp='gl', domain='s')
            self.update_memory(cfeats[hard_mask, :], idx, tp='hd', domain='s')
            self.update_memory(cfeats[style_mask, :], idx, tp='sl', domain='s')
            '''
            prob_v, prob_i = cprobs.max(dim=1)
            label_mask = clabels == idx
            prob_mask = prob_i == idx
            sVec_mask = csVec > 0.5
            
            gerenal_mask = mark & prob_mark & (~sVec_mask)
            hard_mask = mark & (~prob_mark) & (~sVec_mask)
            similar_mask = (~mark) & prob_mark & (~sVec_mask)
            style_mask = mark & sVec_mask
            
            if mFeats is not None:
                self.update_memory(mFeats[gerenal_mask, :], idx, tp='gl', domain='s')
                self.update_memory(mFeats[hard_mask, :], idx, tp='hd', domain='s')
                self.update_memory(mFeats[style_mask, :], idx, tp='sl', domain='s')
            else:
                self.update_memory(feats[gerenal_mask, :], idx, tp='gl', domain='s')
                self.update_memory(feats[hard_mask, :], idx, tp='hd', domain='s')
                self.update_memory(feats[style_mask, :], idx, tp='sl', domain='s')
            
            
        self.update_memory(sfeats, idx, tp='sr', domain='s')
        
        return loss
    
    '''
    def target_forward(self, feats, idx, labels=None, probs=None, sVec=None):
        device = feats.device
        
        mark = labels == idx
        cfeats = feats[mark, :] 
        clabels = labels[mark]
        cprobs = probs[mark,:]
        
        _, probsi = probs.max(dim=1)
        prob_mark = probsi == idx
        similar_mask = (~mark) & prob_mark
        sfeats = feats[similar_mask, :]
        
        if sVec is None:
            csVec = torch.zeros_like(clabels)
        else:
            csVec = sVec[mark]
        
        run_condition = False
        if self.tmemory_index[:,0].min() == 1 and self.tmemory_index[:,1].min() ==1 and \
        self.tmemory_index[:,2].min() ==1:
             run_condition = True
        
        if cfeats.size(0) <= 0:
             run_condition = False
             
        #print('----target', run_condition)
        if run_condition:
            #select general pos samples
            dist_gl_pos_in = self.jaccard_distance(cfeats, self.tfeature_general_memory[idx].clone().detach())
            
            #select hard pos samples
            dist_hd_pos_in = self.jaccard_distance(cfeats, self.tfeature_hard_memory[idx].clone().detach())
            
            #select style pos samples
            dist_sl_pos_in = self.jaccard_distance(cfeats, self.tfeature_style_memory[idx].clone().detach())
            
            #select pos samples from another domain
            dist_gl_pos_et = self.jaccard_distance(cfeats, self.tfeature_general_memory[idx].clone().detach())
            
            #select general neg samples
            g_neg_feats_mark = torch.rand((self.num_classes, self.num_general_samples)).to(device)
            g_neg_feats_markv, g_neg_feats_marki = g_neg_feats_mark.topk(10, dim=1)
            g_neg_feats_marki = g_neg_feats_marki.unsqueeze(2).expand(g_neg_feats_marki.size(0), g_neg_feats_marki.size(1), self.out_channels)
            g_neg_feats = torch.gather(self.tfeature_general_memory.clone().detach(), 1, g_neg_feats_marki)
            g_neg_feats_mark = torch.ones(self.num_classes, 10).to(device)
            g_neg_feats_mark[idx] = 0.0
            g_neg_feats_mask = g_neg_feats_mark > 0.5
            g_neg_nfeats = g_neg_feats[g_neg_feats_mask, :]
            
            dist_gl_neg_in = self.jaccard_distance(cfeats, g_neg_nfeats)
            
            loss_gl_gl_in = self.loss_similary(dist_gl_pos_in, dist_gl_neg_in)
            loss_gl_hd_in = self.loss_similary(dist_hd_pos_in, dist_gl_neg_in)
            loss_gl_sl_in = 0.0#self.loss_similary(dist_sl_pos_in, dist_gl_neg_in)
            loss_gl_et_tr = self.loss_similary(dist_gl_pos_et, dist_gl_neg_in)
            
            
            #select similar neg samples
            dist_sr_neg_in = self.jaccard_distance(cfeats, self.tfeature_style_memory[idx].clone().detach())
            
            loss_sr_gl_in = self.loss_similary(dist_gl_pos_in, dist_sr_neg_in)
            loss_sr_hd_in = self.loss_similary(dist_hd_pos_in, dist_sr_neg_in)
            loss_sr_sl_in = 0.0#self.loss_similary(dist_sl_pos_in, dist_sr_neg_in)
            loss_sr_et_tr = self.loss_similary(dist_gl_pos_et, dist_sr_neg_in)
            
            
            #select neg samples from another domain
            e_neg_feats_mark = torch.rand((self.num_classes, self.num_general_samples)).to(device)
            e_neg_feats_markv, e_neg_feats_marki = e_neg_feats_mark.topk(10, dim=1)
            e_neg_feats_marki = e_neg_feats_marki.unsqueeze(2).expand(e_neg_feats_marki.size(0), e_neg_feats_marki.size(1), self.out_channels)
            e_neg_feats = torch.gather(self.sfeature_general_memory.clone().detach(), 1, e_neg_feats_marki)
            e_neg_feats_mark = torch.ones(self.num_classes, 10).to(device)
            e_neg_feats_mark[idx] = 0.0
            e_neg_feats_mask = e_neg_feats_mark>0.5
            e_neg_nfeats = e_neg_feats[e_neg_feats_mask, :]
            
            dist_et_neg_in = self.jaccard_distance(cfeats, e_neg_nfeats)
            
            loss_et_gl_in = self.loss_similary(dist_gl_pos_in, dist_et_neg_in)
            loss_et_hd_in = self.loss_similary(dist_hd_pos_in, dist_et_neg_in)
            loss_et_sl_in = 0.0#self.loss_similary(dist_sl_pos_in, dist_et_neg_in)
            loss_et_et_tr = self.loss_similary(dist_gl_pos_et, dist_et_neg_in)
            
            ###compute loss
            loss_gl = loss_gl_gl_in + loss_gl_hd_in + loss_gl_sl_in + loss_gl_et_tr
            loss_sr = loss_sr_gl_in + loss_sr_hd_in + loss_sr_sl_in + loss_sr_et_tr
            loss_et = loss_et_gl_in + loss_et_hd_in + loss_et_sl_in + loss_et_et_tr
            
            loss = loss_gl + loss_sr + loss_et
            print('---target:', loss)
        else:
            loss = feats.mean() * 0.0
        
        ###update memory
        if cfeats.size(0) > 0:
            prob_v, prob_i = cprobs.max(dim=1)
            label_mask = clabels == idx
            prob_mask = prob_i == idx
            sVec_mask = csVec > 0.5
            
            gerenal_mask = label_mask & prob_mask & (~sVec_mask)
            hard_mask = label_mask & (~prob_mask) & (~sVec_mask)
            similar_mask = (~label_mask) & prob_mask & (~sVec_mask)
            style_mask = label_mask & sVec_mask
            
            self.update_memory(cfeats[gerenal_mask, :], idx, tp='gl', domain='t')
            self.update_memory(cfeats[hard_mask, :], idx, tp='hd', domain='t')
            self.update_memory(cfeats[style_mask, :], idx, tp='sl', domain='t')
        self.update_memory(sfeats, idx, tp='sr', domain='t')
        
        return loss
    '''
    
    def target_forward(self, feats, idx, labels=None, probs=None, sVec=None, mFeats=None):
        device = feats.device
        
        mark = labels == idx
        cfeats = feats[mark, :] 
        clabels = labels[mark]
        cprobs = probs[mark,:]
        
        _, probsi = probs.max(dim=1)
        prob_mark = probsi == idx
        similar_mask = (~mark) & prob_mark
        if mFeats is not None:
            sfeats = mFeats[similar_mask, :]
        else:
            sfeats = feats[similar_mask, :]
        
        if sVec is None:
            csVec = torch.zeros_like(labels)
        else:
            csVec = sVec[mark]
        
        run_condition = False
        if self.tmemory_index[:,0].min() == 1 and idx>0:
             run_condition = True
        
        if cfeats.size(0) <= 0:
             run_condition = False
             
        if run_condition:
            #select general pos samples
            dist_gl_pos_in = self.jaccard_distance(cfeats, self.tfeature_general_memory[idx].clone().detach())
            
            #select hard pos samples
            dist_hd_pos_in = self.jaccard_distance(cfeats, self.tfeature_hard_memory[idx].clone().detach())
            
            #select style pos samples
            dist_sl_pos_in = self.jaccard_distance(cfeats, self.tfeature_style_memory[idx].clone().detach())
            
            #select pos samples from another domain
            dist_gl_pos_et = self.jaccard_distance(cfeats, self.tfeature_general_memory[idx].clone().detach())
            
            #select general neg samples
            g_neg_feats_mark = torch.rand((self.num_classes, self.num_general_samples)).to(device)
            g_neg_feats_markv, g_neg_feats_marki = g_neg_feats_mark.topk(10, dim=1)
            g_neg_feats_marki = g_neg_feats_marki.unsqueeze(2).expand(g_neg_feats_marki.size(0), g_neg_feats_marki.size(1), self.out_channels)
            g_neg_feats = torch.gather(self.tfeature_general_memory.clone().detach(), 1, g_neg_feats_marki)
            g_neg_feats_mark = torch.ones(self.num_classes, 10).to(device)
            g_neg_feats_mark[idx] *= 0.0
            g_neg_feats_mask = g_neg_feats_mark > 0.5
            g_neg_nfeats = g_neg_feats[g_neg_feats_mask, :]
            
            dist_gl_neg_in = self.jaccard_distance(cfeats, g_neg_nfeats)
            
            loss_gl_gl_in = self.loss_similary(dist_gl_pos_in, dist_gl_neg_in)
            loss_gl_hd_in = 0.0#self.loss_similary(dist_hd_pos_in, dist_gl_neg_in)
            loss_gl_sl_in = 0.0#self.loss_similary(dist_sl_pos_in, dist_gl_neg_in)
            loss_gl_et_tr = 0.0#self.loss_similary(dist_gl_pos_et, dist_gl_neg_in)
            
            
            #select similar neg samples
            dist_sr_neg_in = 0.0#self.jaccard_distance(cfeats, self.tfeature_similar_memory[idx].clone().detach())
            
            loss_sr_gl_in = 0.0#self.loss_similary(dist_gl_pos_in, dist_sr_neg_in)
            loss_sr_hd_in = 0.0#self.loss_similary(dist_hd_pos_in, dist_sr_neg_in)
            loss_sr_sl_in = 0.0#self.loss_similary(dist_sl_pos_in, dist_sr_neg_in)
            loss_sr_et_tr = 0.0#self.loss_similary(dist_gl_pos_et, dist_sr_neg_in)
            
            
            #select neg samples from another domain
            e_neg_feats_mark = torch.rand((self.num_classes, self.num_general_samples)).to(device)
            e_neg_feats_markv, e_neg_feats_marki = e_neg_feats_mark.topk(10, dim=1)
            e_neg_feats_marki = e_neg_feats_marki.unsqueeze(2).expand(e_neg_feats_marki.size(0), e_neg_feats_marki.size(1), self.out_channels)
            e_neg_feats = torch.gather(self.sfeature_general_memory.clone().detach(), 1, e_neg_feats_marki)
            e_neg_feats_mark = torch.ones(self.num_classes, 10).to(device)
            e_neg_feats_mark[idx] *= 0.0
            e_neg_feats_mask = e_neg_feats_mark>0.5
            e_neg_nfeats = e_neg_feats[e_neg_feats_mask, :]
            
            dist_et_neg_in = self.jaccard_distance(cfeats, e_neg_nfeats)
            
            loss_et_gl_in = self.loss_similary(dist_gl_pos_in, dist_et_neg_in)
            loss_et_hd_in = 0.0#self.loss_similary(dist_hd_pos_in, dist_et_neg_in)
            loss_et_sl_in = 0.0#self.loss_similary(dist_sl_pos_in, dist_et_neg_in)
            loss_et_et_tr = 0.0#self.loss_similary(dist_gl_pos_et, dist_et_neg_in)
            
            ###compute loss
            loss_gl = loss_gl_gl_in + loss_gl_hd_in + loss_gl_sl_in + loss_gl_et_tr
            loss_sr = loss_sr_gl_in + loss_sr_hd_in + loss_sr_sl_in + loss_sr_et_tr
            loss_et = loss_et_gl_in + loss_et_hd_in + loss_et_sl_in + loss_et_et_tr
            
            loss = loss_gl + loss_sr + loss_et
            #print('---target:', loss)
        else:
            loss = feats.mean() * 0.0
        
        ###update memory
        if cfeats.size(0) > 0:
            '''
            prob_v, prob_i = cprobs.max(dim=1)
            label_mask = clabels == idx
            prob_mask = prob_i == idx
            '''
            sVec_mask = csVec > 0.5
            
            gerenal_mask = mark & prob_mark & (~sVec_mask)
            hard_mask = mark & (~prob_mark) & (~sVec_mask)
            similar_mask = (~mark) & prob_mark & (~sVec_mask)
            style_mask = mark & sVec_mask
            
            if mFeats is not None:
                self.update_memory(mFeats[gerenal_mask, :], idx, tp='gl', domain='t')
                self.update_memory(mFeats[hard_mask, :], idx, tp='hd', domain='t')
                self.update_memory(mFeats[style_mask, :], idx, tp='sl', domain='t')
            else:
                self.update_memory(feats[gerenal_mask, :], idx, tp='gl', domain='t')
                self.update_memory(feats[hard_mask, :], idx, tp='hd', domain='t')
                self.update_memory(feats[style_mask, :], idx, tp='sl', domain='t')
                
        self.update_memory(sfeats, idx, tp='sr', domain='t')
        
        return loss * 0.05
    
    @torch.no_grad()
    def update_memory(self, cfeats, idx, tp='gl', domain='s'):
        if cfeats.size(0) <= 0:
           return
        #print('--', domain, idx, tp, cfeats.size(0))
           
        if domain == 's':
             if tp == 'gl':
                 ptr = int(self.smemory_ptr[idx, 0])
                 g_width = cfeats.size(0) if cfeats.size(0) < (self.num_general_samples-ptr) else self.num_general_samples-ptr
                 self.sfeature_general_memory[idx, ptr:ptr+g_width] = cfeats[0:g_width]
                 self.smemory_ptr[idx, 0] = (ptr + g_width) % self.num_general_samples
                 if cfeats.size(0) >= self.num_general_samples-ptr and int(self.smemory_index[idx, 0]) == 0:
                     self.smemory_index[idx, 0] = 1
             elif tp == 'hd':
                 ptr = int(self.smemory_ptr[idx, 1])
                 h_width = cfeats.size(0) if cfeats.size(0) < (self.num_hard_samples-ptr) else (self.num_hard_samples-ptr)
                 self.sfeature_hard_memory[idx, ptr:ptr+h_width] = cfeats[0:h_width]
                 self.smemory_ptr[idx, 1] = (ptr + h_width) % self.num_hard_samples
                 #print(self.sfeature_hard_memory-ptr, int(self.smemory_index[idx, 1]))
                 if cfeats.size(0) >= self.num_hard_samples-ptr and int(self.smemory_index[idx, 1]) == 0:
                     self.smemory_index[idx, 1] = 1
             elif tp == 'sr':
                 ptr = int(self.smemory_ptr[idx, 2])
                 sr_width = cfeats.size(0) if cfeats.size(0) < (self.num_similar_samples-ptr) else (self.num_similar_samples-ptr)
                 self.sfeature_similar_memory[idx, ptr:ptr+sr_width] = cfeats[0:sr_width]
                 self.smemory_ptr[idx, 2] = (ptr + sr_width) % self.num_similar_samples
                 if cfeats.size(0) >= self.num_similar_samples-ptr and int(self.smemory_index[idx, 2]) == 0:
                     self.smemory_index[idx, 2] = 1
             elif tp == 'sl':
                 ptr = int(self.smemory_ptr[idx, 3])
                 sl_width = cfeats.size(0) if cfeats.size(0) < (self.num_style_samples-ptr) else (self.num_style_samples-ptr)
                 self.sfeature_style_memory[idx, ptr:ptr+sl_width] = cfeats[0:sl_width]
                 self.smemory_ptr[idx, 3] = (ptr + sl_width) % self.num_style_samples
                 if cfeats.size(0) >= self.num_style_samples-ptr and int(self.smemory_index[idx, 3]) == 0:
                     self.smemory_index[idx, 3] = 1
        elif domain == 't':
             if tp == 'gl':
                 ptr = int(self.tmemory_ptr[idx, 0])
                 g_width = cfeats.size(0) if cfeats.size(0) < (self.num_general_samples-ptr) else (self.num_general_samples-ptr)
                 self.tfeature_general_memory[idx, ptr:ptr+g_width] = cfeats[0:g_width]
                 self.tmemory_ptr[idx, 0] = (ptr + g_width) % self.num_general_samples
                 if cfeats.size(0) >= self.num_general_samples-ptr and int(self.tmemory_index[idx, 0]) == 0:
                     self.tmemory_index[idx, 0] = 1
             elif tp == 'hd':
                 ptr = int(self.tmemory_ptr[idx, 1])
                 h_width = cfeats.size(0) if cfeats.size(0) < (self.num_hard_samples-ptr) else (self.num_hard_samples-ptr)
                 self.tfeature_hard_memory[idx, ptr:ptr+h_width] = cfeats[0:h_width]
                 self.tmemory_ptr[idx, 1] = (ptr + h_width) % self.num_hard_samples
                 if cfeats.size(0) >= self.num_hard_samples-ptr and int(self.tmemory_index[idx, 1]) == 0:
                     self.tmemory_index[idx, 1] = 1
             elif tp == 'sr':
                 ptr = int(self.tmemory_ptr[idx, 2])
                 sr_width = cfeats.size(0) if cfeats.size(0) < (self.num_similar_samples-ptr) else (self.num_similar_samples-ptr)
                 self.tfeature_similar_memory[idx, ptr:ptr+sr_width] = cfeats[0:sr_width]
                 self.tmemory_ptr[idx, 2] = (ptr + sr_width) % self.num_similar_samples
                 if cfeats.size(0) >= self.num_similar_samples-ptr and int(self.tmemory_index[idx, 2]) == 0:
                     self.tmemory_index[idx, 2] = 1
             elif tp == 'sl':
                 ptr = int(self.tmemory_ptr[idx, 3])
                 sl_width = cfeats.size(0) if cfeats.size(0) < (self.num_style_samples-ptr) else (self.num_style_samples-ptr)
                 self.tfeature_style_memory[idx, ptr:ptr+sl_width] = cfeats[0:sl_width]
                 self.tmemory_ptr[idx, 3] = (ptr + sl_width) % self.num_style_samples
                 if cfeats.size(0) >= self.num_style_samples-ptr and int(self.tmemory_index[idx, 3]) == 0:
                     self.tmemory_index[idx, 3] = 1
             
        return
        
    def jaccard_distance(self, feat_a, feat_b):
        na, adm = feat_a.size()
        nb, bdm = feat_b.size()
        
        '''
        feat_a = feat_a.unsqueeze(1).expand(na, nb, adm)
        feat_b = feat_b.unsqueeze(0).expand(na, nb, adm)
        
        s_interdection = feat_a.mul(feat_b)
        s_interdection = s_interdection.sum(dim=2)
        s_uion = feat_a.mul(feat_a) + feat_b.mul(feat_b)
        s_uion = s_uion.sum(dim=2)
        dist = s_interdection.div(s_uion + 1e-5)
        '''
        dist = torch.einsum('nd, md->nm', feat_a, feat_b)
        
        return dist
        
    def loss_similary(self, pos_sm_mt, neg_sm_mt):
        #pos sum
        pos_sm_mt = pos_sm_mt.mean(dim=1, keepdim=True)
        
        dist = torch.cat((pos_sm_mt, neg_sm_mt), dim=1)
        dist= F.softmax(dist, dim=1)
        loss = -torch.log(dist[:, 0:pos_sm_mt.size(1)].sum(dim=1) + 1e-5)
        return loss.mean()
