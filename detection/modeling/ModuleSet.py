import numpy as np
import random
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as T
import detection.utils.transforms.transforms as T
from detection.utils.layer import GradientReversal
from terminaltables import AsciiTable

from detection.layers import grad_reverse, softmax_focal_loss, sigmoid_focal_loss, style_pool2d, l2_loss
from .backbone import build_backbone
from .roi_heads import BoxHead
from .rpn import RPN

import cv2
import math
import copy

#from detection.modeling.contrastive.GraphContrast import GraphContrastive
from detection.modeling.discriminator.InDiscriminator import InstanceDiscriminator
#from gan.I2IPool_1115 import I2ITienhance
#from detection.modeling.norm.SNorm import SNorm
#from detection.modeling.norm.R202503X.SNorm_0322_god import SNorm
#from detection.modeling.norm.R202503X.SNorm_0322_valid import SNorm
#from detection.modeling.norm.R202503X.SNorm_0322_valid_refine import SNorm
from detection.modeling.norm.R202509x.SNorm_0322_valid_refine import SNorm
#from detection.modeling.norm.R202509x.SNorm_0322_valid_refine_expmamba import SNorm

#
from torch.utils.tensorboard import SummaryWriter
from detection.modeling.norm.script import generate_rect_boxes, select_rect_proposals, select_best_feature, \
generate_rect_boxes_v2, select_rect_proposals_v2, select_rect_weight_v2, select_best_weight, \
select_rect_weight_v3, select_rect_weight_v4, scoring_loss, select_rect_weight_v5, class_loss, \
select_rect_weight_v6, select_rect_weight_v7, select_rect_labels_v7

from detection.modeling.norm.dice_loss import DiceLoss


class ModuleSet(nn.Module):
    def __init__(self, cfg, positive_fraction=0.25, use_eval_head = False, eval_head_num=5, use_memory_conv=False):
        super(ModuleSet, self).__init__()
        self.cfg = cfg
        backbone = build_backbone(cfg)
        in_channels = backbone.out_channels

        #self.backbone = backbone
        
        # 处理 VGG16 backbone
        if cfg.MODEL.BACKBONE.NAME == 'vgg16':
            # VGG16 features 结构: [0-3] 第一个卷积块, [4-7] 第二个卷积块, [8-11] 第三个卷积块, [12-15] 第四个卷积块, [16-19] 第五个卷积块
            self.backbone_0 = backbone[:4]  # 第一个卷积块
            self.backbone_1 = backbone[4:8]  # 第二个卷积块
            self.backbone_2 = backbone[8:12]  # 第三个卷积块
            self.backbone_3 = backbone[12:16]  # 第四个卷积块
            self.backbone_3n = copy.deepcopy(backbone[12:16])
        else:
            # 处理 ResNet backbone
            self.backbone_0 = backbone.stem
            self.backbone_1 = backbone.layer1
            self.backbone_2 = backbone.layer2
            self.backbone_3 = backbone.layer3
            self.backbone_3n = copy.deepcopy(backbone.layer3)
        
        
        #self.rpn = RPN(cfg, in_channels)
        #self.box_head = BoxHead(cfg, in_channels)

        self.rpn_eval = RPN(cfg, in_channels)
        self.box_head_eval = BoxHead(cfg, in_channels)

        self.rpn_norm = RPN(cfg, in_channels)
        self.box_head_norm = BoxHead(cfg, in_channels)

        if cfg.MODEL.MODE.PJOTYPE == 'BN':
            pass
        elif cfg.MODEL.MODE.PJOTYPE == 'NBIAS':
            #base: 0000_1116.log
            #self.proj_norm = nn.BatchNorm2d(512)
            #base: 0000_111x.log
            #self.proj_norm = nn.InstanceNorm2d(512)

            #
            '''
            self.proj_norm = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.ReLU()
            )
            '''
            '''
            self.proj_norm = nn.Sequential(
                nn.InstanceNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU()
            )
            '''
            
            #self.proj_norm = nn.Sequential(
            #    SNorm(),
            #    #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            #    #nn.ReLU()
            #)
            self.proj_norm = SNorm()

            #self.proj_dbias = I2ITienhance(cfg)
        
        self.ada_layers = [False, True, True]
        self.ada_layers_stride = []
        
        self.llayers_rois = [[0,0.5], [0.2, 0.5], [0.2, 0.5]]
        self.ada_layers_rois = []

        self.positive_fraction = positive_fraction
    
    def prev_norm(self, x):
        
        return x
    
    def back_norm_wo_snorm(self, x):
        #x = self.backbone_2(x)

        x = self.backbone_0(x)
        x = self.backbone_1(x)
        x = self.backbone_2(x)
        x = self.backbone_3(x)
        return x
    
    def back_norm_w_snorm(self, x):
        x = self.backbone_0(x)
        x = self.backbone_1(x)
        x = self.backbone_2(x)
        x = self.backbone_3(x)

        #x = self.backbone_2n(x)
        return x
    

    def forward_resnet101(self, x):
        b,c, h, w = x.size()
        adaptation_feats = []
        adaptation_strides = []
        idx = 0
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
            

        idx += 1
        x = self.backbone.layer2(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        

        idx += 1
        x = self.backbone.layer3(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        
        self.ada_layers_stride = adaptation_strides

        return x, adaptation_feats
    
    def forward_resnet50(self, x):
        b,c, h, w = x.size()
        adaptation_feats = []
        adaptation_strides = []
        idx = 0
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
            

        idx += 1
        x = self.backbone.layer2(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        

        idx += 1
        x = self.backbone.layer3(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        
        self.ada_layers_stride = adaptation_strides

        return x, adaptation_feats
    
    def forward(self, images, img_metas, targets=None, proposals_=None, \
               is_use_no_grad=False, output_tcs=None, domain='source', use_dbiase=False, \
               results_parameters=None, mode=0, features_ff=None, use_moe=False, use_moe_fusion=False, \
               test_exm=0, moe_mode=0, update_norm_p=False, is_update=False, moe_mark=0):
        outputs = dict()
        #function define        
        
        ##rpn and rcnn
        if mode== 0: #score training/evaluation
            # 对于默认模式，使用正常的前向传播流程
            features_prev = self.prev_norm(images)
            # 通过 dist_prog 处理特征，传递 is_night 参数
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode, moe_mark=moe_mark)
            features_ff = self.back_norm_w_snorm(features_norm)
            features_ff = features_ff.detach()
            proposals, rpn_losses, rpn_logits, rpn_boxes = self.rpn_eval(images, features_ff, img_metas, targets)
            
            ##
            dets, box_losses, _, _, _, _, _, obj_dets, _ = \
            self.box_head_eval(features_ff, proposals, img_metas, targets, results_parameters=results_parameters, test_exm=test_exm)
            dets_bb = None
            features = None
            outputs['proposals'] = proposals
            outputs['rpn_losses_eval'] = rpn_losses
            outputs['box_losses_eval'] = box_losses
            outputs['dets_eval'] = dets
            outputs['dets_w'] = dets

        elif mode== 1: #detection training/evaluation
            features_prev = self.prev_norm(images)

            #without norm
            '''
            features_wo = self.back_norm_wo_snorm(features_prev)
            proposals_wo, rpn_losses_wo, _, _ = self.rpn(images, features_wo, img_metas, targets)
            dets, box_losses_wo, _, _, _, _, _, obj_dets, _ = self.box_head(features_wo, proposals_wo, img_metas, targets)
            '''

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            #if update_norm_p:
            #    self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
                
            _snorm_experiment = os.environ.get("SNORM_EXPERIMENT", "E0").upper()
            _snorm_mode = 8 if _snorm_experiment == "E3" else moe_mode
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=_snorm_mode, moe_mark=moe_mark)
            # 保存渲染结果用于评估可视化
            self._last_render_features = features_norm.detach()
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            '''
            if update_norm_p:
                loss_weights_0 = sum(list(rpn_losses_w.values()))
                loss_weights_1 = sum(list(box_losses_w.values()))
                loss_weights = loss_weights_0 + loss_weights_1
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update, loss_weights=loss_weights)
            '''

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
        
            
        
        elif mode== 2: #fusion training
            #with norm
            features_norm, feat_dict = self.dist_prog(features_ff.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)

            '''
            features_w = self.back_norm_w_snorm(features_norm)

            mproposals, rpn_losses_m, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            _, box_losses_m, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, mproposals, img_metas, targets)
            outputs['rpn_losses_m'] = rpn_losses_m
            outputs['box_losses_m'] = box_losses_m
            dets_bb = None
            outputs['features_norm'] = features_norm
            outputs['features_w'] = features_w
            '''
            
            
            rpn_losses_m = 0.0
            box_losses_m = 0.0
            
            for idx_ii in range(len(feat_dict['feature_dict_per'])):
                win_sz = feat_dict['win_sz_set'][idx_ii]
                features_norm_l = feat_dict['feature_dict_per'][idx_ii]

                nimages, ntargets, nimg_metas, nfeatures_norm_l = select_rect_labels_v7(images, targets, img_metas, \
                features_norm_l, win_sz_set=win_sz)

                if len(ntargets) == 0:
                    continue
                
                features_w_l = self.back_norm_w_snorm(nfeatures_norm_l)
                mproposals_l, rpn_losses_ml, _, _ = self.rpn_norm(nimages, features_w_l, nimg_metas, ntargets)
                _, box_losses_ml, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w_l, mproposals_l, \
                                                                nimg_metas, ntargets)
                rpn_losses_m += sum(rpn_losses_ml.values())
                box_losses_m += sum(box_losses_ml.values())
                #print(rpn_losses_ml)
            
            '''
            for idx_ii in range(len(feat_dict['feature_dict_fused'])):
                features_norm_l = feat_dict['feature_dict_fused'][idx_ii]
                features_w_l = self.back_norm_w_snorm(features_norm_l)
                mproposals_l, rpn_losses_ml, _, _ = self.rpn_norm(images, features_w_l, img_metas, targets)
                _, box_losses_ml, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w_l, mproposals_l, img_metas, targets)
                rpn_losses_m += sum(rpn_losses_ml.values())
                box_losses_m += sum(box_losses_ml.values())
            '''
            
                
            
            outputs['rpn_losses_m'] = rpn_losses_m
            outputs['box_losses_m'] = box_losses_m
            outputs['features_norm'] = features_norm
            outputs['dets_w'] = dets_w
        
        elif mode== 3: #eval_Net training
            with torch.no_grad():
                features_prev = self.prev_norm(images)
                #with norm
                if update_norm_p:
                    self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
                
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)

            with torch.no_grad():
                #features_w = self.back_norm_w_snorm(features_norm)
                
                dets_w_set = []
                len_num = len(feat_dict['feature_dict'])
                results_parameters = {'score_thresh':0.3, 'nms_thresh':0.6}
                for idx_num in range(len_num):
                    features_w = self.back_norm_w_snorm(feat_dict['feature_dict'][idx_num])
                    proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
                    dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets, results_parameters=results_parameters)
                    dets_w_set.append(dets_w)
            
            outputs['dets_w'] = dets_w_set
        
        ####detection training####
        elif mode== 4: 
            features_prev = self.prev_norm(images)

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            #print('----1')
            #features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            #use_dbiase=use_dbiase, moe_mode=moe_mode)
            features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
        
        elif mode== 5: #detection training
            features_prev = self.prev_norm(images)

            # E3 detection path explicitly invokes the iterative AHDR renderer.
            if os.environ.get("SNORM_EXPERIMENT", "E0").upper() == "E3":
                moe_mode = 8

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)
            #features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
        
        #####e-net####
        elif mode== 6: 
            features_prev = self.prev_norm(images)

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            #print('----1')
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)
            #features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev

        elif mode== 7: 
            features_prev = self.prev_norm(images)
            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #print('----1')
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)
            #features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev

        elif mode== 8: 
            features_prev = self.prev_norm(images)

            dets = None
            dets_w = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            #print('----1')
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode, moe_mark=moe_mark)

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_norm'] = features_norm
            if feat_dict is not None:
                if 'attn' in feat_dict.keys():
                    outputs['attn'] = feat_dict['attn']
                if 'attn_masks' in feat_dict.keys():
                    outputs['attn_masks'] = feat_dict['attn_masks']

        outputs['feat_dict'] = feat_dict
            
        return outputs
    
    def dist_prog(self, x, targets, stride=8, use_dbiase=False, domain='source', use_moe=False, use_moe_fusion=False, moe_mode=0, moe_mark=0):
        if self.cfg.MODEL.MODE.PJOTYPE == 'BN':
            _empty = {
                'feature_dict': [],
                'feature_dict_per': [],
                'feature_dict_fused': [],
                'weight_dict': [],
                'win_sz_set': [],
                'attn': [],
                'attn_masks': None,
            }
            return x, _empty
        elif self.cfg.MODEL.MODE.PJOTYPE == 'NBIAS':
            nx, feat_dict = self.proj_norm(x, use_moe=use_moe, use_moe_fusion=use_moe_fusion, moe_mode=moe_mode, moe_mark=moe_mark)
            if  use_dbiase:
                domain_num = 0 if domain == 'source' else 1
                nx = self.proj_dbias.read_memory(nx, targets, stride=stride, domain=1)
                self.pjo_ops.update_memory(x, targets, stride=stride, domain=1)
            return nx, feat_dict
    
    def compute_targets_for_locations(self, locations, targets, feats):
        bn, lc, lh, lw = feats.size()
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        #print(locations.size(), len(targets))
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im['boxes']
            labels_per_im = targets_per_im['labels']
            
            if bboxes.size(0) == 0:
                labels_per_im = torch.zeros_like(locations[:,0])
                
                reg_targets_per_im = torch.zeros_like(torch.cat([locations, locations], dim=1))
                
            else:
                l = xs[:, None] - bboxes[:, 0][None]
                t = ys[:, None] - bboxes[:, 1][None]
                r = bboxes[:, 2][None] - xs[:, None]
                b = bboxes[:, 3][None] - ys[:, None]
                reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

                max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]

                locations_to_gt_area = torch.ones_like(max_reg_targets_per_im)#area[None].repeat(len(locations), 1)
                locations_to_gt_area[is_in_boxes == 0] = 0

                # if there are still more than one objects for a location,
                # we choose the one with minimal area
                locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.max(dim=1)

                reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
                labels_per_im = labels_per_im[locations_to_gt_inds]
                labels_per_im[locations_to_min_area == 0] = 0
                
                #print('--0:', labels_per_im.size())
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
        
        #print(locations.size(), len(targets),feats.size())
        labels = torch.stack(labels, dim=0)
        labels = labels.view(bn, lh, lw)
        
        reg_targets = torch.stack(reg_targets, dim=0)
        reg_targets = reg_targets.view(bn, lh, lw, 4)
        
        return labels, reg_targets
    
    def compute_locations(self, features, fpn_strides=[8]):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    
    def tranformer_target3(self, features, img_metas, targets_, vis=False):
        targetn_ = []
        nfeatures = []
        nimg_metas = []
        
        for idx in range(len(targets_)):
            target_ = targets_[idx]

            if target_['labels'].size(0) == 0:
                continue
            
            bbs = target_['boxes'].clone().detach()
            lls = target_['labels'].clone().detach()
            
            if len(bbs.size()) == 2 and bbs.size(0) > 0:
                #nimages.append(images[idx])
                nfeatures.append(features[idx])
                nimg_metas.append(img_metas[idx])
                targetn_.append({'boxes':bbs, 'labels': lls})
        
        if len(nfeatures) > 0:
            #nimages = torch.stack(nimages, dim=0)
            nfeatures = torch.stack(nfeatures, dim=0)
            return nfeatures, nimg_metas, targetn_
        else:
            return features, img_metas, None