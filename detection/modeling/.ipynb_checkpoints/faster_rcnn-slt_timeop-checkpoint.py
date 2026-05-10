import numpy as np
import random
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

class InstanceExtractor(nn.Module):
    def __init__(self, in_channels):
        super(InstanceExtractor, self).__init__()
        
        self.extractor = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels)
        )
        
        self.grad_weight = GradientReversal(-10.0)

    def forward(self, x):
        x = x + self.extractor(self.grad_weight(x))
        
        return x

class MAE_Decoder(nn.Module):
    def __init__(self, in_channels, ada_layers, cfg):
        super(MAE_Decoder, self).__init__()
        
        for idx in range(len(ada_layers)):
            decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            
            self.add_module('decoder'+str(idx+2), decoder)
        
        self.grad_weight = GradientReversal(-10.0)

    def forward(self, x):
        for idx in range(len(x)):
            decoder = getattr(self, 'decoder'+str(idx+2))
            re_ims = decoder(x[idx])
            self.loss(re_ims, )
        return x

class ModuleSet(nn.Module):
    def __init__(self, cfg, positive_fraction=0.25, use_eval_head = False, eval_head_num=5, use_memory_conv=False):
        super(ModuleSet, self).__init__()
        self.cfg = cfg
        backbone = build_backbone(cfg)
        in_channels = backbone.out_channels

        self.backbone = backbone
        self.rpn = RPN(cfg, in_channels)
        self.box_head = BoxHead(cfg, in_channels)
        #self.eval_head = BoxHead(cfg, in_channels)
        #self.eval_head = BoxHead(cfg, in_channels, dropout_i=False, is_eval_head=True, positive_fraction=0.4, BATCH_SIZE_PER_IMAGE=20)
        
        if use_eval_head:
            #self.eval_head_cp = BoxHead(cfg, in_channels, dropout_i=False, is_eval_head=True)
            self.eval_train_head = BoxHead(cfg, in_channels, dropout_i=False, is_eval_head=True, positive_fraction=0.4, BATCH_SIZE_PER_IMAGE=20, use_memory_conv=use_memory_conv)
            #self.eval_eval_head = BoxHead(cfg, in_channels, dropout_i=False, is_eval_head=True, positive_fraction=0.4, BATCH_SIZE_PER_IMAGE=20, use_memory_conv=use_memory_conv)
        
        self.ada_layers = [False, True, True]
        self.ada_layers_stride = []
        
        self.llayers_rois = [[0,0.5], [0.2, 0.5], [0.2, 0.5]]
        self.ada_layers_rois = []

        self.positive_fraction = positive_fraction
    
    def forward_vgg16(self, x):
        b,c, h, w = x.size()
        adaptation_feats = []
        adaptation_strides = []
        ada_layers_rois = []
        idx = 0
        for i in range(14):
            x = self.backbone[i](x)
            
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
            ada_layers_rois.append(self.llayers_rois[idx])

        idx += 1
        for i in range(14, 21):
            x = self.backbone[i](x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
            ada_layers_rois.append(self.llayers_rois[idx])
            #print('--t0:',x.size())

        idx += 1
        for i in range(21, len(self.backbone)):
            x = self.backbone[i](x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
            ada_layers_rois.append(self.llayers_rois[idx])
            #print('--t1:',x.size())
            
        #print('--adv:', len(adaptation_feats))
        self.ada_layers_stride = adaptation_strides
        self.ada_layers_rois = ada_layers_rois
        
        return x, adaptation_feats

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
    
    def forward(self, images, img_metas, targets=None, proposals_=None, is_eval_update=False, results_parameters=None, is_use_no_grad=False, output_tcs=None):
        outputs = dict()
        #function define
        forward_func = getattr(self, 'forward_{}'.format(self.cfg.MODEL.BACKBONE.NAME))
        
        if is_eval_update:
            if results_parameters is None:
                outputs['dets'] = None
                outputs['box_losses_el'] = 0
                return outputs

            if self.training:
                if targets is None:
                    #print('----2:', len(results_parameters))
                    ##
                    with torch.no_grad():
                        features, adpat_feats = forward_func(images)
                        proposals, rpn_losses, rpn_logits, rpn_boxes = self.rpn(images, features, img_metas, targets)
                        dets_el_tr, box_losses, _, _, _, _, _, _, _ = self.box_head(features, proposals, img_metas, targets, results_parameters=results_parameters)
                        #dets_el_tr, box_losses, _, _, _, _, _, _, _ = self.box_head(features, fproposals, img_metas, targets, results_parameters=results_parameters)
                    outputs['dets'] = dets_el_tr
                else:
                    features, adpat_feats = forward_func(images)
                    box_losses_el = []
                    for idx in range(len(results_parameters)):
                        #print(dets_el_tr[0])
                        nfeatures, nimg_metas, ntargets = self.tranformer_target3(features, img_metas, targets[idx])
                        if ntargets is None:
                            box_losses_el.append(None)
                        else:
                            nfproposals, rpn_losses, rpn_logits, rpn_boxes = self.rpn(images, nfeatures, nimg_metas, ntargets)
                            l_dets, l_box_losses, l_proposals, l_box_features, l_roi_features, l_class_logits, l_labels, l_obj_dets, l_box_regression = \
                            self.eval_train_head(nfeatures, nfproposals, nimg_metas, ntargets, results_parameters=results_parameters[idx], eval_pms_mode=True)
                            box_losses_el.append(l_box_losses)
                        
                    outputs['box_losses_el'] = box_losses_el
            else:
                dets_el = []
                with torch.no_grad():
                    features, adpat_feats = forward_func(images)
                    proposals, rpn_losses, rpn_logits, rpn_boxes = self.rpn(images, features, img_metas, targets)
                    for idx in range(len(results_parameters)):
                        l_dets, l_box_losses, l_proposals, l_box_features, l_roi_features, l_class_logits, l_labels, l_obj_dets, l_box_regression = \
                        self.eval_train_head(features, proposals, img_metas, targets, results_parameters=results_parameters[idx], eval_pms_mode=True)
                        dets_el.append(l_dets)
                outputs['dets'] = dets_el
        else:
            ##rpn and rcnn
            if is_use_no_grad:
                with torch.no_grad():
                    features, adpat_feats = forward_func(images)
                    proposals, rpn_losses, rpn_logits, rpn_boxes = self.rpn(images, features, img_metas, targets)
                
                    if proposals_ is not None:
                        fproposals = proposals_
                    else:
                        fproposals = proposals
                
                    ##
                    dets, box_losses, proposals, box_features, roi_features, class_logits, labels, obj_dets, box_regression = \
                    self.box_head(features, fproposals, img_metas, targets)
                    dets_bb = obj_dets['dets_bb']

            else:
                features, adpat_feats = forward_func(images)
                proposals, rpn_losses, rpn_logits, rpn_boxes = self.rpn(images, features, img_metas, targets)
            
                if proposals_ is not None:
                    fproposals = proposals_
                else:
                    fproposals = proposals
                
                ##
                dets, box_losses, proposals, box_features, roi_features, class_logits, labels, obj_dets, box_regression = \
                self.box_head(features, fproposals, img_metas, targets)
                #print('-----ss')
                #dets, box_losses, proposals, box_features, roi_features, class_logits, labels, obj_dets, box_regression = \
                #self.eval_train_head(features, fproposals, img_metas, targets)
                dets_bb = None

            outputs['rpn_losses'] = rpn_losses
            outputs['box_losses'] = box_losses
            outputs['box_features'] = box_features
            outputs['class_logits'] = class_logits
            outputs['dets'] = dets
            outputs['labels'] = labels
            outputs['features'] = features.clone().detach()
            outputs['dets_bb'] = dets_bb
            outputs['obj_dets'] = obj_dets

            nproposals = []
            for idx_p in range(len(fproposals)):
                nproposals.append(fproposals[idx_p].clone().detach())
            outputs['fproposals'] = nproposals
            #print(len(proposals), proposals[0].size())
        
            
        return outputs
    
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


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        self.cfg = cfg
        self.class_num = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.anchor_num = len(cfg.MODEL.RPN.ANCHOR_SIZES) * len(cfg.MODEL.RPN.ASPECT_RATIOS)

        ##
        self.copy_eval_head = None
        self.use_eval_head = True
        self.eval_head_per_cls_num = 10

        self.eval_param_num = 1
        self.eval_param_head_num = 10
        self.eval_head_num = self.eval_param_head_num * self.eval_param_num

        
        self.teacher = ModuleSet(cfg)
        if cfg.MODEL.MODE.USE_STUDENT:
             self.student = ModuleSet(cfg)
             #self.eval_net = ModuleSet(cfg, use_eval_head=self.use_eval_head, eval_head_num=self.eval_head_num, use_memory_conv=True)
             
        self.update_teacher_interum = cfg.MODEL.MODE.UPDATE_TEACHER_ITERUM
        self.NET_MOMENTUM = cfg.MODEL.MODE.NET_MOMENTUM
        
        self.out_fpn_strides=[]
        
        self.loss_wegihts = [0.1, 1.0, 0.0, 1.0, 1.0, 0.0]
        
        
        #self.grad_weight = GradientReversal(-0.01)
        self.grad_weight = GradientReversal(-0.001)
        self.global_weight = 0
        
        self.images_tc = None
        self.images_pd = None
        self.images_src = None
        self.loss_hs = 100000
        
        self.index_iter = 0
        
        self.neg_weight = 0.0
        
        
        

    def forward(self, tc_images, img_metas, targets_=None, domain='s', step_iter=0, stsc_images=None, stsc_trans=None, stpd_images=None, stpd_trans=None, outputs_teacher=None, epoch=0, vis=False, \
        network_gated_mark_psudo=1, head_gated_mark_psudo=0, network_gated_mark_psudo_el=1, head_gated_mark_psudo_el=0, \
        network_gated_mark_train=2, head_gated_mark_train=0, network_gated_mark_train_el=0, head_gated_mark_train_el=1, \
        network_gated_mark_test=1, head_gated_mark_test=0, network_gated_mark_test_el=0, head_gated_mark_test_el=1, \
        is_eval_update=False, results_parameters=None, only_psudo_el=False):
        '''
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        '''
        #output dict
        outputs = dict()
        loss_dict = dict()
        
        if self.training:
            if self.cfg.MODEL.MODE.TRAIN_PROCESS == 'S0':
                 outputs_teacher = self.teacher(tc_images, img_metas, targets_)
                 for key, value in outputs_teacher['rpn_losses'].items():
                     loss_dict.update({key+'_ts':value})
                 for key, value in outputs_teacher['box_losses'].items():
                     loss_dict.update({key+'_ts':value})
                
                 return loss_dict, outputs_teacher
            
            if is_eval_update :   
                #print('---1:', tc_images.size())
                if network_gated_mark_psudo_el == 0:
                    outputs_teacher = self.eval_net(tc_images, img_metas, targets_, is_eval_update=is_eval_update, results_parameters=results_parameters, output_tcs=outputs_teacher)
                elif network_gated_mark_psudo_el == 1:
                    outputs_teacher = self.teacher(tc_images, img_metas, targets_, is_eval_update=is_eval_update, results_parameters=results_parameters, output_tcs=outputs_teacher)
                else:
                    outputs_teacher = self.student(tc_images, img_metas, targets_, is_eval_update=is_eval_update, results_parameters=results_parameters, output_tcs=outputs_teacher)
                
                #print(outputs_teacher.keys())
                if network_gated_mark_train_el == 0:
                    outputs_eval = self.eval_net(tc_images, img_metas, outputs_teacher['dets'], is_eval_update=is_eval_update, results_parameters=results_parameters, output_tcs=outputs_teacher)
                elif network_gated_mark_train_el == 1:
                    outputs_eval = self.teacher(tc_images, img_metas, outputs_teacher['dets'], is_eval_update=is_eval_update, results_parameters=results_parameters, output_tcs=outputs_teacher)
                else:
                    outputs_eval = self.student(tc_images, img_metas, outputs_teacher['dets'], is_eval_update=is_eval_update, results_parameters=results_parameters, output_tcs=outputs_teacher)
                
                
                if targets_ is None:
                    weight = 1.0       
                    box_losses_els = 0
                    print(len(outputs_eval['box_losses_el']), outputs_eval['box_losses_el'])
                    for idx_iter_sm in range(self.eval_head_num):
                        if outputs_eval['box_losses_el'][idx_iter_sm] is not None:
                            for key, value in outputs_eval['box_losses_el'][idx_iter_sm].items():
                                box_losses_els += weight * value
                    
                    loss_dict.update({'box_losses_els': box_losses_els})
                
                return loss_dict, outputs_eval

            if outputs_teacher is None and targets_ is None:
                #with torch.no_grad():
                if network_gated_mark_psudo == 0:
                    outputs_teacher = self.eval_net(tc_images, img_metas, targets_, is_use_no_grad=True)
                elif network_gated_mark_psudo == 1:
                    outputs_teacher = self.teacher(tc_images, img_metas, targets_, is_use_no_grad=True)
                else:
                    outputs_teacher = self.student(tc_images, img_metas, targets_, is_use_no_grad=True)
                    

            if targets_ is None:
                targets_ = self.generate_target(outputs_teacher['dets'])
                #targets_bb = self.generate_target(outputs_teacher['obj_dets']['dets_bb'])
                targets_bb = None
                #if not is_eval_update:
                #    targets_bb = self.generate_target(outputs_teacher['obj_dets']['dets_bb'])
                
                if only_psudo_el:
                    outputs_teacher['ims'] = tc_images
                    outputs_teacher['img_metas'] = outputs_teacher['obj_dets']['img_metas']
                    outputs_teacher['targets'] = outputs_teacher['dets_bb']
                    
                    return loss_dict, outputs_teacher

             
            if stsc_images is not None:
                stsc_images, nimg_metas, st_targets_ = self.tranformer_target(tc_images, stsc_images, img_metas, targets_, stsc_trans, None, vis=vis)
                
                if stsc_images is None:
                    loss_dict = None
                else:
                    weight = 1.0
                    if network_gated_mark_train == 0:
                        outputs_stsc = self.eval_net(stsc_images, nimg_metas, st_targets_)

                        for idx_iter_sm in range(self.eval_head_num):
                            for key, value in outputs_stsc['rpn_losses'][idx_iter_sm].items():
                                loss_dict.update({key+'_sc'+str(idx_iter_sm): weight * value})
                            for key, value in outputs_stsc['box_losses'][idx_iter_sm].items():
                                loss_dict.update({key+'_sc'+str(idx_iter_sm):weight * value})
                    elif network_gated_mark_train == 1:
                        outputs_stsc = self.teacher(stsc_images, nimg_metas, st_targets_)

                        for key, value in outputs_stsc['rpn_losses'].items():
                            loss_dict.update({key+'_sc': weight * value})
                        for key, value in outputs_stsc['box_losses'].items():
                            loss_dict.update({key+'_sc':weight * value})
                    else:
                        outputs_stsc = self.student(stsc_images, nimg_metas, st_targets_)

                        for key, value in outputs_stsc['rpn_losses'].items():
                            loss_dict.update({key+'_sc': weight * value})
                        for key, value in outputs_stsc['box_losses'].items():
                            loss_dict.update({key+'_sc':weight * value})

                    if outputs_teacher is None:
                        outputs_teacher = {}
                        outputs_teacher['ims'] = stsc_images
                        outputs_teacher['img_metas'] = outputs_stsc['obj_dets']['img_metas']
                        outputs_teacher['targets'] = st_targets_
                    else:
                        outputs_teacher['ims'] = tc_images
                        outputs_teacher['img_metas'] = outputs_teacher['obj_dets']['img_metas']
                        outputs_teacher['targets'] = outputs_teacher['dets_bb']
                
            
            if stpd_images is not None:
                self.images_pd = stpd_images.clone().detach()
                self.images_src = tc_images.clone().detach()
                #stpd_targets_ = self.tranformer_target(targets_, stpd_trans)
                stpd_proposals = self.tranformer_proposals(stpd_images, outputs_teacher['proposals'], stpd_trans)
                #self.draw_proposals(tc_images, outputs_teacher['proposals'], stpd_images, stpd_proposals)
                #print(stpd_images.size())
                outputs_stpd = self.student(stpd_images, img_metas, None, proposals_=stpd_proposals)
                
                attn_maps = self.generate_attention_maps(img_metas, outputs_teacher['proposals'])
                loss_consis = self.align_consisence(outputs_teacher['features'], outputs_stpd['features'], attn_maps, stpd_trans, outputs_teacher['obj_dets'])
                
                
                det_loss = 0.0 * outputs_stpd['class_logits'].mean()
                loss_dict.update({'det_loss': det_loss})
                loss_dict.update({'loss_consis': self.loss_wegihts[0] * loss_consis})
        
                
            return loss_dict, outputs_teacher
            
        else:
            if is_eval_update:
                if network_gated_mark_test_el == 0:
                    outputs = self.eval_net(tc_images, img_metas, targets_, is_eval_update=is_eval_update, results_parameters=results_parameters,\
                    output_tcs=outputs_teacher)
                elif network_gated_mark_test_el == 1:
                    outputs = self.teacher(tc_images, img_metas, targets_, is_eval_update=is_eval_update, results_parameters=results_parameters,\
                    output_tcs=outputs_teacher)
                else:
                    outputs = self.student(tc_images, img_metas, targets_, is_eval_update=is_eval_update, results_parameters=results_parameters,\
                    output_tcs=outputs_teacher)

            else:
                #print('----network_gated_mark_test:', network_gated_mark_test)
                if network_gated_mark_test == 0:
                    outputs = self.eval_net(tc_images, img_metas, targets_)
                elif network_gated_mark_test == 1:
                    outputs = self.teacher(tc_images, img_metas, targets_)
                else:
                    outputs = self.student(tc_images, img_metas, targets_)
            
            return outputs['dets']
    
    def generate_target(self, dets, results_parameters=None):
        #dets['masks'] = None
        ndets = []
        if results_parameters is None:
            for idx in range(len(dets)):
                #dets[idx]['masks'] = None
                ndets.append({'labels':dets[idx]['labels'].clone().detach(), 'boxes':dets[idx]['boxes'].clone().detach()})
        else:
            for idx_eval_head_num in range(len(results_parameters)):
                lndets = []
                #print('---s:', len(results_parameters), len(dets))
                for idx in range(len(dets[idx_eval_head_num])):
                    #dets[idx]['masks'] = None
                    lndets.append({'labels':dets[idx_eval_head_num][idx]['labels'].clone().detach(), 'boxes':dets[idx_eval_head_num][idx]['boxes'].clone().detach()})
                
                ndets.append(lndets)
        return ndets
    
    def tranformer_feats(self, feats_tc, feats_st, st_trans, cls_weights=None, txt_weights=None, type='ft', class_c=8, loss_txt_weight = [1.0],  region_roi_masks=None, scopes_sz=16, layer_id=0):
        nfeats = []
        
        B, C, H, W = feats_st.size()
        mseloss = torch.nn.MSELoss(reduction='none')
        '''
        avg_1_8 = nn.AvgPool2d((1,1), stride=7, padding=0)
        avg_5_8 = nn.AvgPool2d((5,5), stride=7, padding=2)
        avg_7_8 = nn.AvgPool2d((7,7), stride=7, padding=3)
        avg_9_8 = nn.AvgPool2d((9,9), stride=7, padding=4)
        avg_3_8 = nn.AvgPool2d((3,3), stride=7, padding=1)
        '''
        
        avg_1_8 = nn.AvgPool2d((1,1), stride=7, padding=0)
        avg_5_8 = nn.AvgPool2d((5,5), stride=7, padding=2)
        avg_7_8 = nn.AvgPool2d((5,5), stride=7, padding=2)
        avg_9_8 = nn.AvgPool2d((7,7), stride=7, padding=3)
        avg_3_8 = nn.AvgPool2d((3,3), stride=7, padding=1)
        
        loss_pixel_ldi = 0.0
        pixel_tc_feats = []
        pixel_st_feats = []
        pixel_clses = []
        for idx in range(len(st_trans)):
            st_tran = st_trans[idx]
            lfeats_tc = feats_tc[idx:idx+1]
            lfeats_st = feats_st[idx:idx+1]
            
            if cls_weights is not None:
                 lcls_weights = (cls_weights[idx:(idx+1)]+0.05).long()
            else:
                 lcls_weights = torch.ones_like(lfeats_tc[:,0,:,:]).float()
            
            if txt_weights is not None:
                 ltxt_weights = txt_weights[idx:(idx+1)].long()
            else:
                 ltxt_weights = torch.zeros_like(lfeats_tc[:,0,:,:]).long()
            
            if 'RVF' in st_tran.keys():
                lfeats_st = lfeats_st.flip([2])
                
            if 'RHF' in st_tran.keys():
                lfeats_st = lfeats_st.flip([3])
                
            #scale
            if 'RandomSized' in st_tran.keys():
                lh, lw = int(H / st_tran['RandomSized'][1]), int(W / st_tran['RandomSized'][0])
                lfeats_st = F.interpolate(lfeats_st, size=(lh, lw), mode='bilinear')
                
                ow, oh = self.images_pd.size(3), self.images_pd.size(2)
                ih, iw = int(oh / st_tran['RandomSized'][1]), int(ow / st_tran['RandomSized'][0])
                limages_pd = F.interpolate(self.images_pd[idx:idx+1], size=(ih, iw), mode='bilinear')
                
                
                if lh > H or lw > W:
                     pad_h = int((lh - H)/2.0)
                     pad_w = int((lw - W)/2.0)
                     if pad_h < 0:
                         pad_h = 0
                     if pad_w < 0:
                         pad_w = 0
                    
                     nlfeats_st = lfeats_st[:, :, pad_h:(pad_h+H), pad_w:(pad_w+W)]
                     nlfeats_tc = lfeats_tc
                     nlcls_weights = lcls_weights
                     nltxt_weights = ltxt_weights
                     
                     
                     pad_ih = int((ih - oh)/2.0)
                     pad_iw = int((iw - ow)/2.0)
                     if pad_ih < 0:
                         pad_ih = 0
                     if pad_iw < 0:
                         pad_iw = 0
                     limages_pd = limages_pd[:, :, pad_ih:(pad_ih+oh), pad_iw:(pad_iw+ow)]
                     
                      
                else:
                     pad_h =  int((H - lh)/2.0)
                     pad_w = int((W - lw)/2.0)
                     if pad_h < 0:
                          pad_h = 0
                     if pad_w < 0:
                          pad_w = 0
                     
                     nlfeats_tc = lfeats_tc[:, :, pad_h:(pad_h+lh), pad_w:(pad_w+lw)]
                     nlcls_weights = lcls_weights[:, pad_h:(pad_h+lh), pad_w:(pad_w+lw)]
                     nltxt_weights = ltxt_weights[:, pad_h:(pad_h+lh), pad_w:(pad_w+lw)]
                     nlfeats_st = lfeats_st
                     limages_pd = limages_pd

            else:
                nlfeats_tc = lfeats_tc
                nlfeats_st = lfeats_st
                nlcls_weights = lcls_weights
                nltxt_weights = ltxt_weights
                limages_pd = self.images_pd[idx:idx+1]
            
            if 'Translate' in st_tran.keys():
                if st_tran['Translate'][0] > st_tran['Translate'][1]:
                     Translate_mx = st_tran['Translate'][0]
                else:
                     Translate_mx = st_tran['Translate'][1]
                lscope = Translate_mx * 1.0 / scopes_sz
                #03_14: 1.0-52.5
                #03_15: 1.5-
                if lscope > 1.0:
                    im_weights = 0.0
                else:
                    im_weights = 1.0
            else:
                im_weights = 1.0
            
            #draw
            
            #print(nlcls_weights.min(), nlcls_weights.max())
            '''
            npos_weights1 = (nlcls_weights > 0.5).float()
            npos_weights2 = (nltxt_weights > 1.5).float()
            npos_weights3 = (nlcls_weights > 0.5).float() * (nltxt_weights > 1.5).float()
            lnpos_weights1 = F.interpolate(npos_weights1.unsqueeze(1), size=(limages_pd.size(2), limages_pd.size(3)), mode='bilinear',align_corners=False)
            lnpos_weights2 = F.interpolate(npos_weights2.unsqueeze(1), size=(limages_pd.size(2), limages_pd.size(3)), mode='bilinear',align_corners=False)
            lnpos_weights3 = F.interpolate(npos_weights3.unsqueeze(1), size=(limages_pd.size(2), limages_pd.size(3)), mode='bilinear',align_corners=False)
            #lim = self.images_pd[idx:idx+1]
            lim = limages_pd
            lim -= lim.min()
            lim0 = lim.mul(lnpos_weights1)
            lim1 = lim.mul(lnpos_weights2)
            lim2 = lim.mul(lnpos_weights3)
            
            lim0 = lim0[0].permute(1,2,0).contiguous()
            lim1 = lim1[0].permute(1,2,0).contiguous()
            lim2 = lim2[0].permute(1,2,0).contiguous()
            
            cv_im0 = lim0.cpu().detach().numpy()
            cv_im1 = lim1.cpu().detach().numpy()
            cv_im2 = lim2.cpu().detach().numpy()
            
            cv_im0 = np.array(cv_im0*255/2, np.int32)
            cv_im1 = np.array(cv_im1*255/2, np.int32)
            cv_im2 = np.array(cv_im2*255/2, np.int32)
            #print(cv_im.min(), cv_im.max())
            cv2.imwrite('./attn/attn_pd_'+str(self.index_iter) +'_0.jpg', cv_im0)
            cv2.imwrite('./attn/attn_pd_'+str(self.index_iter) +'_1.jpg', cv_im1)
            cv2.imwrite('./attn/attn_pd_'+str(self.index_iter) +'_2.jpg', cv_im2)
            #cv2.imwrite('./attention_test_tc.jpg', cv_im)
            self.index_iter +=1
            '''
            
            ###ld
            el_pixels_st = nlfeats_st# - avg_7_8(nlfeats_st)
            el_pixels_tc = nlfeats_tc# - avg_7_8(nlfeats_tc)
            
            #l2
            #print(el_pixels_st.size(), el_pixels_tc.size(), el_pixels_st[:,:, 1:-1, 1:-1].size())
            lloss = mseloss(el_pixels_st, el_pixels_tc).mean(dim=1)
            #dc
            klcls_weights = (nlcls_weights > 0.5).float()
            for idx_c in range(len(loss_txt_weight)):
                lmask = (nltxt_weights == idx_c).float()
                
                lmask_pos = lmask.mul(klcls_weights)
                lmask_neg = lmask.mul(1 - klcls_weights)
                llloss = lloss[:, 3:-3, 3:-3]
                lmask_pos = lmask_pos[:, 3:-3, 3:-3]
                lmask_neg = lmask_neg[:, 3:-3, 3:-3]

                lloss_l0 = (llloss.mul(lmask_pos).sum()).div(lmask_pos.sum()+1)
                lloss_l1 = (llloss.mul(lmask_neg).sum()).div(lmask_neg.sum()+1)
                
                lloss_lf = lloss_l0 + self.neg_weight * lloss_l1
                #lloss_lf = lloss_l0 + 0.02 * lloss_l1
                
                loss_pixel_ldi += im_weights * loss_txt_weight[idx_c] * lloss_lf
            
            ##hd
            if 'Translate' in st_tran.keys():
                mb, mc, mh, mw = nlfeats_tc.size()
                x0 = 0
                y0 = 0
                x1 = mw
                y1 = mh
                x10 = 0
                y10 = 0
                if st_tran['Translate'][2] < 0:
                     lx = int(st_tran['Translate'][2] / scopes_sz)
                     x1 = x1 + lx
                     x10 = -lx
                else:
                     lx = int(st_tran['Translate'][2] / scopes_sz)
                     x0 = x0 + lx
                     
                if st_tran['Translate'][3] < 0:
                     ly = int(st_tran['Translate'][3] / scopes_sz)
                     y1 = y1 + ly
                     y10 = -ly
                else:
                     ly = int(st_tran['Translate'][3] / scopes_sz)
                     y0 = y0 + ly
                
                mlfeats_tc = nlfeats_tc[:,:, y0:y1, x0:x1]
                mlfeats_st = nlfeats_st[:,:, y10:y10+(y1-y0), x10:x10+(x1-x0)]
                mlcls_weights = nlcls_weights[:, y0:y1, x0:x1]
                mltxt_weights = nltxt_weights[:, y0:y1, x0:x1]
            else:
                mlfeats_tc = nlfeats_tc
                mlfeats_st = nlfeats_st
                mlcls_weights = nlcls_weights
                mltxt_weights = nltxt_weights
            
            label_cls_maps = torch.zeros((mlcls_weights.size(0), class_c, mlcls_weights.size(1), mlcls_weights.size(2))).to(mlcls_weights.device).scatter_(1, mlcls_weights.unsqueeze(1), 1)
            
            label_txt_maps = torch.zeros((mlcls_weights.size(0), len(loss_txt_weight), mltxt_weights.size(1), mltxt_weights.size(2))).to(mltxt_weights.device).scatter_(1, mltxt_weights.unsqueeze(1), 1)
            
            #print(label_cls_maps.size(), label_txt_maps.size())
            #obtain vital pixel
            glcls_weights = avg_5_8(label_cls_maps)
            glcls_weights_v, glcls_weights_i = torch.topk(glcls_weights, 2, dim=1)
            glcls_weights_m0 = (glcls_weights_i[:,0] == 0).float()
            glcls_weights_m1 = (glcls_weights_v[:,1] > 0.05).float()
            glcls_weights_m2 = glcls_weights_m0.mul(glcls_weights_m1)
            
            glcls_weights_i = glcls_weights_i.float()
            glcls_weights_id = glcls_weights_i[:,0].mul(1 - glcls_weights_m2) + glcls_weights_i[:,1].mul(glcls_weights_m2)
            glcls_weights_id = (glcls_weights_id + 0.1).long()
            
            glcls_weights7 = avg_7_8(label_cls_maps) * 25
            glcls_weights9 = avg_9_8(label_cls_maps) * 49
            glcls_weights3 = avg_3_8(label_cls_maps) * 9
            
            glcls_weights_roi0 = (glcls_weights9 - glcls_weights7) < 0.5
            #glcls_weights_roi1 = (glcls_weights7 - glcls_weights3) > 1.0
            glcls_weights_roi1 = glcls_weights3 > 5
            glcls_weights_roi = glcls_weights_roi0 & glcls_weights_roi1
            glcls_weights_sroi = torch.gather(glcls_weights_roi, 1, glcls_weights_id.unsqueeze(1)).view(-1)
            
            
            gltxt_weights = avg_3_8(label_txt_maps)
            gltxt_weights_i = gltxt_weights[:,len(loss_txt_weight) -1].view(-1)
            glcls_weights_id = glcls_weights_id.view(-1)
            
            gel_pixels_tc = avg_1_8(mlfeats_tc).permute(0,2,3,1).contiguous().view(-1, C)
            gel_pixels_st = avg_1_8(mlfeats_st).permute(0,2,3,1).contiguous().view(-1, C)
            
            gltxt_weights_mask = gltxt_weights_i > 0.1
            fltxt_weights_mask = glcls_weights_sroi | ((glcls_weights_id == 0) & gltxt_weights_mask)
            
            flcls_weights = glcls_weights_id[fltxt_weights_mask]
            fel_pixels_st = gel_pixels_st[fltxt_weights_mask,:]
            fel_pixels_tc = gel_pixels_tc[fltxt_weights_mask,:]
            
            pixel_tc_feats.append(fel_pixels_tc)
            pixel_st_feats.append(fel_pixels_st)
            pixel_clses.append(flcls_weights)
            
        pixel_tc_feats = torch.cat(pixel_tc_feats,dim=0)
        pixel_st_feats = torch.cat(pixel_st_feats,dim=0)
        pixel_clses = torch.cat(pixel_clses,dim=0)
        
        loss_pixel_id = self.matrix_pixel_loss_v2(pixel_tc_feats, pixel_st_feats, pixel_clses, class_c=class_c, layer_id=layer_id)
        if loss_pixel_id is None:
              loss_pixel_id = 0.0 * feats_st.mean()
        
        return loss_pixel_ldi / B, loss_pixel_id
    
    def draw_proposals(self, sttc_images, sttc_proposals, stpd_images, stpd_proposals):
        self.global_weight += 1
        if self.global_weight < 10:
            for idx in range(len(stpd_proposals)):
                  proposal_pd = stpd_proposals[idx].clone().detach()
                  proposal_tc = sttc_proposals[idx].clone().detach()
                  
                  lim = sttc_images[idx].permute(1,2,0).contiguous()
                  lim -= lim.min()
                  cv_im = lim.cpu().detach().numpy()
                  cv_im = np.array(cv_im*255, np.int32)
                  
                  mbbs = proposal_tc.cpu().detach().numpy()
                  #print(mbbs.shape)
                  for ii in range(mbbs.shape[0]):
                      cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)
                  
                  cv2.imwrite('./proposals_tc_'+str(self.global_weight)+'.jpg', cv_im)
                      
                      
                  lim = stpd_images[idx].permute(1,2,0).contiguous()
                  lim -= lim.min()
                  cv_im = lim.cpu().detach().numpy()
                  cv_im = np.array(cv_im*255, np.int32)
                  
                  mbbs = proposal_pd.cpu().detach().numpy()
                  for ii in range(mbbs.shape[0]):
                      cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)
                  
                  #print('-----00:',lim.size())
                  cv2.imwrite('./proposals_pd'+str(self.global_weight)+'.jpg', cv_im)
    

    def tranformer_proposals(self, stpd_images, proposals, st_trans):
        proposaln = []
        for idx in range(len(proposals)):
            proposal = proposals[idx].clone().detach()
            st_tran = st_trans[idx]
            
            bbs = proposal
            #print(bbs)
            min_value = 10
            if 'Translate' in st_tran.keys():
                base_ = 0.5
                #ld_translate_x = random.random() * (1 - base_) + base_
                #ld_translate_y = random.random() * (1 - base_) + base_
                #ld_translate_x *= st_tran['Translate'][2]
                #ld_translate_y *= st_tran['Translate'][3]
                
                bbs[:, 0] -= st_tran['Translate'][2]
                bbs[:, 1] -= st_tran['Translate'][3]
                bbs[:, 2] -= st_tran['Translate'][2]
                bbs[:, 3] -= st_tran['Translate'][3]
                
                ww = bbs[:, 2] - bbs[:, 0]
                hh = bbs[:, 3] - bbs[:, 1]
                
                ldist_w = random.random() * 0.4 * ww
                if random.random() > 0.5:
                    ldist_w *= -1
                   
                ldist_h = random.random() * 0.4 * hh
                if random.random() > 0.5:
                    ldist_h *= -1
                    
                bbs[:, 0] += ldist_w
                bbs[:, 1] += ldist_h
                bbs[:, 2] += ldist_w
                bbs[:, 3] += ldist_h
                 
            if 'RandomSized' in st_tran.keys():
                 bbs[:, 0] *= st_tran['RandomSized'][0]
                 bbs[:, 1] *= st_tran['RandomSized'][1]
                 bbs[:, 2] *= st_tran['RandomSized'][0]
                 bbs[:, 3] *= st_tran['RandomSized'][1]
            
            if 'CenterCrop' in st_tran.keys():
                 x00 = bbs[:, 0] - st_tran['CenterCrop'][0]
                 y00 = bbs[:, 1] - st_tran['CenterCrop'][1]
                 x01 = bbs[:, 2] - st_tran['CenterCrop'][0]
                 y01 = bbs[:, 3] - st_tran['CenterCrop'][1]
                 
                 x0_mask = x00<0.0
                 y0_mask = y00<0.0
                 x00[x0_mask] = 0.0
                 y00[y0_mask] = 0.0
                 
                 
                 #x1_mask = (x01<0.0).float()
                 #y1_mask = (y01<0.0).float()
                 x01[x01<min_value] = min_value
                 y01[y01<min_value] = min_value
                 
                 #x00 = x00.mul(1-x1_mask) - x1_mask * 17
                 #y00 = y00.mul(1-y1_mask) - y1_mask * 17
                 
                 x00[x00>=st_tran['CenterCrop'][4]-min_value] = st_tran['CenterCrop'][4] - min_value
                 y00[y00>=st_tran['CenterCrop'][5]-min_value] = st_tran['CenterCrop'][5] - min_value
                 
                 x01[x01>=st_tran['CenterCrop'][4]] = st_tran['CenterCrop'][4]-1
                 y01[y01>=st_tran['CenterCrop'][5]] = st_tran['CenterCrop'][5] - 1
                 
                 bbs = torch.stack((x00, y00, x01, y01), dim=1)
            
            if 'RandomCrop' in st_tran.keys():
                x00 = bbs[:, 0] - st_tran['RandomCrop'][0]
                y00 = bbs[:, 1] - st_tran['RandomCrop'][1]
                x01 = bbs[:, 2] - st_tran['RandomCrop'][0]
                y01 = bbs[:, 3] - st_tran['RandomCrop'][1]
                
                mask0 = (x01 > 10) & (y01 > 10)
                
                x10 = st_tran['RandomCrop'][4] - x00
                y10 = st_tran['RandomCrop'][5] - y00
                x11 = st_tran['RandomCrop'][4] - x01
                y11 = st_tran['RandomCrop'][5] - y01
                
                mask1 = (x10 > 10) & (y10 > 10)
                
                final_mask = mask0 & mask1
                nbbs = torch.stack((x00, y00, x01, y01), dim=1)
                bbs = nbbs[final_mask, :]
                
                im_bbs = torch.zeros_like(bbs)
                im_bbs[:,0] = 0
                im_bbs[:,1] = 0
                im_bbs[:,2] = st_tran['RandomCrop'][4]
                im_bbs[:,3] = st_tran['RandomCrop'][5]
                
                ##gt iou
                gt_iou_ww = bbs[:,2] - bbs[:,0]
                gt_iou_hh = bbs[:,3] - bbs[:,1]
                gt_iou_ww[gt_iou_ww<0] = 0
                gt_iou_hh[gt_iou_hh<0] = 0
                gt_gl_areas = gt_iou_ww.mul(gt_iou_hh)
                
                gt_com_x0= torch.where((bbs[:, 0] - im_bbs[:, 0])>0, bbs[:, 0], im_bbs[:, 0])
                gt_com_x1 = torch.where((bbs[:, 2] - im_bbs[:, 2])<0, bbs[:, 2], im_bbs[:, 2])
                
                gt_com_y0= torch.where((bbs[:, 1] - im_bbs[:, 1])>0, bbs[:, 1], im_bbs[:, 1])
                gt_com_y1 = torch.where((bbs[:, 3] - im_bbs[:, 3])<0, bbs[:, 3], im_bbs[:, 3])
                
                
                gt_com_ww = gt_com_x1 - gt_com_x0
                gt_com_hh = gt_com_y1 - gt_com_y0
                gt_com_areas = gt_com_ww.mul(gt_com_hh)
                gt_iou = gt_com_areas.div(gt_gl_areas + 1e-5)
                iou_mask = gt_iou > 0.6
                
                bbs = bbs[iou_mask,:]
                
                
                bb2 = bbs[:, 2]
                bb3 = bbs[:, 3]
                bb2[bb2>=st_tran['RandomCrop'][4]] = st_tran['RandomCrop'][4] - 1
                bb3[bb3>=st_tran['RandomCrop'][5]] = st_tran['RandomCrop'][5] - 1
                bbs[:, 2] = bb2
                bbs[:, 3] = bb3
                bbs[bbs<0] = 0.0
                
                sww = bbs[:, 2] - bbs[:, 0]
                shh = bbs[:, 3] - bbs[:, 1]
                
                mask2 = (sww > 10) & (sww < 0.75 * st_tran['RandomCrop'][4])
                mask3 = (shh > 10) & (shh < 0.75 * st_tran['RandomCrop'][5])
                sz_mask = mask2 & mask3
                bbs = bbs[sz_mask,:]
            
            if 'RHF' in st_tran.keys():
                x0 = st_tran['RHF'][0] - bbs[:, 2]
                x1 = st_tran['RHF'][0] - bbs[:, 0]
                bbs[:, 0] = x0
                bbs[:, 2] = x1
            
            if 'RVF' in st_tran.keys():
                y0 = st_tran['RVF'][1] - bbs[:, 3]
                y1 = st_tran['RVF'][1] - bbs[:, 1]
                bbs[:, 1] = y0
                bbs[:, 3] = y1
            
            if len(bbs.size()) == 2 and bbs.size(0)>0:
                proposaln.append(bbs.clone().detach())
            
            
        return proposaln
    
    
    def tranformer_target(self, tc_images, st_images, img_metas, targets_, st_trans, targets_bb=None, vis=False):
        targetn_ = []
        stn_images = []
        nimg_metas = []
        for idx in range(len(targets_)):
            target_ = targets_[idx]
            st_tran = st_trans[idx]
            #print(target_['labels'].size(0))

            if target_['labels'].size(0) == 0:
                continue
            
            bbs = target_['boxes'].clone().detach()
            lls = target_['labels'].clone().detach()
            min_wh = 5
            if 'RandomSized' in st_tran.keys():
                 bbs[:, 0] *= st_tran['RandomSized'][0]
                 bbs[:, 1] *= st_tran['RandomSized'][1]
                 bbs[:, 2] *= st_tran['RandomSized'][0]
                 bbs[:, 3] *= st_tran['RandomSized'][1]
            
            if 'CenterCrop' in st_tran.keys():
                 x00 = bbs[:, 0] - st_tran['CenterCrop'][0]
                 y00 = bbs[:, 1] - st_tran['CenterCrop'][1]
                 x01 = bbs[:, 2] - st_tran['CenterCrop'][0]
                 y01 = bbs[:, 3] - st_tran['CenterCrop'][1]
                 
                 mask0 = (x01 > min_wh) & (y01 > min_wh)
                 
                 x10 = st_tran['CenterCrop'][4] - x00
                 y10 = st_tran['CenterCrop'][5] - y00
                 x11 = st_tran['CenterCrop'][4] - x01
                 y11 = st_tran['CenterCrop'][5] - y01
                 
                 mask1 = (x10 > min_wh) & (y10 > min_wh)
                 
                 final_mask = mask0 & mask1
                 nbbs = torch.stack((x00, y00, x01, y01), dim=1)
                 bbs = nbbs[final_mask, :]
                 lls = lls[final_mask]
                 
                 im_bbs = torch.zeros_like(bbs)
                 im_bbs[:,0] = 0
                 im_bbs[:,1] = 0
                 im_bbs[:,2] = st_tran['CenterCrop'][4]
                 im_bbs[:,3] = st_tran['CenterCrop'][5]
                 
                 ##gt iou
                 gt_iou_ww = bbs[:,2] - bbs[:,0]
                 gt_iou_hh = bbs[:,3] - bbs[:,1]
                 gt_iou_ww[gt_iou_ww<0] = 0
                 gt_iou_hh[gt_iou_hh<0] = 0
                 gt_gl_areas = gt_iou_ww.mul(gt_iou_hh)
                 
                 gt_com_x0= torch.where((bbs[:, 0] - im_bbs[:, 0])>0, bbs[:, 0], im_bbs[:, 0])
                 gt_com_x1 = torch.where((bbs[:, 2] - im_bbs[:, 2])<0, bbs[:, 2], im_bbs[:, 2])
                 
                 gt_com_y0= torch.where((bbs[:, 1] - im_bbs[:, 1])>0, bbs[:, 1], im_bbs[:, 1])
                 gt_com_y1 = torch.where((bbs[:, 3] - im_bbs[:, 3])<0, bbs[:, 3], im_bbs[:, 3])
                 
                 
                 gt_com_ww = gt_com_x1 - gt_com_x0
                 gt_com_hh = gt_com_y1 - gt_com_y0
                 gt_com_areas = gt_com_ww.mul(gt_com_hh)
                 gt_iou = gt_com_areas.div(gt_gl_areas + 1e-5)
                 iou_mask = gt_iou > 0.6
                 
                 bbs = bbs[iou_mask,:]
                 lls = lls[iou_mask]
                 
                 
                 bb2 = bbs[:, 2]
                 bb3 = bbs[:, 3]
                 bb2[bb2>=st_tran['CenterCrop'][4]] = st_tran['CenterCrop'][4] - 1
                 bb3[bb3>=st_tran['CenterCrop'][5]] = st_tran['CenterCrop'][5] - 1
                 bbs[:, 2] = bb2
                 bbs[:, 3] = bb3
                 bbs[bbs<0] = 0.0
                 
                 sww = bbs[:, 2] - bbs[:, 0]
                 shh = bbs[:, 3] - bbs[:, 1]
                 
                 mask2 = (sww > min_wh) & (sww < 0.8 * st_tran['CenterCrop'][4])
                 mask3 = (shh > min_wh) & (shh < 0.8 * st_tran['CenterCrop'][5])
                 sz_mask = mask2 & mask3
                 bbs = bbs[sz_mask,:]
                 lls = lls[sz_mask]
            
            if 'RandomCrop' in st_tran.keys():
                x00 = bbs[:, 0] - st_tran['RandomCrop'][0]
                y00 = bbs[:, 1] - st_tran['RandomCrop'][1]
                x01 = bbs[:, 2] - st_tran['RandomCrop'][0]
                y01 = bbs[:, 3] - st_tran['RandomCrop'][1]
                
                mask0 = (x01 > 10) & (y01 > 10)
                
                x10 = st_tran['RandomCrop'][4] - x00
                y10 = st_tran['RandomCrop'][5] - y00
                x11 = st_tran['RandomCrop'][4] - x01
                y11 = st_tran['RandomCrop'][5] - y01
                
                mask1 = (x10 > 10) & (y10 > 10)
                
                final_mask = mask0 & mask1
                nbbs = torch.stack((x00, y00, x01, y01), dim=1)
                bbs = nbbs[final_mask, :]
                lls = lls[final_mask]
                
                im_bbs = torch.zeros_like(bbs)
                im_bbs[:,0] = 0
                im_bbs[:,1] = 0
                im_bbs[:,2] = st_tran['RandomCrop'][4]
                im_bbs[:,3] = st_tran['RandomCrop'][5]
                
                ##gt iou
                gt_iou_ww = bbs[:,2] - bbs[:,0]
                gt_iou_hh = bbs[:,3] - bbs[:,1]
                gt_iou_ww[gt_iou_ww<0] = 0
                gt_iou_hh[gt_iou_hh<0] = 0
                gt_gl_areas = gt_iou_ww.mul(gt_iou_hh)
                
                gt_com_x0= torch.where((bbs[:, 0] - im_bbs[:, 0])>0, bbs[:, 0], im_bbs[:, 0])
                gt_com_x1 = torch.where((bbs[:, 2] - im_bbs[:, 2])<0, bbs[:, 2], im_bbs[:, 2])
                
                gt_com_y0= torch.where((bbs[:, 1] - im_bbs[:, 1])>0, bbs[:, 1], im_bbs[:, 1])
                gt_com_y1 = torch.where((bbs[:, 3] - im_bbs[:, 3])<0, bbs[:, 3], im_bbs[:, 3])
                
                
                gt_com_ww = gt_com_x1 - gt_com_x0
                gt_com_hh = gt_com_y1 - gt_com_y0
                gt_com_areas = gt_com_ww.mul(gt_com_hh)
                gt_iou = gt_com_areas.div(gt_gl_areas + 1e-5)
                iou_mask = gt_iou > 0.6
                
                bbs = bbs[iou_mask,:]
                lls = lls[iou_mask]
                
                
                bb2 = bbs[:, 2]
                bb3 = bbs[:, 3]
                bb2[bb2>=st_tran['RandomCrop'][4]] = st_tran['RandomCrop'][4] - 1
                bb3[bb3>=st_tran['RandomCrop'][5]] = st_tran['RandomCrop'][5] - 1
                bbs[:, 2] = bb2
                bbs[:, 3] = bb3
                bbs[bbs<0] = 0.0
                
                sww = bbs[:, 2] - bbs[:, 0]
                shh = bbs[:, 3] - bbs[:, 1]
                
                mask2 = (sww > 10) & (sww < 0.75 * st_tran['RandomCrop'][4])
                mask3 = (shh > 10) & (shh < 0.75 * st_tran['RandomCrop'][5])
                sz_mask = mask2 & mask3
                bbs = bbs[sz_mask,:]
                lls = lls[sz_mask]
            
            if 'RHF' in st_tran.keys():
                x0 = st_tran['RHF'][0] - bbs[:, 2]
                x1 = st_tran['RHF'][0] - bbs[:, 0]
                bbs[:, 0] = x0
                bbs[:, 2] = x1
            
            if 'RVF' in st_tran.keys():
                y0 = st_tran['RVF'][1] - bbs[:, 3]
                y1 = st_tran['RVF'][1] - bbs[:, 1]
                bbs[:, 1] = y0
                bbs[:, 3] = y1
            
            #target_['boxes'] = bbs
            #target_['labels'] = lls
            
            if len(bbs.size()) == 2 and bbs.size(0) > 0:
               targetn_.append({'boxes':bbs, 'labels': lls})
               stn_images.append(st_images[idx])
               nimg_metas.append(img_metas[idx])
               #print('----1')
               if vis:#'RHF' in st_tran.keys():
                   lim = st_images[idx].permute(1,2,0).contiguous()
                   lim -= lim.min()
                   cv_im = lim.cpu().detach().numpy()
                   cv_im = np.array(cv_im*255, np.int32)
                   
                   mbbs = bbs.cpu().detach().numpy()
                   for ii in range(bbs.size(0)):
                       cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)
                   
                   #print('-----00:',lim.size())
                   cv2.imwrite('./test/'+str(idx)+'.jpg', cv_im)
            elif target_['labels'].size(0) > 0:
                #print(target_['labels'].size(0), target_['labels'])
                #print({'boxes':bbs, 'labels': lls})
                targetn_.append({'boxes':target_['boxes'], 'labels': target_['labels']})
                stn_images.append(tc_images[idx])
                nimg_metas.append(img_metas[idx])
               
        
        if len(stn_images) > 0:
            stn_images = torch.stack(stn_images, dim=0)
            return stn_images, nimg_metas, targetn_
        else:
            return None, None, None
    
    def tranformer_target2(self, st_images, img_metas, targets_, vis=False, results_parameters=None):
        targetn_ = []
        if results_parameters is None:
            for idx in range(len(targets_)):
                target_ = targets_[idx]

                if target_['labels'].size(0) == 0:
                    targetn_.append(None)
                    continue
                
                bbs = target_['boxes'].clone().detach()
                lls = target_['labels'].clone().detach()
                
                if len(bbs.size()) == 2 and bbs.size(0) > 0:
                    targetn_.append({'boxes':bbs, 'labels': lls})
                
                if vis:#'RHF' in st_tran.keys():
                    lim = st_images[idx].permute(1,2,0).contiguous()
                    lim -= lim.min()
                    cv_im = lim.cpu().detach().numpy()
                    cv_im = np.array(cv_im*255, np.int32)
                    
                    mbbs = bbs.cpu().detach().numpy()
                    for ii in range(bbs.size(0)):
                        cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)
                    
                    #print('-----00:',lim.size())
                    cv2.imwrite('./test1/'+str(idx)+'.jpg', cv_im)
        else:
            for idx_num_iter in range(len(results_parameters)):
                ltargetn_ = []
                for idx in range(len(targets_[idx_num_iter])):
                    target_ = targets_[idx_num_iter][idx]
                    
                    if target_['labels'].size(0) == 0:
                        ltargetn_.append(None)
                        continue
                    
                    bbs = target_['boxes'].clone().detach()
                    lls = target_['labels'].clone().detach()
                    
                    if len(bbs.size()) == 2 and bbs.size(0) > 0:
                        ltargetn_.append({'boxes':bbs, 'labels': lls})
                    
                
                targetn_.append(ltargetn_)
               
        
        return st_images, img_metas, targetn_

            
    
    def generate_attention_maps(self, img_metas, oproposals):
        return
    
    ###rpn
    
    def generate_attention_pix_weights_v1(self, tc_feats, st_feats, obj_dets=None, stride=3):
        b, c, fh, fw = tc_feats.size()
        
        #mseloss = torch.nn.MSELoss(reduction='none')
        tc_feats_mask = tc_feats.abs().mean(dim=1)
        tc_feats_av = tc_feats_mask.mean(dim=(1,2), keepdim=True)
        tc_feats_mark = ((tc_feats_mask - tc_feats_av) > 0).float()

        return tc_feats_mark.detach()
    
    
    #v6-5
    def generate_attention_pix_weights_v2(self, tc_feats, st_feats, obj_dets=None, stride=3):
        b, c, fh, fw = tc_feats.size()
        
        #mseloss = torch.nn.MSELoss(reduction='none')
        
        class_num = 8
        ntc_feats = tc_feats.view(b, class_num, c // class_num, fh, fw).abs().mean(dim=2)
        ntc_feats_v, ntc_feats_i = ntc_feats.max(dim=1,keepdim=True)
        weights = torch.zeros_like(tc_feats.mean(dim=1))
        for idx in range(class_num):
            mask  = (ntc_feats_i == idx).float()
            mean_ftc = ((tc_feats.mul(mask)).sum(dim=(2,3))).div(mask.sum(dim=(2,3)) + 1e-5)
            
            feats_dist = (tc_feats - mean_ftc.view(b,c,1,1)).abs().mean(dim=1,keepdim=True)
            
            feats_dict_mn = ((feats_dist.mul(mask)).sum(dim=(2,3))).div(mask.sum(dim=(2,3)) + 1e-5)
            
            feats_mask = (feats_dist - 1.1 * feats_dict_mn.view(b,-1,1,1) > 0.0).float()
            
            #print((feats_mask.mul(mask)).squeeze(1).size())
            weights += (feats_mask.mul(mask)).squeeze(1)

        return (weights > 0.5).float()
    
    #v
    def generate_attention_pix_weights_v3(self, tc_feats, st_feats, obj_dets=None, stride=3):
        b, c, fh, fw = tc_feats.size()
        
        #mseloss = torch.nn.MSELoss(reduction='none')
        avg = nn.AvgPool2d((3,3), stride=1, padding=2)
        ntc_feats = tc_feats - avg(tc_feats)
        
        
        tc_feats_mask = ntc_feats.abs().mean(dim=1)
        tc_feats_av = tc_feats_mask.mean(dim=(1,2), keepdim=True)
        tc_feats_mark = ((tc_feats_mask - tc_feats_av) > 0).float()

        return tc_feats_mark.detach()
    
    def generate_attention_pix_weights_v4(self, tc_feats, st_feats, obj_dets=None, stride=3):
        b, c, fh, fw = tc_feats.size()
        
        #mseloss = torch.nn.MSELoss(reduction='none')
        avg = nn.AvgPool2d((5,5), stride=1, padding=2)
        ntc_feats1 = tc_feats - avg(tc_feats)
        
        ntc_feats2 = ntc_feats1 - avg(ntc_feats1)
        
        
        tc_feats_mask = ntc_feats2.abs().mean(dim=1)
        tc_feats_av = tc_feats_mask.mean(dim=(1,2), keepdim=True)
        tc_feats_mark = ((tc_feats_mask - tc_feats_av) > 0).float()

        return tc_feats_mark.detach()
    
    def generate_attention_pix_weights_v5(self, tc_feats, st_feats, obj_dets=None, stride=3):
        b, c, fh, fw = tc_feats.size()
        
        #mseloss = torch.nn.MSELoss(reduction='none')
        avg_3 = nn.AvgPool2d((3,3), stride=1, padding=1)
        avg_7 = nn.AvgPool2d((7,7), stride=1, padding=3)
        ntc_feats = avg_3(tc_feats) - avg_7(tc_feats)
        
        
        tc_feats_mask = ntc_feats.abs().mean(dim=1)
        tc_feats_av = tc_feats_mask.mean(dim=(1,2), keepdim=True)
        #print(tc_feats_av)
        tc_feats_mark = ((tc_feats_mask - 1.1 * tc_feats_av) > 0).float()

        return tc_feats_mark.detach()
    
    def generate_attention_pix_weights_v6(self, tc_feats, st_feats, obj_dets=None, stride=3):
        b, c, fh, fw = tc_feats.size()
        
        #mseloss = torch.nn.MSELoss(reduction='none')
        avg_3 = nn.AvgPool2d((3,3), stride=1, padding=1)
        avg_7 = nn.AvgPool2d((7,7), stride=1, padding=3)
        ntc_feats = avg_3(tc_feats) - avg_7(tc_feats)
        
        
        tc_feats_mask = ntc_feats.abs().mean(dim=1)
        tc_feats_av = tc_feats_mask.mean(dim=(1,2), keepdim=True)
        #print(tc_feats_av)
        #tc_feats_mark = ((tc_feats_mask - 1.1 * tc_feats_av) > 0).float()
        
        tc_feats_mx, _ = tc_feats_mask.view(b,-1).max(dim=1)
        tc_feats_mi, _ = tc_feats_mask.view(b,-1).min(dim=1)
        
        tc_feats_step = (tc_feats_mx - tc_feats_mi) / 3
        tc_feats_mark = tc_feats_mask.div(tc_feats_step.view(b,1,1)+1e-5)

        return tc_feats_mark.detach().long()
    
    def generate_attention_pix_weights_v7(self, tc_feats, st_feats, obj_dets=None, stride=3):
        B, C, H, W = tc_feats.size()
        
        avg_3_8 = nn.AvgPool2d((3,3), stride=1, padding=1)
        #avg_7_8 = nn.AvgPool2d((7,7), stride=1, padding=3)
        avg_7_8 = nn.AvgPool2d((5,5), stride=1, padding=2)
        max_7_8 = nn.MaxPool2d((7,7), stride=1, padding=1)
            
        with torch.no_grad():
            lb, lc, lh, lw = tc_feats.size()
            avg_pixels = avg_7_8(tc_feats)
            avg_mask = (tc_feats - avg_pixels).abs().mean(dim=1)
            
            avg_mask = avg_mask.div(avg_pixels.abs().mean(dim=1)+1e-5)
            mx = avg_mask.max()
            mi = avg_mask.min()
            avg_mask = (avg_mask - mi) / (mx- mi)
            
            #454
            #npos_weights = ((avg_mask - 1.6 * avg_mask.mean(dim=(1,2), keepdim=True)) > 0.0).float()
            
            #
            avg_mn_mask = avg_mask.mean(dim=(1,2), keepdim=True)
            npos_weights_0 = ((avg_mask - 1.6 * avg_mn_mask) > 0.0).float()
            
            npos_weights_1 = (((avg_mask - 1.3 * avg_mn_mask) > 0.0) & ((avg_mask - 1.6 * avg_mn_mask) <= 0.0)).float()
            
            npos_weights_2 = ((avg_mask - 1.3 * avg_mn_mask) <= 0.0).float()
            
            npos_weights = 0 * npos_weights_2 + 1 * npos_weights_1 + 2 * npos_weights_0
        
            '''
            idx = 0
            #npos_weights = 1 - npos_weights
            lnpos_weights = F.interpolate(npos_weights.unsqueeze(1), size=(self.images_pd.size(2), self.images_pd.size(3)), mode='bilinear')
            #lim = 1.0 * self.images_pd[idx:idx+1].mul(1.0 + 0.3 * lnpos_weights) #+ 0.3 * self.images_pd[idx:idx+1].mul(1-lnpos_weights)
            lim = 0.8 * self.images_pd[idx:idx+1].mul(lnpos_weights)
            lim = lim[0].permute(1,2,0).contiguous()
            lim -= lim.min()
            cv_im = lim.cpu().detach().numpy()
            cv_im = np.array(cv_im*255, np.int32)
            
            #print('-----00:',lim.size())
            cv2.imwrite('./attention_test_2.jpg', cv_im)
            
            
            lim = 0.8 * self.image_src[idx:idx+1].mul(lnpos_weights)
            lim = lim[0].permute(1,2,0).contiguous()
            lim -= lim.min()
            cv_im = lim.cpu().detach().numpy()
            cv_im = np.array(cv_im*255, np.int32)
            
            #print('-----00:',lim.size())
            cv2.imwrite('./attention_test_3.jpg', cv_im)
            '''
            
        return (npos_weights+0.1).long()
                
    
    
    def generate_attention_install_weights(self, feats, obj_dets, stride=3):
        n, d = feats.size()
        
        bbs = obj_dets['boxes']#torch.stack(dets['boxes'], dim=0)
        lls = obj_dets['labels']#torch.stack(dets['labels'], dim=0)
        scores = obj_dets['scores']#torch.stack(dets['scores'], dim=0)
        
        
        weights_areas = (lls > 0).float()
        weights_areas = 1 - torch.exp(-0.5 * (1 - weights_areas))
        weights_final = weights_areas.detach()
        
        return feats, weights_final
    
    def reduce_bias(self, tc_feat_set, st_feat_set):
        b, c, fh, fw = tc_feat_set.size()
        avg = nn.AvgPool2d((9,9), stride=1, padding=4)
        
        ltc_feat_set = tc_feat_set - avg(tc_feat_set)
        lst_feat_set = st_feat_set - avg(st_feat_set)
        
        return ltc_feat_set, lst_feat_set
        
    def cos_loss(self, ltc_feat_set, lst_feat_set):
        b, c, fh, fw = ltc_feat_set.size()
        
        fz = ltc_feat_set.mul(lst_feat_set)
        fm1 = ltc_feat_set.mul(ltc_feat_set)
        fm2 = lst_feat_set.mul(lst_feat_set)
        
        fz = fz.sum(dim=1)
        fm1 = (fm1.sum(dim=1)+1e-5).sqrt()
        fm2 = (fm2.sum(dim=1)+1e-5).sqrt()
        
        loss = 1 - fz.div(fm1.mul(fm2) + 1e-5)
        
        return loss.mean()
    
    ##region
    def split_map_to_patch(self, feats, window_size=8):
        bn, dm, fh, fw = feats.size()
        avg = nn.AvgPool2d((11,11), stride=1, padding=4)
        
        lfeats = feats# - avg(feats)
        
        
        #compute padding
        snh = fh // window_size
        sn_pad_h = fh - snh * window_size
        sn_pad_th = sn_pad_h // 2
        sn_pad_bh = sn_pad_h -  sn_pad_th
        
        sh_start_h = sn_pad_th
        sh_end_h = sn_pad_th + snh * window_size
        
        
        snw = fw // window_size
        sn_pad_w = fw - snw * window_size
        sn_pad_lw = sn_pad_w // 2
        sn_pad_rw = sn_pad_w -  sn_pad_lw
        
        sh_start_w = sn_pad_lw
        sh_end_w = sn_pad_lw + snw * window_size
        
        
        #crop tensor by removing the pad
        nfeats = lfeats[:,:,sh_start_h:sh_end_h, sh_start_w:sh_end_w]
        
        #divide patches
        _, _, fh_p, fw_p = nfeats.size()
        nfeats = nfeats.view(bn, dm, window_size, snh, window_size, snw)
        nfeats = nfeats.permute(0, 1, 2, 4, 3, 5).contiguous().view(bn, dm, window_size*window_size, snh*snw)
        return nfeats
    
    def get_vital_points(self, tc_feats, st_feats, window_size=8):
        bn, dm, fh, fw = tc_feats.size()
        
        #one stage
        tc_feats = self.split_map_to_patch(tc_feats, window_size= window_size)
        st_feats = self.split_map_to_patch(st_feats, window_size= window_size)

        #select sharp points
        l2_tc_nfeats = (tc_feats ** 2).mean(dim=1)
        l2_st_nfeats = (st_feats ** 2).mean(dim=1)
        tk_vl_tc0, tk_id_tc0 = torch.topk(l2_tc_nfeats, 1, dim=2)

        ntk_id_tc0 = tk_id_tc0.unsqueeze(1).expand(bn, dm, window_size*window_size, 1)
        tc_ft_tc0 = torch.gather(tc_feats, 3, ntk_id_tc0).squeeze(3)
        st_ft_tc0 = torch.gather(st_feats, 3, ntk_id_tc0).squeeze(3)
        tc_ft_tc0 = tc_ft_tc0.view(bn, dm, window_size, window_size)
        st_ft_tc0 = st_ft_tc0.view(bn, dm, window_size, window_size)
        
        return tc_ft_tc0, st_ft_tc0
    
    def get_vital_points_v2(self, tc_feats, st_feats, window_size=8):
        bn, dm, fh, fw = tc_feats.size()
        
        #one stage
        tc_feats = self.split_map_to_patch(tc_feats, window_size= window_size)
        st_feats = self.split_map_to_patch(st_feats, window_size= window_size)

        #select sharp points
        '''
        l2_tc_nfeats = (tc_feats ** 2).mean(dim=1)
        l2_st_nfeats = (st_feats ** 2).mean(dim=1)
        tk_vl_tc0, tk_id_tc0 = torch.topk(l2_tc_nfeats, 1, dim=2)

        ntk_id_tc0 = tk_id_tc0.unsqueeze(1).expand(bn, dm, window_size*window_size, 1)
        tc_ft_tc0 = torch.gather(tc_feats, 3, ntk_id_tc0).squeeze(3)
        st_ft_tc0 = torch.gather(st_feats, 3, ntk_id_tc0).squeeze(3)
        '''
        
        tc_ft_tc0 = tc_feats.mean(dim=3)
        st_ft_tc0 = st_feats.mean(dim=3)
        tc_ft_tc0 = tc_ft_tc0.view(bn, dm, window_size, window_size)
        st_ft_tc0 = st_ft_tc0.view(bn, dm, window_size, window_size)
        
        return tc_ft_tc0, st_ft_tc0

    def generate_region_features(self, tc_feats, st_feats, window_size=8):
        bn, dm, fh, fw = tc_feats.size()
        
        #one stage
        window_size0 = window_size*2
        tc_ft_tc0, st_ft_tc0 = self.get_vital_points(tc_feats, st_feats, window_size0)

        
        #two stage
        tc_ft_tc1, st_ft_tc1 = self.get_vital_points(tc_ft_tc0, st_ft_tc0, window_size)

        tc_ft_tc1 = tc_ft_tc1.view(bn, dm, -1).permute(0, 2, 1).contiguous()
        st_ft_tc1 = st_ft_tc1.view(bn, dm, -1).permute(0, 2, 1).contiguous()
        
        return tc_ft_tc1, st_ft_tc1
    
    
    def matrix_pixel_loss(self, tc_feats, st_feats, obj_dets=None, stride=3):
        bn, nb, dm = tc_feats.size()
        tc_feats = tc_feats.view(-1, dm)
        st_feats = st_feats.view(-1, dm)
        
        if obj_dets is not None:
            bbss = obj_dets['boxes']
            lls = obj_dets['labels']
            scores = obj_dets['scores']
            bbs = torch.gather(bbss, 1, lls.view(n,1,1).expand(n,1,4))
            bbs = bbs.squeeze(1)
            proposals = obj_dets['proposals']
        
        
        #compute similary matirx
        n_split = bn
        d_split = bn // n_split
        simi_mtx_tfz = []
        #print(n_split, d_split, n)
        for idx in range(n_split):
            start_id = idx * d_split
            if idx == n_split - 1:
                end_id = bn
            else:
                end_id = (idx+1) * d_split
            
            dist_rt = self.dist_fucntion(tc_feats[start_id:end_id], tc_feats)
            simi_mtx_tfz.append(dist_rt)
            
        simi_mtx_tfz = torch.cat(simi_mtx_tfz, dim=0)
        
        
        simi_mtx_fz = []
        for idx in range(n_split):
            start_id = idx * d_split
            if idx == n_split - 1:
                end_id = bn
            else:
                end_id = (idx+1) * d_split
            
            dist_rt = self.dist_fucntion(st_feats[start_id:end_id], tc_feats)
            simi_mtx_fz.append(dist_rt)
            
        simi_mtx_fz = torch.cat(simi_mtx_fz, dim=0)

        with torch.no_grad():
            ll_eye_mtx = torch.eye(simi_mtx_fz.size(0), simi_mtx_fz.size(1)).to(simi_mtx_fz.device)
            simi_mtx_mk = simi_mtx_fz.mul(ll_eye_mtx) + ll_eye_mtx - 1.0
            simi_vl_mk, simi_id_mk =  simi_mtx_mk.max(dim=1, keepdim=True)

            ll_pos_mtx_jl = ((simi_mtx_fz - 1.5 * simi_vl_mk) <= 0.0).float()
            
            '''
            pos_mask = (lls > 0).float()
            ll_pos_mtx_f = pos_mask[:,None].mul(ll_pos_mtx) + ll_pos_mtx_jl.mul(1 - pos_mask[:,None])
            '''
            
            ll_pos_mtx = ll_pos_mtx_jl
            ll_neg_mtx = ((simi_mtx_fz - 2.5 * simi_vl_mk) > 0.0).float()
            ll_pos2_mtx = ll_pos_mtx.mul(1 - ll_eye_mtx)
            
        #print(ll_pos_mtx.sum(dim=1).min())
        pos_mean_dist = (ll_pos_mtx.mul(simi_mtx_fz).sum(dim=1)).div(ll_pos_mtx.sum(dim=1))
        pos_mean_dist = pos_mean_dist.unsqueeze(1).expand(simi_mtx_fz.size())
        
        simi_mtx_prob = F.softmax(-0.1 * torch.stack([3*pos_mean_dist, simi_mtx_fz], dim=2), dim=2)
        simi_mtx_probl = -torch.log(simi_mtx_prob[:,:,0] + 1e-4)
        
        loss = (simi_mtx_probl.mul(ll_neg_mtx).sum(dim=1)).div(ll_neg_mtx.sum(dim=1)+1)
        
        return loss.mean()
        
    def matrix_pixel_loss_v1(self, tc_feats, st_feats, obj_dets=None, stride=3):
        nb, dm = tc_feats.size()
        
        #compute similary matirx
        

        with torch.no_grad():
            ll_eye_mtx = torch.eye(simi_mtx_fz.size(0), simi_mtx_fz.size(1)).to(simi_mtx_fz.device)
            simi_mtx_mk = simi_mtx_fz.mul(ll_eye_mtx) + ll_eye_mtx - 1.0
            simi_vl_mk, simi_id_mk =  simi_mtx_mk.max(dim=1, keepdim=True)

            ll_pos_mtx_jl = ((simi_mtx_fz - 1.5 * simi_vl_mk) <= 0.0).float()
            
            '''
            pos_mask = (lls > 0).float()
            ll_pos_mtx_f = pos_mask[:,None].mul(ll_pos_mtx) + ll_pos_mtx_jl.mul(1 - pos_mask[:,None])
            '''
            
            ll_pos_mtx = ll_pos_mtx_jl
            ll_neg_mtx = ((simi_mtx_fz - 2.5 * simi_vl_mk) > 0.0).float()
            ll_pos2_mtx = ll_pos_mtx.mul(1 - ll_eye_mtx)
            
        #print(ll_pos_mtx.sum(dim=1).min())
        pos_mean_dist = (ll_pos_mtx.mul(simi_mtx_fz).sum(dim=1)).div(ll_pos_mtx.sum(dim=1))
        pos_mean_dist = pos_mean_dist.unsqueeze(1).expand(simi_mtx_fz.size())
        
        simi_mtx_prob = F.softmax(-0.1 * torch.stack([3*pos_mean_dist, simi_mtx_fz], dim=2), dim=2)
        simi_mtx_probl = -torch.log(simi_mtx_prob[:,:,0] + 1e-4)
        
        loss = (simi_mtx_probl.mul(ll_neg_mtx).sum(dim=1)).div(ll_neg_mtx.sum(dim=1)+1)
        
        return loss.mean()
    
    def matrix_pixel_loss_v2(self, tc_feats, st_feats, cls_weights, class_c=8, layer_id=0):
        n, d = tc_feats.size()
        
        if tc_feats.size(0) == 0:
            return None
        
        
        #select samples with labels
        ##pos samples
        lls = cls_weights
        pos_mask = (lls > 0).float()
        ll_pos_mtx = (lls[:,None] - lls[None,:]).long()
        ll_pos_mtx = (ll_pos_mtx == 0).float()
        
        #fg,bg samples
        ll_fg_mtx = pos_mask.unsqueeze(0)
        ll_bg_mtx = 1 - ll_fg_mtx
        
        #update center
        bg_queue = []
        fg_queue = []
        fg_index_queue = []
        for idx in range(class_c+1):
            if idx == 0:
                continue
            else:
                mask = cls_weights == idx
                cls_feats = tc_feats[mask, :]
                if len(cls_feats.size()) == 2 and cls_feats.size(0) > 0:
                     #cls_feats = cls_feats.mean(dim=0)
                     
                     cls_feats_st2tc = (cls_feats.mul(cls_feats).sum(dim=1)+1e-10).sqrt()
                     cls_feats = cls_feats.div(cls_feats_st2tc.unsqueeze(1) + 1e-10)
                     
                     #27
                     '''
                     cls_feats_v, cls_feats_i = cls_feats.abs().max(dim=0,keepdim=True)
                     cls_feats_mask = ((cls_feats.abs() - cls_feats_v) > 1e-8).float()
                     
                     ncls_feats = (cls_feats.mul(cls_feats_mask).sum(dim=0)).div(cls_feats_mask.sum(dim=0)+1e-10)
                     '''
                     
                     ncls_feats = cls_feats.mean(dim=0)
                     
                     fg_queue.append(ncls_feats)
                     fg_index_queue.append(0*cls_feats.mean() + idx)
        
        if len(fg_queue) == 0:
            return None
        
        fg_queue = torch.stack(fg_queue, dim=0)
        fg_index_queue = torch.stack(fg_index_queue, dim=0)
        #print(fg_queue.size())
        
        #simi_mtx_fz_0 = self.similar_l2(st_feats, tc_feats, n, 2)
        #simi_mtx_fz_1 = self.similar_l2(st_feats, fg_queue, n, 2)
        simi_mtx_fz_0 = self.similar_dc(st_feats, tc_feats)
        simi_mtx_fz_1 = self.similar_dc(st_feats, fg_queue, mode=0)
        
        #select bg samples
        simi_mtx_fz_bg = simi_mtx_fz_0.mul(ll_bg_mtx) - 2 * (1 - ll_bg_mtx)
        simi_mtx_fz_bg_v, simi_mtx_fz_bg_i = torch.topk(simi_mtx_fz_bg, 2, 1)
        
        #pos loss
        ll_pos_mtx = (lls[:, None] - fg_index_queue[None, :]).long()
        ll_pos_mtx = (ll_pos_mtx == 0).float()
        
        #print(ll_pos_mtx.sum(dim=1))
        pos_attn_0 = simi_mtx_fz_1.mul(ll_pos_mtx).sum(dim=1)
        pos_attn = 1 - pos_attn_0
        
        #neg loss
        fg_attn = self.contrast_loss(pos_attn_0.unsqueeze(1), simi_mtx_fz_1, 1-ll_pos_mtx)
        bg_attn = self.contrast_loss(pos_attn_0.unsqueeze(1), simi_mtx_fz_bg_v)
        
        #loss_attn = 0.01 * pos_attn + 1.0 * fg_attn + 0.1 * bg_attn
        loss_attn = 1.0 * fg_attn + 0.1 * bg_attn
        
        
        #pos_mean_dist = pos_mean_dist.unsqueeze(1).expand(simi_mtx_fz.size())
        
        neg_mask = 1 - pos_mask
        pos_mask = pos_mask
        pos_loss = (loss_attn.mul(pos_mask).sum()).div(pos_mask.sum()+1e-5)
        neg_loss = (loss_attn.mul(neg_mask).sum()).div(neg_mask.sum()+1e-5)
        loss = pos_loss + 0.0 * neg_loss
        
        #print('--0:', class_c)
        
        return loss
    
    def matrix_inst_loss_v3(self, tc_feats, st_feats, obj_dets, stride=3, class_c=8):
        n, d = tc_feats.size()
        
        bbss = obj_dets['boxes']
        lls = obj_dets['labels']
        scores = obj_dets['scores']
        bbs = torch.gather(bbss, 1, lls.view(n,1,1).expand(n,1,4))
        bbs = bbs.squeeze(1)
        proposals = obj_dets['proposals']
        
        
        #compute similary matirx
        #l2
        #simi_mtx_fz = self.similar_dc(tc_feats, st_feats)
        simi_mtx_fz = self.similar_dc(st_feats, tc_feats)
        
        #select samples with boxes
        with torch.no_grad():
            ##gt iou
            gt_iou_ww = proposals[:,2] - proposals[:,0]
            gt_iou_hh = proposals[:,3] - proposals[:,1]
            gt_iou_ww[gt_iou_ww<0] = 0
            gt_iou_hh[gt_iou_hh<0] = 0
            
            gt_max_ww = torch.where((gt_iou_ww[:, None]-gt_iou_ww[None, :])>0, gt_iou_ww[:, None], gt_iou_ww[None, :])
            gt_min_ww = torch.where((gt_iou_ww[:, None]-gt_iou_ww[None, :])<0, gt_iou_ww[:, None], gt_iou_ww[None, :])
            
            gt_max_hh = torch.where((gt_iou_hh[:, None]-gt_iou_hh[None, :])>0, gt_iou_hh[:, None], gt_iou_hh[None, :])
            gt_min_hh = torch.where((gt_iou_hh[:, None]-gt_iou_hh[None, :])<0, gt_iou_hh[:, None], gt_iou_hh[None, :])
            
            gt_max = gt_max_ww.mul(gt_max_hh)
            gt_min = gt_min_ww.mul(gt_min_hh)
            gt_iou_weights = gt_min.div(gt_max+1e-5)
            gt_iou_weights = gt_iou_weights.clamp(min=0.0, max=1.0)
            
            #box weights
            areas_com_x0 = torch.where((proposals[:, 0]-bbs[:, 0])<0, bbs[:, 0], proposals[:, 0])
            areas_com_x1 = torch.where((proposals[:, 2]-bbs[:, 2])<0, proposals[:, 2], bbs[:, 2])
            
            areas_com_y0 = torch.where((proposals[:, 1]-bbs[:, 1])<0, bbs[:, 1], proposals[:, 1])
            areas_com_y1 = torch.where((proposals[:, 3]-bbs[:, 3])<0, proposals[:, 3], bbs[:, 3])
            areas_comm_ww = areas_com_x1 - areas_com_x0
            areas_comm_hh = areas_com_y1 - areas_com_y0
            areas_comm_ww[areas_comm_ww<0] = 0
            areas_comm_hh[areas_comm_hh<0] = 0
            
            areas_com = areas_comm_ww * areas_comm_hh
    
    
            ##max iou
            areas_mx_x0 = torch.where((proposals[:, 0]-bbs[:, 0])>0, bbs[:, 0], proposals[:, 0])
            areas_mx_x1 = torch.where((proposals[:, 2]-bbs[:, 2])>0, proposals[:, 2], bbs[:, 2])
            areas_mx_y0 = torch.where((proposals[:, 1]-bbs[:, 1])>0, bbs[:, 1], proposals[:, 1])
            areas_mx_y1 = torch.where((proposals[:, 3]-bbs[:, 3])>0, proposals[:, 3], bbs[:, 3])
            areas_mx_ww = areas_mx_x1 - areas_mx_x0
            areas_mx_hh = areas_mx_y1 - areas_mx_y0
            areas_mx_ww[areas_mx_ww<0] = 0
            areas_mx_hh[areas_mx_hh<0] = 0
            areas_mx = areas_mx_ww * areas_mx_hh
            
            areas_iou = areas_com.div(areas_mx+1e-5)
            areas_iou = areas_iou.clamp(min=0.0, max=1.0)
            areas_iou_mx = torch.where((areas_iou[:,None]-areas_iou[None,:])<0, areas_iou[None,:], areas_iou[:,None])
            simi_iou = (areas_iou[:,None]+areas_iou[None,:]).div(2 * areas_iou_mx + 1e-5)
            
            bb_weights_vl = gt_iou_weights.mul(simi_iou)
            bb_weights_mask = (gt_iou_weights > 0.55).float()
            
        #select samples with labels
        ##pos samples
        ll_pos_mtx = (lls[:,None] - lls[None,:]).long()
        ll_pos_mtx = (ll_pos_mtx == 0).float()
        
        
        #select samples with dbscan
        with torch.no_grad():
            ll_eye_mtx = torch.eye(ll_pos_mtx.size(0), ll_pos_mtx.size(1)).to(ll_pos_mtx.device)
            simi_mtx_mk = simi_mtx_fz.mul(ll_eye_mtx) + ll_eye_mtx - 1.0
            simi_vl_mk, simi_id_mk =  simi_mtx_mk.max(dim=1, keepdim=True)

            ll_pos_mtx_jl = ((simi_mtx_fz - 1.5 * simi_vl_mk) < 0.0).float()
            ll_pos_mtx_jl = ll_pos_mtx_jl.mul(ll_pos_mtx)
            
            pos_mask = (lls > 0).float()
            ll_pos_mtx_f = ll_pos_mtx#pos_mask[:,None].mul(ll_pos_mtx) + ll_pos_mtx_jl.mul(1 - pos_mask[:,None])
            
        
            ll_neg_mtx = 1 - ll_pos_mtx
            ll_eye_mtx = torch.eye(ll_pos_mtx.size(0), ll_pos_mtx.size(1)).to(ll_pos_mtx.device)
            ll_pos2_mtx = ll_pos_mtx.mul(1 - ll_eye_mtx)
            
            ll_pos_mtx_f = ll_pos_mtx_f.mul(bb_weights_mask)
                
        
        #v2
        '''
        t = 1.0 #/ math.sqrt(d * 1.0)
        #loss = 1 - (ll_pos_mtx_f.mul(simi_mtx_fz).sum(dim=1)).div(ll_pos_mtx_f.sum(dim=1)+1e-5)
        loss_l0 = 1 - (ll_pos_mtx_f.mul(simi_mtx_fz).sum(dim=1)).div(ll_pos_mtx_f.sum(dim=1)+1e-5)
        #loss_l1 = (ll_neg_mtx.mul(simi_mtx_fz).sum(dim=1)).div(ll_neg_mtx.sum(dim=1)+1e-5)
        nsimi_mtx_fz = (simi_mtx_fz - 0.8 * (1 - loss_l0).unsqueeze(1).detach()).clamp(min=0.0)
        loss_l1 = (ll_neg_mtx.mul(nsimi_mtx_fz).sum(dim=1)).div(nsimi_mtx_fz.sum(dim=1)+1e-5)
        loss =  loss_l0 + 0.0 * loss_l1.clamp(min=0.0)
        '''
        
        #v1
        #pos_mean_dist = (ll_pos_mtx_f.mul(simi_mtx_fz).sum(dim=1)).div(ll_pos_mtx_f.sum(dim=1)+1e-5)
        #pos_mean_dist = pos_mean_dist.unsqueeze(1).expand(simi_mtx_fz.size())
        
        #t = 1.0 #/ math.sqrt(d * 1.0)
        #simi_mtx_prob = F.softmax(t * torch.stack([pos_mean_dist, simi_mtx_fz], dim=2), dim=2)
        #simi_mtx_probl = -torch.log(simi_mtx_prob[:,:,0] + 1e-4)
        
        #loss = (simi_mtx_probl.mul(ll_neg_mtx).sum(dim=1)).div(ll_neg_mtx.sum(dim=1)+1e-5)
        
        #v3
        pos_mark = (ll_pos_mtx_f.sum(dim=1) > 0.5).float()
        
        pos_mask = (lls > 0).float()
        
        #fg,bg samples
        ll_fg_mtx = pos_mask.unsqueeze(0)
        ll_bg_mtx = 1 - ll_fg_mtx
        
        #update center
        bg_queue = []
        fg_queue = []
        fg_index_queue = []
        for idx in range(class_c+1):
            if idx == 0:
                continue
            else:
                mask = lls == idx
                cls_feats = tc_feats[mask, :]
                if len(cls_feats.size()) == 2 and cls_feats.size(0) > 0:
                     #cls_feats = cls_feats.mean(dim=0)
                     
                     cls_feats_st2tc = (cls_feats.mul(cls_feats).sum(dim=1)+1e-10).sqrt()
                     cls_feats = cls_feats.div(cls_feats_st2tc.unsqueeze(1) + 1e-10)
                     
                     ncls_feats = cls_feats.mean(dim=0)
                     
                     fg_queue.append(ncls_feats)
                     fg_index_queue.append(0*cls_feats.mean() + idx)
        
        if len(fg_queue) == 0:
            return None
        
        fg_queue = torch.stack(fg_queue, dim=0)
        fg_index_queue = torch.stack(fg_index_queue, dim=0)
        #print(fg_queue.size())
        
        #simi_mtx_fz_0 = self.similar_l2(st_feats, tc_feats, n, 2)
        #simi_mtx_fz_1 = self.similar_l2(st_feats, fg_queue, n, 2)
        simi_mtx_fz_0 = self.similar_dc(st_feats, tc_feats)
        simi_mtx_fz_1 = self.similar_dc(st_feats, fg_queue, mode=0)
        
        #select bg samples
        simi_mtx_fz_bg = simi_mtx_fz_0.mul(ll_bg_mtx) - 2 * (1 - ll_bg_mtx)
        simi_mtx_fz_bg_v, simi_mtx_fz_bg_i = torch.topk(simi_mtx_fz_bg, 2, 1)
        
        #pos loss
        ll_pos_mtx = (lls[:, None] - fg_index_queue[None, :]).long()
        ll_pos_mtx = (ll_pos_mtx == 0).float()
        
        #print(ll_pos_mtx.sum(dim=1))
        pos_attn_0 = simi_mtx_fz_1.mul(ll_pos_mtx).sum(dim=1)
        pos_attn = 1 - pos_attn_0
        
        #neg loss
        fg_attn = self.contrast_loss(pos_attn_0.unsqueeze(1), simi_mtx_fz_1, 1-ll_pos_mtx)
        bg_attn = self.contrast_loss(pos_attn_0.unsqueeze(1), simi_mtx_fz_bg_v)
        
        loss = 1.0 * fg_attn + 0.1 * bg_attn
        
        
        neg_mask = (1 - pos_mask).mul(pos_mark)
        pos_mask = pos_mask.mul(pos_mark)
        pos_loss = (loss.mul(pos_mask).sum()).div(pos_mask.sum()+1e-5)
        neg_loss = (loss.mul(neg_mask).sum()).div(neg_mask.sum()+1e-5)
        loss = pos_loss + 0.0 * neg_loss
        
        #print('--1:', class_c)
        
        return loss
    
    def contrast_loss(self, pos_samples, neg_samples, neg_sample_mask=None):
        #v0
        '''
        loss = (neg_samples - pos_samples + 0.1).clamp(min=0.0)
        if neg_sample_mask is not None:
            loss = (loss.mul(neg_sample_mask).sum(dim=1)).div(neg_sample_mask.sum(dim=1) + 1e-5)
        else:
            loss = loss.mean(dim=1)
        
        return loss
        '''
        
        #v1       
        probs_a =  torch.exp(pos_samples * 1.0)
        probs_b =  torch.exp(neg_samples * 1.0)
        
        probs = probs_a.div((probs_a + 2 * probs_b).clamp(min=1e-6))
        loss = - torch.log(probs + 1e-6)
        
        if neg_sample_mask is not None:
            loss = (loss.mul(neg_sample_mask).sum(dim=1)).div(neg_sample_mask.sum(dim=1) + 1e-5)
        else:
            loss = loss.mean(dim=1)
        
        return loss
    
    
    def instance_extractor(self, feats, layer_idx=0, momentum=False):
        if momentum:
             instance_net = getattr(self, 'tc_instance_net'+str(layer_idx+2))
        else:
             instance_net = getattr(self, 'instance_net'+str(layer_idx+2))
        feats = instance_net(feats)
        return feats
        
         
    #output vector
    def tranformer_out_vectors(self, tc_vectors, st_vectors, st_trans, cls_weights=None):
        mseloss = torch.nn.MSELoss(reduction='none')
        loss_ins_out = 0.0
        neg_weight = 0.02
        
        if tc_vectors.size(1) == self.class_num:
               lb, lc = tc_vectors.size()
               
               lloss_l2 = 1.0 * mseloss(tc_vectors, st_vectors)
               lloss_l2 = lloss_l2.mean(dim=1)
               lls = cls_weights
               l1_mask = (lls > 0).float()
               lloss_l10 = (lloss_l2.mul(l1_mask).sum()).div(l1_mask.sum() + 1e-5)
               lloss_l11 = (lloss_l2.mul(1-l1_mask).sum()).div((1-l1_mask).sum() + 1e-5)
               lloss_l1 = lloss_l10 + self.neg_weight * lloss_l11
               lloss_ins_l2 = lloss_l1.mean()
               loss_ins_out += lloss_ins_l2
        else:
               if tc_vectors.size(1) == self.class_num * 4:
                   ln = len(st_trans)
                   lb, lc = tc_vectors.size()
                   ntc_vectors = tc_vectors.view(ln, -1, self.class_num, 4)
                   
                   ftc_vectors = []
                   for idx in range(ln):
                       ltc_bbs = ntc_vectors[idx]
                       st_tran = st_trans[idx]
                       with torch.no_grad():
                           if 'RVF' in st_tran.keys():
                                ltc_bbs[:, :, 1] *= -1
                           if 'RHF' in st_tran.keys():
                                ltc_bbs[:, :, 0] *= -1
                       ftc_vectors.append(ltc_bbs)
                  
                   ftc_vectors = torch.stack(ftc_vectors, dim=0).view(-1, lc)
               else:
                   ftc_vectors = tc_vectors
                    
               lloss_l2 = 1.0 * mseloss(ftc_vectors, st_vectors)
               lloss_l2 = lloss_l2.mean(dim=1)
               lls = cls_weights
               l1_mask = (lls > 0).float()
               lloss_l10 = (lloss_l2.mul(l1_mask).sum()).div(l1_mask.sum() + 1e-5)
               lloss_l11 = (lloss_l2.mul(1-l1_mask).sum()).div((1-l1_mask).sum() + 1e-5)
               lloss_l1 = lloss_l10 + self.neg_weight * lloss_l11
               lloss_ins_l2 = lloss_l1.mean()
               loss_ins_out += lloss_ins_l2
               
        return loss_ins_out
    
    #output map
    def tranformer_out_maps(self, feats_tc, feats_st, st_trans, cls_weights=None, scopes_sz=16):
        B, C, H, W = feats_st.size()
        mseloss = torch.nn.MSELoss(reduction='none')
        loss_ins_out = 0.0
        neg_weight = 0.1
        for idx in range(len(st_trans)):
            st_tran = st_trans[idx]
            lfeats_tc = feats_tc[idx:idx+1]
            lfeats_st = feats_st[idx:idx+1]
            
            if cls_weights is not None:
                 lcls_weights = (cls_weights[idx:(idx+1)] > 0.5).float()
            else:
                 lcls_weights = torch.ones_like(lfeats_tc[:,0,:,:]).float()
            
            im_weights = 1.0
            
            if 'RandomSized' in st_tran.keys():
                if st_tran['RandomSized'][0] > 1.0:
                    scale_factor = st_tran['RandomSized'][0]
                else:
                    scale_factor = 1.0 / st_tran['RandomSized'][0]
                    
                if scale_factor > 1.5:
                    im_weights = 0.0
                
            
            if lfeats_tc.size(1) == self.anchor_num * 4:
                if 'RVF' in st_tran.keys():
                    lfeats_st = lfeats_st.view(1, -1, 4, H, W)
                    y0 = st_tran['RVF'][1] - lfeats_st[:, :, 3]
                    y1 = st_tran['RVF'][1] - lfeats_st[:, :, 1]
                    lfeats_st[:, :, 1] = y0
                    lfeats_st[:, :, 3] = y1
                    lfeats_st = lfeats_st.view(1, -1, H, W)
                    
                if 'RHF' in st_tran.keys():
                    lfeats_st = lfeats_st.view(1, -1, 4, H, W)
                    x0 = st_tran['RHF'][0] - lfeats_st[:, :, 2]
                    x1 = st_tran['RHF'][0] - lfeats_st[:, :, 0]
                    lfeats_st[:, :, 0] = x0
                    lfeats_st[:, :, 2] = x1
                    lfeats_st = lfeats_st.view(1, -1, H, W)
                    
                ##boxes
                with torch.no_grad():
                    lfeats_tc = lfeats_tc.view(1, -1, 4, H, W)
                    if 'RandomSized' in st_tran.keys():
                         lfeats_tc[:, :, 0] *= st_tran['RandomSized'][0]
                         lfeats_tc[:, :, 1] *= st_tran['RandomSized'][1]
                         lfeats_tc[:, :, 2] *= st_tran['RandomSized'][0]
                         lfeats_tc[:, :, 3] *= st_tran['RandomSized'][1]
                    
                    if 'CenterCrop' in st_tran.keys():
                         x00 = lfeats_tc[:, :, 0] - st_tran['CenterCrop'][0]
                         y00 = lfeats_tc[:, :, 1] - st_tran['CenterCrop'][1]
                         x01 = lfeats_tc[:, :, 2] - st_tran['CenterCrop'][0]
                         y01 = lfeats_tc[:, :, 3] - st_tran['CenterCrop'][1]
                         
                         lfeats_tc = torch.stack((x00, y00, x01, y01), dim=2)
                    lfeats_tc = lfeats_tc.view(1, -1, H, W)
                    
            else:
                if 'RVF' in st_tran.keys():
                    lfeats_st = lfeats_st.flip([1])
                    
                if 'RHF' in st_tran.keys():
                    lfeats_st = lfeats_st.flip([2])
                    
            #scale
            if 'RandomSized' in st_tran.keys():
                lh, lw = int(H / st_tran['RandomSized'][1]), int(W / st_tran['RandomSized'][0])
                lfeats_st = F.interpolate(lfeats_st, size=(lh, lw), mode='bilinear')
                
                if lh > H or lw > W:
                     pad_h = int((lh - H)/2.0)
                     pad_w = int((lw - W)/2.0)
                     if pad_h < 0:
                         pad_h = 0
                     if pad_w < 0:
                         pad_w = 0
                    
                     mlfeats_st = lfeats_st[:, :, pad_h:(pad_h+H), pad_w:(pad_w+W)]
                     mlfeats_tc = lfeats_tc
                     mlcls_weights = lcls_weights
                      
                else:
                     pad_h =  int((H - lh)/2.0)
                     pad_w = int((W - lw)/2.0)
                     if pad_h < 0:
                          pad_h = 0
                     if pad_w < 0:
                          pad_w = 0
                     
                     mlfeats_tc = lfeats_tc[:, :, pad_h:(pad_h+lh), pad_w:(pad_w+lw)]
                     mlcls_weights = lcls_weights[:, pad_h:(pad_h+lh), pad_w:(pad_w+lw)]
                     mlfeats_st = lfeats_st
                     
            else:
                mlfeats_tc = lfeats_tc
                mlfeats_st = lfeats_st
                mlcls_weights = lcls_weights
            
            if 'Translate' in st_tran.keys():
                mb, mc, mh, mw = mlfeats_tc.size()
                x0 = 0
                y0 = 0
                x1 = mw
                y1 = mh
                x10 = 0
                y10 = 0
                if st_tran['Translate'][2] < 0:
                     lx = int(st_tran['Translate'][2] / scopes_sz)
                     x1 = x1 + lx
                     x10 = -lx
                else:
                     lx = int(st_tran['Translate'][2] / scopes_sz)
                     x0 = x0 + lx
                     
                if st_tran['Translate'][3] < 0:
                     ly = int(st_tran['Translate'][3] / scopes_sz)
                     y1 = y1 + ly
                     y10 = -ly
                else:
                     ly = int(st_tran['Translate'][3] / scopes_sz)
                     y0 = y0 + ly
                
                nlfeats_tc = mlfeats_tc[:,:, y0:y1, x0:x1]
                nlfeats_st = mlfeats_st[:,:, y10:y10+(y1-y0), x10:x10+(x1-x0)]
                nlcls_weights = mlcls_weights[:, y0:y1, x0:x1]
            else:
                nlfeats_tc = mlfeats_tc
                nlfeats_st = mlfeats_st
                nlcls_weights = mlcls_weights
            
            '''
            npos_weights = (nlcls_weights > 0.5).float()
            lnpos_weights = F.interpolate(npos_weights.unsqueeze(1), size=(self.images_pd.size(2), self.images_pd.size(3)), mode='bilinear',align_corners=False)
            lim = self.images_src[idx:idx+1]
            lim -= lim.min()
            lim = lim.mul(lnpos_weights)
            lim = lim[0].permute(1,2,0).contiguous()
            cv_im = lim.cpu().detach().numpy()
            cv_im = np.array(cv_im*255/2, np.int32)
            cv2.imwrite('./out_attn.jpg', cv_im)
            '''
            
            if lfeats_tc.size(1) == self.anchor_num * 4:
                el_pixels_tc, el_pixels_st = self.uniformization(nlfeats_tc, nlfeats_st)
                #print(el_pixels_tc.size(), el_pixels_st.size(), self.anchor_num)
                
            else:
                el_pixels_st = nlfeats_st# - avg_7_8(nlfeats_st)
                el_pixels_tc = nlfeats_tc# - avg_7_8(nlfeats_tc)
            
                
            lloss_l2 = mseloss(el_pixels_st, el_pixels_tc.detach()).mean(dim=1)

            lloss_l10 = (lloss_l2.mul(nlcls_weights).sum()).div(nlcls_weights.sum() + 1e-5)
            lloss_l11 = (lloss_l2.mul(1-nlcls_weights).sum()).div((1-nlcls_weights).sum() + 1e-5)
            lloss_l1 = lloss_l10 + self.neg_weight * lloss_l11
            lloss_ins_l2 = lloss_l1.mean()
            loss_ins_out += im_weights * lloss_ins_l2
        
        return loss_ins_out / B
    
    def align_consisence(self, tc_feat_set, st_feat_set, attn_maps, st_trans, dets=None):
        mseloss = torch.nn.MSELoss(reduction='none')
        
        #output map
        loss_pix_out = 0.0
        for idx in range(len(tc_feat_set['output_maps'])):
             tc_pixel_feats = tc_feat_set['output_maps'][idx].clone().detach()
             st_pixel_feats = st_feat_set['output_maps'][idx]
             
             lloss_pix_out = self.tranformer_out_maps(tc_pixel_feats, st_pixel_feats, st_trans, dets['weights']['output_maps'][idx])
             
             #print(tc_pixel_feats.size())
             #36: 0.4
             #9: 0.7
             #if tc_pixel_feats.size(1) == 9:
             #    print(lloss_pix_out)
             loss_pix_out += lloss_pix_out
        
        #output vector
        loss_ins_out = 0.0
        for idx in range(len(tc_feat_set['output_vectors'])):
             tc_inst_feats = tc_feat_set['output_vectors'][idx].clone().detach()
             st_inst_feats = st_feat_set['output_vectors'][idx]

             loss_ins_out += self.tranformer_out_vectors(tc_inst_feats, st_inst_feats, st_trans, dets['labels'])
             
             #8:0.6
             #2:0.2
             #print(tc_inst_feats.size())
             #if tc_inst_feats.size(1) == 2:
             #    print(loss_ins_out)
    
        ##feature map
        loss_pix_l2 = 0.0
        loss_pix_mt = 0.0
        loss_pix_edge = 0.0
        for idx in range(len(tc_feat_set['feature_map'])):
             tc_pixel_feats = tc_feat_set['feature_map'][idx].clone().detach()
             st_pixel_feats = st_feat_set['feature_map'][idx]
             cls_weights = dets['weights']['feature_map'][idx]
             layer_rois = self.teacher.ada_layers_rois[idx]
             scopes_sz = self.teacher.ada_layers_stride[idx]
             
             norm = torch.nn.InstanceNorm2d(tc_pixel_feats.size(1), affine=False)
             tc_pixel_feats = norm(tc_pixel_feats)
             st_pixel_feats = norm(st_pixel_feats)
             #loss_l2 = mseloss(tc_pixel_feats, st_pixel_feats).mean(dim=1)
             
             with torch.no_grad():
                  l2_mask = self.generate_attention_pix_weights_v7(tc_pixel_feats, st_pixel_feats)
            
             loss_l2, loss_id = self.tranformer_feats(tc_pixel_feats, st_pixel_feats, st_trans, cls_weights=cls_weights, txt_weights=l2_mask, loss_txt_weight=[0.0, 0.1, 1.0], scopes_sz=scopes_sz, region_roi_masks=layer_rois, class_c=self.class_num)
             
             loss_pix_l2 += loss_l2 / len(tc_feat_set['feature_map'])
             loss_pix_mt += loss_id / len(tc_feat_set['feature_map'])
             
             #edge with vital
             #loss_pix_edge += self.matrix_edge_loss(tc_ft_tc, st_ft_tc)
        
        ## feature vector
        loss_ins_l2 = 0.0
        loss_ins_mt = 0.0
        loss_ins_edge = 0.0
        norm1d = torch.nn.InstanceNorm1d(tc_inst_feats.size(1), affine=False)
        for idx in range(len(tc_feat_set['feature_vector'])):
            tc_inst_feats = tc_feat_set['feature_vector'][idx].clone().detach()
            st_inst_feats = st_feat_set['feature_vector'][idx]
            ln, lc = tc_inst_feats.size()
            
            if st_inst_feats.size(0) % len(st_trans) != 0:
                print(tc_inst_feats.size(), len(st_trans))
                
            tc_inst_feats = tc_inst_feats.view(len(st_trans), -1, lc)
            st_inst_feats = st_inst_feats.view(len(st_trans), -1, lc)
            
            tc_inst_feats = tc_inst_feats.permute(0,2,1).contiguous()
            st_inst_feats = st_inst_feats.permute(0,2,1).contiguous()
            
            tc_inst_feats = norm1d(tc_inst_feats)
            st_inst_feats = norm1d(st_inst_feats)
            
            tc_inst_feats = tc_inst_feats.permute(0,2,1).contiguous().view(-1, lc)
            st_inst_feats = st_inst_feats.permute(0,2,1).contiguous().view(-1, lc)
            
            #l2
            lloss_l2 = 1.0 * mseloss(tc_inst_feats, st_inst_feats)
            lloss_l2 = lloss_l2.mean(dim=1)
            
            #dc
            #lloss_l2 = 1.0 - self.similar_vec_dc(tc_inst_feats, st_inst_feats)
            
            lls = dets['labels']
            l1_mask = (lls > 0).float()
            lloss_l10 = (lloss_l2.mul(l1_mask).sum()).div(l1_mask.sum() + 1e-5)
            lloss_l11 = (lloss_l2.mul(1-l1_mask).sum()).div((1-l1_mask).sum() + 1e-5)
            lloss_l1 = lloss_l10 + 0.01 * lloss_l11
            lloss_ins_l2 = lloss_l1.mean()
            
            loss_ins_l2 += lloss_ins_l2 / len(tc_feat_set['feature_vector'])
            
            
            lloss_ins_mtx = self.matrix_inst_loss_v3(tc_inst_feats, st_inst_feats, dets, class_c=self.class_num)
            loss_ins_mt += lloss_ins_mtx / len(tc_feat_set['feature_vector'])
            
            #loss_ins_edge += self.matrix_edge_loss(tc_inst_feats, st_inst_feats)
        
        '''
        loss_pix = 1.0 * loss_pix_l2 + 0.1 * loss_pix_mt
        loss_ins = 1.0 * loss_ins_l2 + 0.1 * loss_ins_mt
        loss_output = 2.0 * loss_pix_out + loss_ins_out
        loss =  1.0 * loss_pix + 1.0 * loss_ins + 1.0 * loss_output
        #print(loss_pix_l2, loss_ins_l2, loss_pix_mt, loss_ins_mt)
        return loss * 0.2
        '''
        
        #test
        #10-00_06:49.3
        '''
        loss_pix = 4.0 * loss_pix_l2 + 0.0 * loss_pix_mt
        loss_ins = 4.0 * loss_ins_l2 + 0.0 * loss_ins_mt
        loss_output = 5.0 * loss_pix_out + loss_ins_out
        loss =  1.0 * loss_pix + 0.0 * loss_ins + 0.0 * loss_output
        return loss * 2
        '''
        
        #10-00_06_01
        '''
        loss_pix = 4.0 * loss_pix_l2 + 0.0 * loss_pix_mt
        loss_ins = 4.0 * loss_ins_l2 + 0.0 * loss_ins_mt
        loss_output = 5.0 * loss_pix_out + loss_ins_out
        loss =  1.0 * loss_pix + 1.0 * loss_ins + 0.0 * loss_output
        return loss * 2
        '''
        
        
        #10-00_06_02
        '''
        loss_pix = 4.0 * loss_pix_l2 + 0.0 * loss_pix_mt
        loss_ins = 4.0 * loss_ins_l2 + 0.0 * loss_ins_mt
        loss_output = 5.0 * loss_pix_out + loss_ins_out
        loss =  1.0 * loss_pix + 0.0 * loss_ins + 1.0 * loss_output
        return loss * 2
        '''
        
        #10-00_06_05:0.1
        #10-00_06_06:0.01
        '''
        loss_pix = 4.0 * loss_pix_l2 + 0.0 * loss_pix_mt
        loss_ins = 4.0 * loss_ins_l2 + 0.0 * loss_ins_mt
        loss_output = 5.0 * loss_pix_out + loss_ins_out
        loss =  1.0 * loss_pix + 0.01 * loss_ins + 1.0 * loss_output
        return loss * 2
        '''
        
        #10-00_06_07_31: 1.0-1.0-0.0, (4.0, 0.0), (0.1, 0.1) - (50.2)
        #10-00_06_07_32: 1.0-1.0-0.0, (4.0, 0.0), (0.1, 0.1), (5.0, 1.0) - (50.2)
        '''
        loss_pix = 4.0 * loss_pix_l2 + 0.1 * loss_pix_mt
        loss_ins = 0.0 * loss_ins_l2 + 0.1 * loss_ins_mt
        loss_output = 5.0 * loss_pix_out + loss_ins_out
        loss =  1.0 * loss_pix + 1.0 * loss_ins + 1.0 * loss_output
        return loss * 2
        '''
        
        #10-00_06_07_32: 1.0-1.0-0.0, (4.0, 0.0), (0.0, 0.0), (5.0, 1.0) - (50.2)
        loss_pix = 4.0 * loss_pix_l2 + 0.0 * loss_pix_mt
        loss_ins = 0.0 * loss_ins_l2 + 0.0 * loss_ins_mt
        loss_output = 5.0 * loss_pix_out + loss_ins_out
        loss =  1.0 * loss_pix + 1.0 * loss_ins + 1.0 * loss_output
        return loss * 2
        
        
    def forward_perturbation(self, images):
        trf_dict = {}
        #images = self.transforms(images)
        
        brightness = 0.1 + random.random() * 0.5
        contrast = 0.1 + random.random() * 0.5
        saturation = 0.1 + random.random() * 0.5
        hue = 0.1 + random.random() * 0.5

        transforms = T.Compose([
         T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        ])
        #print(images.size())
        images = transforms(images)
        
        ##
        index = random.randint(0,2)
        images[:, index, :, :] *= 0
        
        ###flip
        if random.random() < 0.8:
            images = images.flip([3])
            trf_dict['wflip'] = True

        if random.random() < 0.8:
            images = images.flip([2])
            trf_dict['hflip'] = True
        return images, trf_dict

    def backward_perturbation(self, images, trfm=None):
        if trfm is None:
            return images

        if 'wflip' in trfm.keys():
             images = images.flip([3])

        if 'hflip' in trfm.keys():
             images = images.flip([2])

        return images
        
    def update_teacher(self, NET_MOMENTUM=None):
        if NET_MOMENTUM is None:
            NET_MOMENTUM = self.NET_MOMENTUM
            
        #for param_diff, param_mn in zip(self.student.parameters(), self.teacher.parameters()):
        #    param_mn.data = param_mn.data.clone().detach() * NET_MOMENTUM + param_diff.data.clone().detach() * (1. - NET_MOMENTUM)
        
        for param_diff, param_mn in zip(self.student.backbone.parameters(), self.teacher.backbone.parameters()):
            param_mn.data = param_mn.data.clone().detach() * NET_MOMENTUM + param_diff.data.clone().detach() * (1. - NET_MOMENTUM)
        
        for param_diff, param_mn in zip(self.student.rpn.parameters(), self.teacher.rpn.parameters()):
            param_mn.data = param_mn.data.clone().detach() * NET_MOMENTUM + param_diff.data.clone().detach() * (1. - NET_MOMENTUM)
        
        for param_diff, param_mn in zip(self.student.box_head.parameters(), self.teacher.box_head.parameters()):
            param_mn.data = param_mn.data.clone().detach() * NET_MOMENTUM + param_diff.data.clone().detach() * (1. - NET_MOMENTUM)
        
        return
    
    
    def update_student(self, NET_MOMENTUM=None):
        if NET_MOMENTUM is None:
            NET_MOMENTUM = self.NET_MOMENTUM
            
        for param_diff, param_mn in zip(self.teacher.parameters(), self.student.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        '''

        for param_diff, param_mn in zip(self.teacher.backbone.parameters(), self.eval_net.backbone.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        for param_diff, param_mn in zip(self.teacher.rpn.parameters(), self.eval_net.rpn.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        for param_diff, param_mn in zip(self.teacher.box_head.parameters(), self.eval_net.box_head.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        for param_diff, param_mn in zip(self.teacher.box_head.parameters(), self.eval_net.eval_head.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        
        if self.use_eval_head:
            print('----init conv memory')
            self.eval_net.eval_head.box_predictor.init_memory_conv()
        '''

        return
    
    def update_eval(self, NET_MOMENTUM=None):
        if NET_MOMENTUM is None:
            NET_MOMENTUM = self.NET_MOMENTUM
            
        for param_diff, param_mn in zip(self.teacher.backbone.parameters(), self.eval_net.backbone.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        for param_diff, param_mn in zip(self.teacher.rpn.parameters(), self.eval_net.rpn.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        for param_diff, param_mn in zip(self.teacher.box_head.parameters(), self.eval_net.box_head.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
            
        for param_diff, param_mn in zip(self.teacher.box_head.parameters(), self.eval_net.eval_train_head.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
            
        
        if self.use_eval_head:
            print('----init conv memory')
            self.eval_net.eval_train_head.box_predictor.init_memory_conv()
            #self.teacher.eval_eval_head.box_predictor.init_memory_conv()
        

        return
    
    def update_eval_head(self, NET_MOMENTUM=None):
        if NET_MOMENTUM is None:
            NET_MOMENTUM = self.NET_MOMENTUM
        
        for param_diff, param_mn in zip(self.teacher.eval_train_head.box_predictor.parameters(), self.teacher.eval_eval_head.box_predictor.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM

        return
    
    def init_eval_head(self):
        #self.teacher.eval_train_head.box_predictor.init_memory_conv()
        #self.student.eval_train_head.box_predictor.init_memory_conv()
        self.eval_net.eval_train_head.box_predictor.init_memory_conv()
        return 
    
    
    #m001/vgg_8_9
    def update_teacher_eval(self, NET_MOMENTUM=None):
        self.student.eval_head._initialize_weights()
        return
    
    def deepcopy_evalhead(self,):
        if self.copy_eval_head is None:
            with torch.no_grad():
                #self.student.eval_head._initialize_weights()
                copy_eval_head = []
                for param_diff in self.student.box_head.parameters():
                     copy_eval_head.append(param_diff.data.clone().cpu().detach().numpy())
                #self.copy_eval_head =  copy.deepcopy(self.student.eval_head)

                self.copy_eval_head =  copy_eval_head
    
    def restore_evalhead(self,NET_MOMENTUM=0.0, mtype=0, init_index=0):
        '''
        for name, params in  self.student.eval_head.named_parameters():
            params.data.copy_(params.data * 0.0 + 0.000001)

            if params.grad is not None:
                params.grad.data.copy_(params.grad.data * 0.0)
        '''

        if mtype == 0:
            self.eval_net.eval_head.box_predictor.init_memory_conv()
        elif mtype == 1:
            self.teacher.eval_head._initialize_weights()
        else:
            self.student.eval_head._initialize_weights()
        
        return
    
    def romve_evalhead(self,):
        del self.copy_eval_head
        self.copy_eval_head = None
    

    def save_old_thresh_parameters(self, mtype=1):
        if mtype == 0:
            self.eval_net.box_head.save_old_thresh_parameters()
        elif mtype == 1:
            self.teacher.box_head.save_old_thresh_parameters()
        else:
            self.student.box_head.save_old_thresh_parameters()
    
    def restore_thresh_parameters(self, mtype=1):
        if mtype == 0:
            self.eval_net.box_head.restore_thresh_parameters()
        elif mtype == 1:
            self.teacher.box_head.restore_thresh_parameters()
        else:
            self.student.box_head.restore_thresh_parameters()
        
    def update_instNet(self):
        NET_MOMENTUM = 0.9
        for idx in range(1):
            tc_instance_net = getattr(self, 'tc_instance_net'+str(idx+2))
            st_instance_net = getattr(self, 'instance_net'+str(idx+2))
            for param_diff, param_mn in zip(st_instance_net.parameters(), tc_instance_net.parameters()):
                param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
    
        return
    
    def frozen_net(self):
        print('---------: teacher frozen')
        for name, param in self.teacher.backbone.named_parameters():
            param.requires_grad = False
        
        for name, param in self.teacher.rpn.named_parameters():
            param.requires_grad = False
        
        return 
    
    #def generate_style_data(self, images, domain):
    ###similar matrix fucntion
    def dist_fucntion(self, tc_feats, st_feats):
        simi_mtx_tfz = (tc_feats.unsqueeze(1) - st_feats.unsqueeze(0)) ** 2
        simi_mtx_tfz = simi_mtx_tfz.mean(dim=2)
        
        return simi_mtx_tfz
        
    def similar_l2(self, tc_feats, st_feats, num_total, n_split):
        d_split = num_total // n_split
        
        simi_mtx_tfz = []
        for idx in range(n_split):
            start_id = idx * d_split
            if idx == n_split - 1:
                end_id = num_total
            else:
                end_id = (idx+1) * d_split
            
            dist_rt = self.dist_fucntion(tc_feats[start_id:end_id], st_feats)
            simi_mtx_tfz.append(dist_rt)
            
        simi_mtx_tfz = torch.cat(simi_mtx_tfz, dim=0)
        
        return simi_mtx_tfz
    
    def similar_cos(self, f1_feats, f2_feats):
        simi_mtx_st2tc_fz = torch.mm(f1_feats, f2_feats.permute(1,0).contiguous())
        st_dist_st2tc = (f1_feats.mul(f1_feats).sum(dim=1)+1e-5).sqrt()
        tc_dist_st2tc = (f2_feats.mul(f2_feats).sum(dim=1)+1e-5).sqrt()
        simi_mtx_st2tc_fm = st_dist_st2tc[:,None].mul(tc_dist_st2tc[None,:])
        simi_mtx_st2tc = simi_mtx_st2tc_fz.div(simi_mtx_st2tc_fm + 1e-5)
        return simi_mtx_tfz
    
    def similar_dc(self, f1_feats, f2_feats, mode=1):
        f1_dist_st2tc = (f1_feats.mul(f1_feats).sum(dim=1)+1e-10).sqrt()
        lf1_feats = f1_feats.div(f1_dist_st2tc.unsqueeze(1) + 1e-10)
        
        if mode == 1:
            f2_dist_st2tc = (f2_feats.mul(f2_feats).sum(dim=1)+1e-10).sqrt()
            lf2_feats = f2_feats.div(f2_dist_st2tc.unsqueeze(1) + 1e-10)
        else:
            lf2_feats = f2_feats
        
        simi_mtx_tfz = torch.mm(lf1_feats, lf2_feats.permute(1,0).contiguous())
        return simi_mtx_tfz
    
    
    def similar_vec_l2(self, tc_feats, st_feats):
        simi_mtx_tfz = (tc_feats - st_feats) ** 2
        simi_mtx_tfz = simi_mtx_tfz.mean(dim=1)
        
        return simi_mtx_tfz
    
    def similar_vec_cos(self, f1_feats, f2_feats):
        simi_mtx_st2tc_fz = f1_feats,mul(f2_feats).sum(dim=1)
        st_dist_st2tc = (f1_feats.mul(f1_feats).sum(dim=1)+1e-5).sqrt()
        tc_dist_st2tc = (f2_feats.mul(f2_feats).sum(dim=1)+1e-5).sqrt()
        simi_mtx_st2tc_fm = st_dist_st2tc.mul(tc_dist_st2tc)
        simi_mtx_st2tc = simi_mtx_st2tc_fz.div(simi_mtx_st2tc_fm + 1e-5)
        return simi_mtx_tfz
    
    def similar_vec_dc(self, f1_feats, f2_feats):
        f1_dist_st2tc = (f1_feats.mul(f1_feats).sum(dim=1)+1e-5).sqrt()
        f2_dist_st2tc = (f2_feats.mul(f2_feats).sum(dim=1)+1e-5).sqrt()
        lf1_feats = f1_feats.div(f1_dist_st2tc.unsqueeze(1) + 1e-5)
        lf2_feats = f2_feats.div(f2_dist_st2tc.unsqueeze(1) + 1e-5)
        
        simi_mtx_tfz = lf1_feats.mul(lf2_feats).sum(dim=1)
        return simi_mtx_tfz
    
    ###norm
    def uniformization(self, f1_feats, f2_feats):
        mx_feats = f1_feats.max(f2_feats)
        f1_feats = f1_feats.div(mx_feats + 1e-6)
        f2_feats = f2_feats.div(mx_feats + 1e-6)

        return f1_feats, f2_feats
    
    def uniformization(self, f1_feats, f2_feats):
        mx_feats = (f1_feats.abs()).max(f2_feats.abs())
        f1_feats = f1_feats.div(mx_feats + 1e-6)
        f2_feats = f2_feats.div(mx_feats + 1e-6)

        return f1_feats, f2_feats
    
    def set_thresh_parameters(self, results_parameters, mtype=1):
        if mtype == 0:
            #eval_net.box_head.set_thresh_parameters(results_parameters[idx])
            self.eval_net.box_head.set_thresh_parameters(results_parameters)
        elif mtype ==1:
            self.teacher.box_head.set_thresh_parameters(results_parameters)
        else:
            self.student.box_head.set_thresh_parameters(results_parameters)

