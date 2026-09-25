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
from detection.modeling.norm.SNorm import SNorm

#
from torch.utils.tensorboard import SummaryWriter
#from detection.modeling.norm.script import generate_rect_boxes, select_rect_proposals, select_best_feature, \
#generate_rect_boxes_v2, select_rect_proposals_v2, select_rect_weight_v2, select_best_weight, \
#select_rect_weight_v3, select_rect_weight_v4, scoring_loss, select_rect_weight_v5, class_loss, \
#select_rect_weight_v6, select_rect_weight_v7, select_rect_labels_v7


from detection.modeling.norm.script_bp import select_rect_weight_w_prob, class_loss_l2, class_loss, \
select_rect_weight_w_prob_1030, generate_moe_mask_v4, class_hl_loss, class_loss_v2

from detection.modeling.norm.dice_loss import DiceLoss
from detection.modeling.ModuleSet import ModuleSet
#from detection.modeling.unet.lib_render import SSIMLoss
from detection.modeling.unet.ssim import SSIM, mse_loss, edge_loss, tv_loss, patch_consistency_loss
from detection.modeling.unet.gen_loss import ImageGenerationLoss, Discriminator, nll_loss_manual
from torch.autograd import Variable


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
        #self.student = ModuleSet(cfg)
        #if cfg.MODEL.MODE.USE_STUDENT:
        #     self.student = ModuleSet(cfg)
        #     #self.eval_net = ModuleSet(cfg, use_eval_head=self.use_eval_head, eval_head_num=self.eval_head_num, use_memory_conv=True)
             
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
        
        self.l1_loss = nn.L1Loss()
        self.cross_loss = nn.CrossEntropyLoss()

        self.iter_mark = 0
        #self.iter_internal = 5 * 371
        self.use_moe_interval = 5 * 371
        self.use_moe_fusion_interval = 10 * 371
        self.use_moe_test = cfg.MODEL.MODE.USE_MOE_TEST
        self.use_moe_fusion_test = cfg.MODEL.MODE.USE_MOE_FUSION_TEST

        #memory
        self.memory_loss = []
        self.memory_loc = 0
        self.memory_num = 10

        #summarywriter
        writer_0 = SummaryWriter("runs/experiment_0")
        writer_1 = SummaryWriter("runs/experiment_1")
        writer_2 = SummaryWriter("runs/experiment_2")
        writer_3 = SummaryWriter("runs/experiment_3")
        writer_4 = SummaryWriter("runs/experiment_4")
        self.writer = [writer_0, writer_1, writer_2, writer_3, writer_4]

        #self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion = DiceLoss()
        self.draw_index = 0
        self.render_loss_61 = SSIM(data_range=1.0, win_size=61)
        self.render_loss_31 = SSIM(data_range=1.0, win_size=31)
        self.render_loss_15 = SSIM(data_range=1.0, win_size=15)
        self.render_loss_7 = SSIM(data_range=1.0, win_size=7)
        self.iter_training_save = 0
        
        
        ##################################
        self.image_gen_loss = ImageGenerationLoss()
        self.dis = Discriminator()
        

    def forward(self, tc_images, img_metas, targets_=None, domain='s', step_iter=0, stsc_images=None, stsc_trans=None, \
        stpd_images=None, stpd_trans=None, outputs_teacher=None, epoch=0, vis=False, \
        network_gated_mark_psudo=1, head_gated_mark_psudo=0, network_gated_mark_psudo_el=1, head_gated_mark_psudo_el=0, \
        network_gated_mark_train=2, head_gated_mark_train=0, network_gated_mark_train_el=0, head_gated_mark_train_el=1, \
        network_gated_mark_test=1, head_gated_mark_test=0, network_gated_mark_test_el=0, head_gated_mark_test_el=1, \
        is_eval_update=False, results_parameters=None, only_psudo_el=False, mode_training=0, is_update=False, \
        update_norm_p=False, moe_mode=0, moe_mark=0):
        '''
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        '''
        #output dict
        outputs = dict()
        loss_dict = dict()
        
        if self.training:
            #wheter to use use_moe and use_moe_fusion
            use_moe = False
            use_moe_fusion = False
            if self.iter_mark  > self.use_moe_fusion_interval:
                use_moe = True
                use_moe_fusion = True
            elif self.iter_mark > self.use_moe_interval:
                use_moe = True
            
            self.use_moe_test = use_moe
            self.use_moe_fusion_test = use_moe_fusion

            ########only detection########
            if mode_training == 0:
                if stsc_trans is not None:
                    tc_images, img_metas, targets_, _ = self.tranformer_target(tc_images, stsc_images, img_metas, targets_, stsc_trans, None, vis=False)

                output_teacher = self.teacher(tc_images, img_metas, targets_, mode=5, moe_mode=moe_mode, 
                                              update_norm_p=update_norm_p, is_update=is_update)
                
                for key, value in output_teacher['rpn_losses_w'].items():
                    loss_dict.update({key+'_w':value})
                for key, value in output_teacher['box_losses_w'].items():
                    loss_dict.update({key+'_w':value})

            ########detection + e-net########
            elif mode_training == 1:
                if stsc_trans is not None:
                    tc_images, img_metas, targets_, _ = self.tranformer_target(tc_images, stsc_images, img_metas, targets_, stsc_trans, None, vis=False)

                output_teacher = self.teacher(tc_images, img_metas, targets_, mode=5, moe_mode=moe_mode, 
                                              update_norm_p=update_norm_p, is_update=is_update)
                for key, value in output_teacher['rpn_losses_w'].items():
                    loss_dict.update({key+'_w':value})
                for key, value in output_teacher['box_losses_w'].items():
                    loss_dict.update({key+'_w':value})


                ###e-net##
                tc_wei_dict = output_teacher['feat_dict']['weight_dict']
                wei_len_num = len(tc_wei_dict)
                tc_feat_dict = output_teacher['feat_dict']['feature_dict'][0:wei_len_num]  
                g_stride = 1 #tc_images.size(-2) / tc_wei_dict[0].size(-2)
                
                for idx_wi in range(wei_len_num):
                    pseudo_targets = output_teacher['dets_w']#[idx_wi]
                    gate_mask = select_rect_weight_w_prob_1030(tc_images, targets_, pseudo_targets, iter_index=self.iter_mark, \
                                                        features=tc_images, stride=g_stride)
                    
                    gate_mask_final = gate_mask.unsqueeze(1)
                    
                    ###to scale
                    stride = int(tc_images.size(-2) / tc_wei_dict[idx_wi].size(-2))
                    kernel_size = 2 * stride + 1
                    padding = stride
                    
                    el_mask = (gate_mask > 0.6).float()
                    el_mask_sum = F.avg_pool2d(el_mask[:,None], kernel_size=kernel_size, stride=stride) * (kernel_size ** 2)
                    gate_mask_sum = F.avg_pool2d(gate_mask[:,None], kernel_size=kernel_size, stride=stride) * (kernel_size ** 2)
                    gate_mask_final = el_mask_sum #gate_mask_sum / (el_mask_sum + 1e-6)

                    gate_mask_final = (gate_mask_final > 2).float()
                    gate_mask_final = F.interpolate(gate_mask_final, size=(tc_wei_dict[idx_wi].size(-2), tc_wei_dict[idx_wi].size(-1)), mode='nearest')
                    l1_loss_c = class_loss_v2(tc_wei_dict[idx_wi], gate_mask_final)
                    

                    ##loss m2##
                    '''
                    gate_mask_final = F.interpolate(gate_mask_final, size=(tc_wei_dict[idx_wi].size(-2), tc_wei_dict[idx_wi].size(-1)), mode='bilinear')
                    l1_loss_c = class_hl_loss(tc_wei_dict[idx_wi], gate_mask_final)
                    '''
                    loss_dict.update({'l1_loss_c'+str(idx_wi): 1.0 * l1_loss_c})
                    #print(l1_loss_c)


                    ##draw##
                    import torchvision.utils as vutils
                    name = str(self.draw_index).zfill(4)    
                    #vutils.save_image(gate_mask_final.data, './samples/' + name +'.png', nrow=12, value_range=(-1, 1), normalize=True)
                    #vis_images_all = torch.zeros((1, gate_mask_final.size(1), 4 * gate_mask_final.size(-2), gate_mask_final.size(-1))).to(output.device)
                    #vis_images_all[:, :, 0:output.size(-2)] = save_im0[0:1]
                    #vis_images_all[:, :, output.size(-2):output.size(-2)*2] = feats_0[0:1]
                    #gate_mask_final_vis = torch.cat([gate_mask_final, tc_wei_dict[0]], dim=0)
                    vutils.save_image(gate_mask_final.data, './samples/' + name +'_o.png', nrow=10, value_range=(-1, 1), normalize=True)
                    vutils.save_image(tc_wei_dict[0].sigmoid().data, './samples/' + name +'_p.png', nrow=10, value_range=(-1, 1), normalize=True)
            
            ################e-net loss#####################
            elif mode_training == 8:
                if stsc_trans is not None:
                    tc_images, img_metas, targets_, _ = self.tranformer_target(tc_images, stsc_images, img_metas, targets_, stsc_trans, None, vis=False)

                output_teacher = self.teacher(tc_images, img_metas, targets_, mode=7, moe_mode=0, update_norm_p=False, is_update=is_update)

                ###
                tc_wei_dict = output_teacher['feat_dict']['weight_dict']
                wei_len_num = len(tc_wei_dict)
                tc_feat_dict = output_teacher['feat_dict']['feature_dict'][0:wei_len_num]  
                g_stride = 1 #tc_images.size(-2) / tc_wei_dict[0].size(-2)
                
                for idx_wi in range(wei_len_num):
                    ###
                    pseudo_targets = output_teacher['dets_w']#[idx_wi]
                    gate_mask = select_rect_weight_w_prob_1030(tc_images, targets_, pseudo_targets, iter_index=self.iter_mark, \
                                                        features=tc_images, stride=g_stride)
                    gate_mask_final = gate_mask.unsqueeze(1)
                    '''
                    ###to scale
                    stride = int(tc_images.size(-2) / tc_wei_dict[idx_wi].size(-2))
                    kernel_size = 2 * stride + 1
                    padding = stride
                    #print(stride, gate_mask.max())
                    
                    el_mask = (gate_mask > 0.6).float()
                    el_mask_sum = F.avg_pool2d(el_mask[:,None], kernel_size=kernel_size, stride=stride) * (kernel_size ** 2)
                    gate_mask_sum = F.avg_pool2d(gate_mask[:,None], kernel_size=kernel_size, stride=stride) * (kernel_size ** 2)
                    gate_mask_final = el_mask_sum #gate_mask_sum / (el_mask_sum + 1e-6)
                    
                    gate_mask_final = (gate_mask_final > 2).float()
                    gate_mask_final = F.interpolate(gate_mask_final, size=(tc_wei_dict[idx_wi].size(-2), tc_wei_dict[idx_wi].size(-1)), mode='nearest')
                    l1_loss_c = class_loss(tc_wei_dict[idx_wi], gate_mask_final)
                    '''
                    gate_mask_final = F.interpolate(gate_mask_final, size=(tc_wei_dict[idx_wi].size(-2), tc_wei_dict[idx_wi].size(-1)), mode='bilinear')
                    
                    l1_loss_c = class_hl_loss(tc_wei_dict[idx_wi], gate_mask_final)
                    loss_dict.update({'l1_loss_c'+str(idx_wi): 100 * l1_loss_c})

                    import torchvision.utils as vutils
                    
                    name = str(0).zfill(4)    
                    #gate_mask_final_vis = torch.cat([gate_mask_final, tc_wei_dict[0]], dim=0)
                    vutils.save_image(gate_mask_final.data, './samples/' + name +'_aug_o.png', nrow=10, value_range=(-1, 1), normalize=True)
                    vutils.save_image(tc_wei_dict[0].sigmoid().data, './samples/' + name +'_aug_p.png', nrow=10, value_range=(-1, 1), normalize=True)
                self.draw_index = (self.draw_index + 1) % 10
            
            #################render###################
            elif mode_training == 9:
                if step_iter < 1000:
                    with torch.no_grad():
                        self.teacher(tc_images, img_metas, targets_, mode=8, moe_mode=4, update_norm_p=True, is_update=is_update)
                #stpd_images
                output_teacher = self.teacher(stpd_images, img_metas, targets_, mode=8, moe_mode=4, update_norm_p=False, is_update=is_update)
                fg_mask = generate_moe_mask_v4([tc_images], targets_, stride=1.0)[0]
                #loss_rec = nn.MSELoss()(tc_images, output_teacher['features_norm'])
                fg_weights = (fg_mask > 0.5).float().unsqueeze(1)
                bg_weights = 1 - fg_weights
                #print(output_teacher['features_norm'])
                
                loss_v = mse_loss(tc_images, output_teacher['features_norm'], fg_weights=fg_weights)

                loss_edge = edge_loss(tc_images, output_teacher['features_norm'], fg_weights=fg_weights)
                #print('---1:', tc_images.size(), stpd_images.size(), output_teacher['features_norm'].size())
                
                loss_ssim = self.render_loss(tc_images, output_teacher['features_norm'], fg_weights)
                loss_rec = 10 * loss_v + 0.1 * loss_ssim +  0.1 * loss_edge
                #print(loss_v, loss_ssim, loss_edge)
                loss_dict.update({'loss_rec': 1.0 * loss_rec})
            
            
            #################ahdr render###################
            elif mode_training == 10:
                
                with torch.no_grad():
                       #output_teacher = self.teacher(tc_images, img_metas, targets_, mode=8, moe_mode=0, update_norm_p=False, is_update=is_update)
                       fg_mask = generate_moe_mask_v4([tc_images], targets_, stride=1.0)[0]
                       fg_weights = (fg_mask > 0.5).float().unsqueeze(1)
                       bg_weights = 1 - fg_weights
                        
                #stpd_images
                #output_teacher_aug = self.teacher(stpd_images, img_metas, targets_, mode=8, moe_mode=7, update_norm_p=False, is_update=is_update)
                output_teacher_aug = self.teacher(stpd_images, img_metas, targets_, mode=8, moe_mode=7, \
                                                  moe_mark=moe_mark, update_norm_p=is_update, is_update=is_update)
                
                target = tc_images #output_teacher['features_norm']
                output = output_teacher_aug['features_norm']
                target = target * 0.5 + 0.5
                #output = output * 0.5 + 0.5
                #print(target.max(), target.min())
                
                #output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
                #target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

                #loss_v = F.l1_loss(output, target)
                #print(loss_rec)
                #loss_rec = F.l1_loss(output, target)
                
                #rgb
                loss_v = (target - output) ** 2 / (output ** 2 + target ** 2 + 1e-5)
                loss_v_fg = (loss_v * fg_weights).sum() / (fg_weights.sum() + 1e-5)
                loss_v_bg = (loss_v * bg_weights).sum() / (bg_weights.sum() + 1e-5)
                loss_v = loss_v_fg + 0.0 * loss_v_bg
                
                #ssim
                loss_ssim_7 = self.render_loss_7(output, target, weights=fg_weights)
                loss_ssim_15 = self.render_loss_15(output, target, weights=fg_weights)
                loss_ssim_31 = self.render_loss_31(output, target, weights=fg_weights)
                loss_ssim_61 = self.render_loss_61(output, target, weights=fg_weights)
                #loss_ssim = (loss_ssim_7 + loss_ssim_15 + loss_ssim_31) / 3.0
                #output nan
                loss_ssim = (loss_ssim_7 + loss_ssim_15 + loss_ssim_31 + loss_ssim_61) / 4.0
                #loss_ssim = (loss_ssim_15 + loss_ssim_31 + loss_ssim_61) / 3.0
                #print(loss_ssim)
                
                #edge
                loss_e = edge_loss(output, target)
                #print(loss_v, loss_e)
                
                #smooth
                loss_m = tv_loss(output)
                
                ###
                loss_con = patch_consistency_loss(output)
                
                #402
                #loss_rec = loss_v + 0.1 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m
                
                #loss_rec = 1.0 * loss_v + 0.1 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m
                #print(loss_v, loss_ssim, loss_e, loss_m)
                
                #loss_rec = 0.1 * loss_v + 1.0 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m + 0.1 * loss_con
                
                #alb, x, 38.x
                #loss_rec = 1.0 * loss_v + 0.0 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m + 0.0 * loss_con
                
                loss_rec = 0.0 * loss_v + 1.0 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m + 0.01 * loss_con
                
                loss_dict.update({'loss_rec': 0.01 * loss_rec})
                
                
                ##################################
                '''
                loss_adv_true = self.image_gen_loss.gan(self.dis(target.detach()), 1.0, fg_weights)
                loss_adv_false = self.image_gen_loss.gan(self.dis(output.detach()), 0.0, fg_weights) 
                loss_gen = self.image_gen_loss(output, target, fg_weights)
                loss_rec = loss_gen + loss_adv_true + loss_adv_false
                #print(loss_rec)
                loss_dict.update({'loss_rec': 0.1 * loss_rec})
                '''
                ##################################
               
                loss_attn = 0.0
                
                for idx_attn in range(len(output_teacher_aug['attn'])):
                    x = output_teacher_aug['attn'][idx_attn]
                    y = F.interpolate(output_teacher_aug['attn_masks'], size=(x.size(-2), x.size(-1)), mode='bilinear')  
                    if y.size(1) > 1:
                        y = y[:, 0:1, :, :]
                    loss_attn_ = nll_loss_manual(x, y)
                    loss_attn += loss_attn_
                loss_dict.update({'loss_attn': 0.1 * loss_attn})
               
                ##################################
                ###draw
                self.iter_training_save = (self.iter_training_save + 1) % 2000
                if self.iter_training_save % 20 == 0:
                    vis_images_all = torch.zeros((1, output.size(1), 4 * output.size(-2), output.size(-1))).to(output.device)
                    vis_images_all[:, :, 0:output.size(-2)] = tc_images[0:1]
                    vis_images_all[:, :, output.size(-2):output.size(-2)*2] = (target[0:1] - 0.5) / 0.5 
                    vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = (output[0:1] - 0.5) / 0.5
                    #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output[0:1] - 0.5) / 0.5
                    #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output * mask)[0:1]

                    import torchvision.utils as vutils
                
                    name = str(self.iter_training_save).zfill(4)    
                    vutils.save_image(vis_images_all.data, './samples/loss/' + name +'.jpg', nrow=12, value_range=(-1, 1), normalize=True)
            
            elif mode_training == 11:
                with torch.no_grad():
                       output_teacher = self.teacher(tc_images, img_metas, targets_, mode=8, moe_mode=0, update_norm_p=False, is_update=is_update)
                       fg_mask = generate_moe_mask_v4([tc_images], targets_, stride=1.0)[0]
                       fg_weights = (fg_mask > 0.5).float().unsqueeze(1)
                       bg_weights = 1 - fg_weights
                        
                #stpd_images
                #output_teacher_aug = self.teacher(stpd_images, img_metas, targets_, mode=8, moe_mode=7, update_norm_p=False, is_update=is_update)
                output_teacher_aug = self.teacher(stpd_images, img_metas, targets_, mode=8, moe_mode=7, \
                                                  moe_mark=moe_mark, update_norm_p=False, is_update=is_update)
                
                target = output_teacher['features_norm']
                output = output_teacher_aug['features_norm']
                target = target * 0.5 + 0.5
                #output = output * 0.5 + 0.5
                #print(target.max(), target.min())
                
                #output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
                #target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

                #loss_v = F.l1_loss(output, target)
                #print(loss_rec)
                #loss_rec = F.l1_loss(output, target)
                
                #rgb
                loss_v = (target - output) ** 2 / (output ** 2 + target ** 2 + 1e-5)
                loss_v_fg = (loss_v * fg_weights).sum() / (fg_weights.sum() + 1e-5)
                loss_v_bg = (loss_v * bg_weights).sum() / (bg_weights.sum() + 1e-5)
                loss_v = loss_v_fg + 0.0 * loss_v_bg
                
                #ssim
                loss_ssim_7 = self.render_loss_7(output, target, weights=fg_weights)
                loss_ssim_15 = self.render_loss_15(output, target, weights=fg_weights)
                loss_ssim_31 = self.render_loss_31(output, target, weights=fg_weights)
                loss_ssim_61 = self.render_loss_61(output, target, weights=fg_weights)
                #loss_ssim = (loss_ssim_7 + loss_ssim_15 + loss_ssim_31) / 3.0
                #output nan
                loss_ssim = (loss_ssim_7 + loss_ssim_15 + loss_ssim_31 + loss_ssim_61) / 4.0
                #loss_ssim = (loss_ssim_15 + loss_ssim_31 + loss_ssim_61) / 3.0
                #print(loss_ssim)
                
                #edge
                loss_e = edge_loss(output, target)
                #print(loss_v, loss_e)
                
                #smooth
                loss_m = tv_loss(output)
                
                #402
                #loss_rec = loss_v + 0.1 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m
                
                #loss_rec = 1.0 * loss_v + 0.1 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m
                #print(loss_v, loss_ssim, loss_e, loss_m)
                
                loss_rec = 1.0 * loss_v + 1.0 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m
                
                loss_dict.update({'loss_rec': 0.01 * loss_rec})
                

            elif mode_training == 12:
                
                with torch.no_grad():
                       output_teacher = self.teacher(tc_images, img_metas, targets_, mode=8, moe_mode=0, update_norm_p=False, is_update=is_update)
                       fg_mask = generate_moe_mask_v4([tc_images], targets_, stride=1.0)[0]
                       fg_weights = (fg_mask > 0.5).float().unsqueeze(1)
                       bg_weights = 1 - fg_weights
                        
                #stpd_images
                output_teacher_aug = self.teacher(stpd_images, img_metas, targets_, mode=8, moe_mode=7, \
                                                  moe_mark=moe_mark, update_norm_p=False, is_update=is_update)
                
                target = output_teacher['features_norm']
                output = output_teacher_aug['features_norm']
                target = target * 0.5 + 0.5
                #output = output * 0.5 + 0.5
                #print(target.max(), target.min())
                
                #output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
                #target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

                #loss_v = F.l1_loss(output, target)
                #print(loss_rec)
                #loss_rec = F.l1_loss(output, target)
                '''
                
                #rgb
                loss_v = (target - output) ** 2 / (output ** 2 + target ** 2 + 1e-5)
                #loss_v = loss_v.mean()
                loss_v_fg = (loss_v * fg_weights).sum() / (fg_weights.sum() + 1e-5)
                loss_v_bg = (loss_v * bg_weights).sum() / (bg_weights.sum() + 1e-5)
                loss_v = loss_v_fg + loss_v_bg
                
                
                loss_ssim = self.render_loss(output, target, weights=fg_weights)
                
                #edge
                loss_e = edge_loss(output, target)
                #print(loss_v, loss_e)
                
                #smooth
                loss_m = tv_loss(output)
                
                loss_rec = loss_v + 0.1 * loss_ssim + 0.0 * loss_e + 0.0 * loss_m
                #print(loss_v, loss_ssim, loss_e, loss_m)
                
                loss_dict.update({'loss_rec': 0.1 * loss_rec})
                '''
                
                #self.dis(target)
                
                ###draw
                self.iter_training_save = (self.iter_training_save + 1) % 2000
                if self.iter_training_save % 40 == 0:
                    vis_images_all = torch.zeros((1, output.size(1), 4 * output.size(-2), output.size(-1))).to(output.device)
                    vis_images_all[:, :, 0:output.size(-2)] = tc_images[0:1]
                    vis_images_all[:, :, output.size(-2):output.size(-2)*2] = (target[0:1] - 0.5) / 0.5 
                    vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = (output[0:1] - 0.5) / 0.5 
                    #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output[0:1] - 0.5) / 0.5
                    #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output * mask)[0:1]
                    #tc_images

                    import torchvision.utils as vutils
                
                    name = str(self.iter_training_save).zfill(4)    
                    vutils.save_image(vis_images_all.data, './samples/loss/' + name +'.jpg', nrow=12, value_range=(-1, 1), normalize=True)
                
                
            return loss_dict, outputs_teacher
            

        else:
            self.use_moe_test = True
            self.use_moe_fusion_test = True

            if network_gated_mark_test == 0:
                outputs = self.eval_net(tc_images, img_metas, targets_, mode=1)
            elif network_gated_mark_test == 1:
                outputs = self.teacher(tc_images, img_metas, targets_, mode=1, moe_mode=moe_mode, moe_mark=moe_mark)
            else:
                outputs = self.student(tc_images, img_metas, targets_, mode=1)
            
            #tc_aug_wei_dict = outputs['feat_dict']['weight_dict']
            #self.save_img_v40(tc_images, tc_aug_wei_dict)

            return outputs['dets_w']
    
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
        sc_masks = []
        for idx in range(len(targets_)):
            target_ = targets_[idx]
            st_tran = st_trans[idx]
            #print(target_['labels'].size(0))

            if target_['labels'].size(0) == 0:
                continue
            
            bbs = target_['boxes'].clone().detach()
            lls = target_['labels'].clone().detach()
            image_box = torch.ones_like(bbs[0])
            image_box[0] = 0
            image_box[1] = 0
            image_box[2] = tc_images.size(-1)
            image_box[3] = tc_images.size(-2)
            
            min_wh = 5
            if 'RandomSized' in st_tran.keys():
                 bbs[:, 0] *= st_tran['RandomSized'][0]
                 bbs[:, 1] *= st_tran['RandomSized'][1]
                 bbs[:, 2] *= st_tran['RandomSized'][0]
                 bbs[:, 3] *= st_tran['RandomSized'][1]
                
                 image_box[0] *= st_tran['RandomSized'][0]
                 image_box[1] *= st_tran['RandomSized'][1]
                 image_box[2] *= st_tran['RandomSized'][0]
                 image_box[3] *= st_tran['RandomSized'][1]
            
            if 'CenterCrop' in st_tran.keys():
                 #
                 image_box[0] = image_box[0] - st_tran['CenterCrop'][0]
                 image_box[1] = image_box[1] - st_tran['CenterCrop'][1]
                 image_box[2] = image_box[2] - st_tran['CenterCrop'][0]
                 image_box[3] = image_box[3] - st_tran['CenterCrop'][1]
                 if image_box[0] < 0:
                    image_box[0] = 0
                
                 if image_box[1] < 0:
                    image_box[1] = 0
                    
                 if image_box[2] >= st_tran['CenterCrop'][4]:
                    image_box[2] = st_tran['CenterCrop'][4]
                 
                 if image_box[3] >= st_tran['CenterCrop'][5]:
                    image_box[3] = st_tran['CenterCrop'][5]
                           
                 #
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
            
            im_mask = torch.zeros_like(tc_images[idx,0])
            im_mask[int(image_box[1]):int(image_box[3]), int(image_box[0]):int(image_box[2])] = 255
            sc_masks.append(im_mask)
            
            if len(bbs.size()) == 2 and bbs.size(0) > 0:
               targetn_.append({'boxes':bbs, 'labels': lls})
               stn_images.append(st_images[idx])
               nimg_metas.append(img_metas[idx])
               if vis:#'RHF' in st_tran.keys():
                   #st_images[idx](
                   lim = st_images[idx].permute(1,2,0).contiguous()
                   lim -= lim.min()
                   cv_im = lim.cpu().detach().numpy()
                   cv_im = np.array(cv_im*255, np.int32)
                   
                   mbbs = bbs.cpu().detach().numpy()
                   for ii in range(bbs.size(0)):
                       cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)
                   
                   #print('-----00:',lim.size())
                   cv2.imwrite('./test/'+str(idx)+'.jpg', cv_im)
                   cv2.imwrite('./test/'+str(idx)+'_mask.jpg', im_mask.cpu().detach().numpy())
            elif target_['labels'].size(0) > 0:
                #print(target_['labels'].size(0), target_['labels'])
                #print({'boxes':bbs, 'labels': lls})
                targetn_.append({'boxes':target_['boxes'], 'labels': target_['labels']})
                stn_images.append(tc_images[idx])
                nimg_metas.append(img_metas[idx])
            
               
        
        if len(stn_images) > 0:
            stn_images = torch.stack(stn_images, dim=0)
            sc_masks = torch.stack(sc_masks, dim=0)
            return stn_images, nimg_metas, targetn_, sc_masks
        else:
            return None, None, None
    
    def select_stands(self, x_naug, x_aug):
        bn, bd, bh, bw = x_naug[0].size()
        x_naug_os = torch.stack(x_naug, dim=1)
        x_aug_os = torch.stack(x_aug, dim=1)
        pool = nn.AdaptiveAvgPool2d((7, 7))
        x_naug = pool(x_naug_os.view(-1, bd, bh, bw))
        x_aug = pool(x_aug_os.view(-1, bd, bh, bw))
        x_naug = x_naug.view(bn, -1, bd, 7, 7)
        x_aug = x_aug.view(bn, -1, bd, 7, 7)

        error_x = (x_naug[:, :, None] - x_aug[:, None]) ** 2
        error_x = error_x.mean(dim=(3,4,5))

        #no aug
        error_x0 = error_x.min(dim=2)[0]
        x_naug_mask = (error_x0 - error_x0.min(dim=1,keepdim=True)[0] <= 0).float()
        x_naug_mask = x_naug_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_naugf = (x_naug_os * x_naug_mask).sum(dim=1)  / x_naug_mask.sum(dim=1)

        #aug
        error_x1 = error_x.permute(0,2,1).min(dim=2)[0]
        x_aug_mask = (error_x1 - error_x1.min(dim=1,keepdim=True)[0] <= 0).float()
        x_aug_mask = x_aug_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_augf = (x_aug_os * x_aug_mask).sum(dim=1)  / x_aug_mask.sum(dim=1)

        return x_naugf, x_augf
    
    def select_stands_v2(self, x_naug, x_aug, eps=1e-20):
        bn, bd, bh, bw = x_naug[0].size()
        bm = len(x_naug)
        x_naug_os = torch.stack(x_naug, dim=1) #N, M, C, H, W
        x_aug_os = torch.stack(x_aug, dim=1) #N, M, C, H, W
        
        '''
        dist_v = torch.pow(x_naug_os - x_aug_os, 2).mean(dim=2) #N, M, H, W
        pool = nn.AdaptiveAvgPool2d((2, 5))
        dist_v = pool(dist_v) #N, M, H, W
        dist_v_loss = dist_v.view(bn, bm, -1) #N, M, HxW
        dist_v_mat = dist_v_loss[:, :, :, None] - dist_v_loss[:, :, None, :]
        '''
        
        pool = nn.AdaptiveAvgPool2d((2, 5))
        x_naug_p = pool(x_naug_os.view(-1, bd, bh, bw))
        x_aug_p = pool(x_aug_os.view(-1, bd, bh, bw))
        x_naug_p = x_naug_p.view(bn, bm, bd, -1).permute(0,1,3,2)
        x_aug_p = x_aug_p.view(bn, bm, bd, -1).permute(0,1,3,2)

        dist_v_mat = torch.pow(x_naug_p[:, :, :, None] - x_aug_p[:, :, None], 2).mean(dim=4) #N, M, HxW, HxW


        #compute loss
        PATCH_N = dist_v_mat.size(-1)
        dist_v_mat_z_mask = torch.eye(PATCH_N, PATCH_N).to(x_naug_os.device)
        dist_v_mat_z_mask = dist_v_mat_z_mask.view(1, 1, PATCH_N, PATCH_N)
        dist_v_mat_z_mask = dist_v_mat_z_mask.expand(bn, bm, PATCH_N, PATCH_N)

        dist_v_mat_z = (dist_v_mat * dist_v_mat_z_mask).sum(dim=3)
        dist_v_mat_m = dist_v_mat.sum(dim=3)
        contrast_loss = -torch.log(1 - dist_v_mat_z / (dist_v_mat_m + eps))
        contrast_loss = contrast_loss.mean()

        #select
        con_mat = dist_v_mat_z / (dist_v_mat_m + eps) #N, M, HxW
        con_mat = con_mat.sum(dim=2)
        con_mat_v, con_mat_i = con_mat.min(dim=1) #N
        
        con_mat_i = con_mat_i.view(bn, 1, 1, 1, 1).expand(bn, 1, bd, bh, bw)
        x_naugf = torch.gather(x_naug_os, 1, con_mat_i)
        x_augf = torch.gather(x_aug_os, 1, con_mat_i)
   

        return  x_naugf, x_augf, contrast_loss


        
    
    def write_img(self, tc_images, img_metas, targets_):
        for idx in range(len(targets_)):
            target_ = targets_[idx]
            
            bbs = target_['boxes'].clone().detach()
            lls = target_['labels'].clone().detach()
            
            #if len(bbs.size()) == 2 and bbs.size(0) > 0:
            if 1:
               lim = tc_images[idx].permute(1,2,0).contiguous()
               lim -= lim.min()
               cv_im = lim.cpu().detach().numpy()
               cv_im = np.array(cv_im*255, np.int32)

               mbbs = bbs.cpu().detach().numpy()
               for ii in range(bbs.size(0)):
                   cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)

               cv2.imwrite('./test_img/'+str(idx)+'.jpg', cv_im)
    
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
    
    #############################multiple experts###############################
    def perEvaluation(self, tc_images, img_metas, targets, tc_feat_dict, tcaug_feat_dict, results_parameters={}):
        tc_targets = []
        tc_aug_targets = []
        
        self.update_student(NET_MOMENTUM=1.0)
        self.student.eval()
        targets = None
        if tc_feat_dict is not None:
            num_list = len(tc_feat_dict)
        else:
            num_list = len(tcaug_feat_dict)
        with torch.no_grad():
            for idx_t in range(num_list):
                if tc_feat_dict is not None:
                    outputs = self.student(tc_images, img_metas, targets, mode=0, results_parameters=results_parameters,\
                                    features_ff=tc_feat_dict[idx_t], test_exm=1)
                    tc_targets.append(outputs['dets_eval'])
                    #print(idx_t, outputs['dets'][0]['boxes'], len(outputs['dets'][0]['boxes']))
                
                if tcaug_feat_dict is not None:
                    outputs_aug = self.student(tc_images, img_metas, targets, mode=0, results_parameters=results_parameters, \
                                    features_ff=tcaug_feat_dict[idx_t], test_exm=1)
                
                    tc_aug_targets.append(outputs_aug['dets_eval'])
                    #print(outputs['dets'][0]['scores'])
        
        #print('---0:', tc_targets[0][0]['boxes'])
        #print('---1:', tc_targets[1][0]['boxes'])
        self.student.train()
        return tc_targets, tc_aug_targets
    
    ##########################method 3#################################
    def save_img(self,img_tensor_, mask_tensor_i, idx='0', outputs='./WZ-Evals'):

        if not os.path.exists(outputs):
            os.makedirs(outputs)

        mask_tensor_ = copy.deepcopy(mask_tensor_i)
        mask_tensor_ = F.interpolate(mask_tensor_.unsqueeze(1).float(), size=(img_tensor_.size(-2), img_tensor_.size(-1)), mode='bilinear') 
        mask_tensor_ = mask_tensor_[:,0]

        img_tensor = copy.deepcopy(img_tensor_[0])
        mask_tensor = copy.deepcopy(mask_tensor_[0])
        device = img_tensor.device

        #color_mean_tensor = torch.from_numpy(color_mean).to(device)
        #color_std_tensor = torch.from_numpy(color_std).to(device)
        img_tensor = img_tensor * 0.5
        img_tensor = img_tensor + 0.5
        img_tensor = img_tensor * 255
        
        #print('---1:', mask_tensor.max(), mask_tensor_.size())
        #mark_ = (mask_tensor == 1).float()
        #mark_ = ((mask_tensor <= 1) & (mask_tensor >= 0)).float()

        mark_0 = (mask_tensor == 0).float()
        mark_1 = (mask_tensor == 1).float()
        mark_2 = (mask_tensor == 2).float()
        mark_3 = (mask_tensor == 3).float()
        mark_4 = (mask_tensor == 4).float()

        img_tensor_0 = img_tensor * 0.3 * (1- mark_0[None] )+ img_tensor * mark_0[None] 
        img_tensor_1 = img_tensor * 0.3 * (1- mark_1[None] )+ img_tensor * mark_1[None] 
        img_tensor_2 = img_tensor * 0.3 * (1- mark_2[None] )+ img_tensor * mark_2[None] 
        img_tensor_3 = img_tensor * 0.3 * (1- mark_3[None] )+ img_tensor * mark_3[None] 
        img_tensor_4 = img_tensor * 0.3 * (1- mark_4[None] )+ img_tensor * mark_4[None] 


        cv_im_0 = img_tensor_0.permute(1,2,0).cpu().detach().numpy()
        cv_im_0 = np.array(cv_im_0, np.int32)

        cv_im_1 = img_tensor_1.permute(1,2,0).cpu().detach().numpy()
        cv_im_1 = np.array(cv_im_1, np.int32)

        cv_im_2 = img_tensor_2.permute(1,2,0).cpu().detach().numpy()
        cv_im_2 = np.array(cv_im_2, np.int32)


        cv_im_3 = img_tensor_3.permute(1,2,0).cpu().detach().numpy()
        cv_im_3 = np.array(cv_im_3, np.int32)

        cv_im_4 = img_tensor_4.permute(1,2,0).cpu().detach().numpy()
        cv_im_4 = np.array(cv_im_4, np.int32)

        cv2.imwrite(outputs+'/'+idx+'_0.jpg', cv_im_0)
        cv2.imwrite(outputs+'/'+idx+'_1.jpg', cv_im_1)
        cv2.imwrite(outputs+'/'+idx+'_2.jpg', cv_im_2)
        cv2.imwrite(outputs+'/'+idx+'_3.jpg', cv_im_3)
        cv2.imwrite(outputs+'/'+idx+'_4.jpg', cv_im_4)
        #cv2.imwrite(outputs+'/0.jpg', cv_im)
    

    def save_img_v2(self,img_tensor_, mask_tensor_i, idx='0', outputs='./WZ-Evals/probs'):
        if not os.path.exists(outputs):
            os.makedirs(outputs)

        mask_tensor_ = mask_tensor_i.clone().detach() #copy.deepcopy(mask_tensor_i)
        mask_tensor_ = F.interpolate(mask_tensor_, size=(img_tensor_.size(-2), img_tensor_.size(-1)), mode='bilinear') 
        #mask_tensor_ = mask_tensor_[:,0]

        img_tensor = copy.deepcopy(img_tensor_[0])
        mask_tensor = copy.deepcopy(mask_tensor_[0])
        device = img_tensor.device

        #color_mean_tensor = torch.from_numpy(color_mean).to(device)
        #color_std_tensor = torch.from_numpy(color_std).to(device)
        img_tensor = img_tensor * 0.5
        img_tensor = img_tensor + 0.5
        img_tensor = img_tensor * 255
        
        #print('---1:', mask_tensor.max(), mask_tensor_.size())
        #mark_ = (mask_tensor == 1).float()
        #mark_ = ((mask_tensor <= 1) & (mask_tensor >= 0)).float()

        mask_tensor_v, mask_tensor_ii = torch.topk(mask_tensor, 2, dim=0)
        mask_tensor = mask_tensor - mask_tensor_v[0:1]

        mark_0 = (mask_tensor[0] - 0.0>=0).float() #(mask_tensor == 0).float()
        mark_1 = (mask_tensor[1] - 0.0>=0).float() #(mask_tensor == 1).float()
        mark_2 = (mask_tensor[2] - 0.0>=0).float() #(mask_tensor == 2).float()
        mark_3 = (mask_tensor[3] - 0.0>=0).float() #(mask_tensor == 3).float()
        mark_4 = (mask_tensor[4] - 0.0>=0).float() #(mask_tensor == 4).float()

        img_tensor_0 = img_tensor * 0.0 + img_tensor * mark_0[None] 
        img_tensor_1 = img_tensor * 0.0 + img_tensor * mark_1[None] 
        img_tensor_2 = img_tensor * 0.0 + img_tensor * mark_2[None] 
        img_tensor_3 = img_tensor * 0.0 + img_tensor * mark_3[None] 
        img_tensor_4 = img_tensor * 0.0 + img_tensor * mark_4[None] 


        cv_im_0 = img_tensor_0.permute(1,2,0).cpu().detach().numpy()
        cv_im_0 = np.array(cv_im_0, np.int32)

        cv_im_1 = img_tensor_1.permute(1,2,0).cpu().detach().numpy()
        cv_im_1 = np.array(cv_im_1, np.int32)

        cv_im_2 = img_tensor_2.permute(1,2,0).cpu().detach().numpy()
        cv_im_2 = np.array(cv_im_2, np.int32)


        cv_im_3 = img_tensor_3.permute(1,2,0).cpu().detach().numpy()
        cv_im_3 = np.array(cv_im_3, np.int32)

        cv_im_4 = img_tensor_4.permute(1,2,0).cpu().detach().numpy()
        cv_im_4 = np.array(cv_im_4, np.int32)

        cv2.imwrite(outputs+'/'+idx+'_0.jpg', cv_im_0)
        cv2.imwrite(outputs+'/'+idx+'_1.jpg', cv_im_1)
        cv2.imwrite(outputs+'/'+idx+'_2.jpg', cv_im_2)
        cv2.imwrite(outputs+'/'+idx+'_3.jpg', cv_im_3)
        cv2.imwrite(outputs+'/'+idx+'_4.jpg', cv_im_4)
        #cv2.imwrite(outputs+'/0.jpg', cv_im)


    #############################method 4############################
    def save_img_v40(self,img_tensor_, mask_tensor_i, idx='0', outputs='./WZ-Evals/probs'):
        if not os.path.exists(outputs):
            os.makedirs(outputs)
        
        for idx_ in range(len(mask_tensor_i)):
            
            mask_tensor_ = mask_tensor_i[idx_].clone().detach() #copy.deepcopy(mask_tensor_i)
            if len(mask_tensor_.size()) < 4:
                mask_tensor_ = mask_tensor_.unsqueeze(1)
            mask_tensor_ = F.interpolate(mask_tensor_, size=(img_tensor_.size(-2), img_tensor_.size(-1)), mode='bilinear') 
            #mask_tensor_ = mask_tensor_[:,0]

            img_tensor = copy.deepcopy(img_tensor_[0])
            mask_tensor = copy.deepcopy(mask_tensor_[0])
            device = img_tensor.device

            #color_mean_tensor = torch.from_numpy(color_mean).to(device)
            #color_std_tensor = torch.from_numpy(color_std).to(device)
            img_tensor = img_tensor * 0.5
            img_tensor = img_tensor + 0.5
            img_tensor = img_tensor * 255
        
            #print('---1:', mask_tensor.max(), mask_tensor_.size())
            #mark_ = (mask_tensor == 1).float()
            #mark_ = ((mask_tensor <= 1) & (mask_tensor >= 0)).float()

            #mask_tensor_v, mask_tensor_ii = torch.topk(mask_tensor, 2, dim=0)
            #mask_tensor = mask_tensor - mask_tensor_v[0:1]

            mark_0 = (mask_tensor - 0.5 >= 0).float() #(mask_tensor == 0).float()
        

            img_tensor_0 = img_tensor * 0.0 + img_tensor * mark_0
        


            cv_im_0 = img_tensor_0.permute(1,2,0).cpu().detach().numpy()
            cv_im_0 = np.array(cv_im_0, np.int32)

            cv2.imwrite(outputs+'/'+str(idx_).zfill(3)+ "_" + idx + '.jpg', cv_im_0)
    

    def save_img_v41(self,img_tensor_, mask_tensor_i, idx='0', outputs='./WZ-Evals/probs'):
        if not os.path.exists(outputs):
            os.makedirs(outputs)
        
        for idx_ in range(len(mask_tensor_i)):
            
            mask_tensor_ = mask_tensor_i[idx_].clone().detach() #copy.deepcopy(mask_tensor_i)
            if len(mask_tensor_.size()) < 4:
                mask_tensor_ = mask_tensor_.unsqueeze(1)
            mask_tensor_ = F.interpolate(mask_tensor_, size=(img_tensor_.size(-2), img_tensor_.size(-1)), mode='bilinear') 
            #mask_tensor_ = mask_tensor_[:,0]

            img_tensor = copy.deepcopy(img_tensor_[0])
            mask_tensor = copy.deepcopy(mask_tensor_[0])
            device = img_tensor.device

            #color_mean_tensor = torch.from_numpy(color_mean).to(device)
            #color_std_tensor = torch.from_numpy(color_std).to(device)
            img_tensor = img_tensor * 0.5
            img_tensor = img_tensor + 0.5
            img_tensor = img_tensor * 255
        
            #print('---1:', mask_tensor.max(), mask_tensor_.size())
            #mark_ = (mask_tensor == 1).float()
            #mark_ = ((mask_tensor <= 1) & (mask_tensor >= 0)).float()

            #mask_tensor_v, mask_tensor_ii = torch.topk(mask_tensor, 2, dim=0)
            #mask_tensor = mask_tensor - mask_tensor_v[0:1]

            mark_0 = (mask_tensor - 0.5 >= 0).float() #(mask_tensor == 0).float()
        

            img_tensor_0 = img_tensor * 0.0 + img_tensor * mark_0
        


            cv_im_0 = img_tensor_0.permute(1,2,0).cpu().detach().numpy()
            cv_im_0 = np.array(cv_im_0, np.int32)

            cv2.imwrite(outputs+'/'+str(idx_).zfill(3)+ "_" + idx + '.jpg', cv_im_0)
