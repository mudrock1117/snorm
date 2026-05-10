import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops

import numpy as np
from torchvision.ops import boxes as box_ops
#from mamba_ssm.modules.mamba_simple import Mamba
#from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from detection.modeling.unet.gate_moe import GatedMoE
from detection.modeling.unet.gate_score import GatedScore, GatedScore_S8, GatedScore_S16
from detection.modeling.unet.MedianPool2d import MedianPool2d
from detection.modeling.conV.tconV import tconV

class SNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_a = None #nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha_u = None
        
        self.memory_len = 100
        self.memory_aplpha_a = np.zeros((self.memory_len, 3))
        self.memory_aplpha_u = np.zeros((self.memory_len, 3))
        self.memory_loss_weights = np.zeros((self.memory_len))
        self.memory_index = 0
        self.memory_loc = 0
        self.ema_rate = 0.9

        self.proj_prev = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid()
            )

        self.proj_norm_1x1 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.ReLU()
            )
        

        #down dims
        self.final_conv = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.ReLU()
            )
        #self.final_conv = tconV(512)
        
        self.alpha_ba = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.alpha_bu = nn.Parameter(torch.zeros(3), requires_grad=True)
        nn.init.uniform_(self.alpha_ba)
        nn.init.uniform_(self.alpha_bu)

        self.gllc_rate = 0.1
        self.error_thresh = 0.05
        self.first_run = True


        ###########################################
        self.weight_gen_unet2 = GatedScore_S16(3, 1) #GatedScore(1024, 1)#GatedScore_S8(512, 1)

        ###########################################
        self.output_index = 0
        self.medpool = MedianPool2d(kernel_size=3, stride=1, padding=1)
        
        self.iter_save = 0
        self.iter_save_train = 0

    def forward(self, feats, use_moe=False, use_moe_fusion=False, moe_mode=0):
        
        if self.training:
            if moe_mode == 0:
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.7)
                feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
                scores_w0 = self.weight_gen_unet2(feats_0_w.detach())
                #scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)), mode='bilinear')
                #output = self.final_conv(feats_0)
                #print('----1:', scores_w0.size())
                output = feats_0
                output_dict = {'feature_dict':[output], 'weight_dict':[scores_w0]}
                self.iter_save_train = (self.iter_save_train + 1) % 2000
                ###draw
                if self.iter_save_train % 200 == 0:
                    #print(scores_w0.size(), output.size())
                    scores_w0_vis = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)), mode='bilinear')
                    scores_w0_vis = scores_w0_vis.expand(output.size())
                    vis_images_all = torch.zeros((1, output.size(1), 3 * output.size(-2), output.size(-1))).to(output.device)
                    vis_images_all[:, :, 0:output.size(-2)] = feats[0:1]
                    vis_images_all[:, :, output.size(-2):output.size(-2)*2] = feats_0[0:1]
                    vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = scores_w0_vis[0:1]
                    import torchvision.utils as vutils
                    
                    name = str(self.iter_save_train).zfill(4)    
                    vutils.save_image(vis_images_all.data, './samples/training/' + name +'.png', nrow=12, value_range=(-1, 1), normalize=True)

                return output, output_dict

            elif moe_mode == 1:
                error_thresh = 0.001
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.7)
                feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
                scores_w0 = self.weight_gen_unet2(feats_0_w.detach())
                #scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)), mode='bilinear')
                output = None #self.final_conv(feats_0)
                save_im0 = feats_0.clone()

                feats_ss = feats_0 * 1
                scores_ws = (scores_w0 * 1).sigmoid().detach()
                weight_ss = scores_w0
                feature_dict = [] #[feats_0]
                weight_dict = [] #[scores_w0]
                index_list = [1,2,3]
                kernel_sz_set = [
                    [2,2],
                    [3,3]
                ]
                
                ####reasoning
                feature_dict_per = []
                weight_dict_per = []
                feature_dict_fused = []
                weight_dict_fused = []
                win_sz_set = []
                error_thresh_1 = 0.005
                final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
                for idx_ in range(len(kernel_sz_set)):
                    feats_ = self.user_norm_v2(feats, mode=0, kernel_sz_set=kernel_sz_set[idx_], w_a=1.0, w_u=1.0, ema=0.7)
                    #scores_w = self.weight_gen_unet2(feats_)
                    feats_w = F.interpolate(feats_, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
                    scores_w = self.weight_gen_unet2(feats_w)
                    scores_cw = scores_w.clone().sigmoid().detach()

                    ###
                    mask = ((scores_cw - scores_ws) > error_thresh_1).float()
                    mask = F.interpolate(mask, size=(feats_.size(-2), feats_.size(-1)), mode='bilinear') #bilinear
                    mask = (mask > 0.5).float()
                    feats_0 = feats_0 * (1 - mask) + feats_ * mask
                    #feats_0 = 0.9 * feats_0 + 0.1 * feats_n0

                    ###
                    #mask = ((scores_cw - scores_ws) > error_thresh).float()
                    #scores_ws = scores_ws * (1 - mask) + scores_cw * mask
                    #weight_ss = weight_ss * (1 - mask) + scores_w * mask
                    #feats_ss = feats_ss * (1 - mask) + feats_ * mask
                    weight_ss = scores_cw
                    feats_ss = feats_0

                    #feature_dict_per.append(self.final_conv(feats_))
                    feature_dict_per.append(feats_)
                    weight_dict_per.append(scores_w)
                    #feature_dict_fused.append(self.final_conv(feats_ss))
                    feature_dict_fused.append(feats_ss)
                    weight_dict_fused.append(weight_ss)
                    win_sz_set.append(kernel_sz_set[idx_])

                    final_mask = final_mask + mask
                    
                ####out
                output = None 
                #self.save_img(feats_ss)
                output_0 = feats_ss #self.final_conv(feats_ss)
                #feature_dict = [output, output_0]
                output_dict = {'feature_dict':feature_dict, 'weight_dict':weight_dict, 'output_0':output_0, \
                'feature_dict_per':feature_dict_per, 'weight_dict_per':weight_dict_per,\
                'feature_dict_fused':feature_dict_fused, 'weight_dict_fused':weight_dict_fused, 'win_sz_set':win_sz_set}

                ###draw
                if self.iter_save % 200 == 0:
                    vis_images_all = torch.zeros((1, output_0.size(1), 4 * output_0.size(-2), output_0.size(-1))).to(output_0.device)
                    vis_images_all[:, :, 0:output_0.size(-2)] = save_im0[0:1]
                    vis_images_all[:, :, output_0.size(-2):output_0.size(-2)*2] = feats_[0:1]
                    vis_images_all[:, :, 2*output_0.size(-2):output_0.size(-2)*3] = feats_ss[0:1]
                    vis_images_all[:, :, 3*output_0.size(-2):output_0.size(-2)*4] = final_mask[0:1] 
                    import torchvision.utils as vutils
                    
                    name = str(self.iter_save).zfill(4)    
                    vutils.save_image(vis_images_all.data, './samples/training/' + name +'.png', nrow=12, value_range=(-1, 1), normalize=True)

                return output_0, output_dict

            elif moe_mode == 2:
                feats_0 = self.user_norm(feats, mode=0)
                scores_w0 = self.weight_gen_unet2(feats_0.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)), mode='bilinear')
                output = self.final_conv(feats_0)

                feats_ss = feats_0 * 1
                scores_ws = (scores_w0 * 1).sigmoid()
                #feature_dict = [feats_0]
                weight_dict = [scores_w0]
                index_list = [1, 2, 3, 4]
                for idx_ in range(len(index_list)):
                    mode_idx = index_list[idx_]
                    feats_ = self.user_norm(feats, mode=mode_idx)
                    scores_w = self.weight_gen_unet2(feats_).sigmoid().detach()

                    scores_w = F.interpolate(scores_w, size=(feats_.size(-2), feats_.size(-1)), mode='bilinear')

                    mask = ((scores_w - scores_ws) > self.error_thresh).float()
                    feats_ss = feats_ss * (1 - mask) + feats_ * mask
                    scores_ws = scores_ws * (1 - mask) + scores_w * mask

                    #feature_dict.append(feats_)
                    #weight_dict.append(scores_w)
                    
                output = self.final_conv(feats_0)
                output_0 = self.final_conv(feats_ss)
                feature_dict = [output, output_0]
                output_dict = {'feature_dict':feature_dict, 'weight_dict':weight_dict, 'output_0':output_0}
                return output, output_dict
        else:
            index_list = [1, 2, 3, 4, 5]
            feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.7)
            #feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
            scores_w0 = self.weight_gen_unet2(feats_0).sigmoid()
            save_im0 = copy.deepcopy(feats_0)

            kernel_sz_set = [
                [1,2],
                [1,3],
                [2,1],
                [2,2],
                [3,3],
                [3,5],
                #[5, 5],
                #[5, 7],
                #[5, 9],
                #[5, 11]
                #[7, 7],
                #[9, 9],
                #[13, 13]
            ]
            
            w_au = [
                [1.0, 1.0, 0.7],
                [1.0, 1.0, 0.6],
                [1.0, 1.0, 0.5],
                [1.0, 1.0, 0.8],
                #[1.2, 1.0, 0.7],
                #[1.3, 1.0, 0.7],
                #[1.5, 1.0, 0.7],
                #[1.0, 1.0, 0.5],

                #[1.2, 1.0, 0.5],
                #[1.3, 1.0, 0.5],
                #[1.0, 1.0, 0.3],
                #[1.2, 1.0, 0.3],
                #[1.3, 1.0, 0.3],
                #[1.0, 1.0, 0.9],
                #[1.2, 1.0, 0.9],
                #[1.3, 1.0, 0.9],
            ]
            
            #005
            #error_thresh_0 = 0.01
            #error_thresh_1 = 0.004
            
            ##006: 34.32
            #error_thresh_0 = 0.06
            #error_thresh_1 = 0.01

            #007: 34.31
            #error_thresh_0 = 0.06
            #error_thresh_1 = 0.015
            
            #008:34.32
            #error_thresh_0 = 0.06
            #error_thresh_1 = 0.02
            
            #009: 34.30
            #error_thresh_0 = 0.06
            #error_thresh_1 = 0.03
            
            #010: 34.31
            #error_thresh_0 = 0.06
            #error_thresh_1 = 0.04
            
            ###0006_007-(0.06, 0.x)
            ##0006_008-(0.06, 0.02)
            #0006_009-(0.06, 0.01)
            #0006_010-(0.06, 0.00)
            #0006_011-(0.1, 0.02)

            #0.2, 0.1 - 33.14
            #0.3, 0.1 - 33.21
            #0.4, 0.1 - 33.23
            #0.5, 0.1 - 33.24
            #0.6, 0.1 - 33.24
            #0.7, 0.1 - 33.24
            error_thresh_0 = 0.1
            error_thresh_1 = 0.1
            
            ##########
            
            final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
            medpool_0 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
            medpool_1 = nn.MaxPool2d(kernel_size=[13, 25], stride=1, padding=[6,12])
            medpool_am = nn.AdaptiveAvgPool2d((5, 7))
            #scores_w0 = self.compute_sim(scores_w0)
            #scores_w0 = medpool_1(scores_w0)
            scores_up = copy.deepcopy(scores_w0)
            
            

            #test
            medpool_2 = nn.MaxPool2d(kernel_size=[13, 25], stride=1, padding=[6,12])
            #medpool_2 = nn.MaxPool2d(kernel_size=[15, 29], stride=1, padding=[7,14])
            #medpool_2 = nn.MaxPool2d(kernel_size=[11, 23], stride=1, padding=[5,11]) #AvgPool2d
            #medpool_2 = nn.MaxPool2d(kernel_size=[9, 19], stride=1, padding=[4,9])
            #medpool_2 = nn.MaxPool2d(kernel_size=[5, 9], stride=1, padding=[2,4])

            #medpool_2 = nn.MaxPool2d(kernel_size=[13, 13], stride=1, padding=[6,6])
            mid_masks = []
            '''
            for idx_ in range(len(kernel_sz_set)):
                for idx_k in range(len(w_au)):
                    feats_ = self.user_norm_v2(feats,  kernel_sz_set=kernel_sz_set[idx_], w_a=w_au[idx_k][0], w_u=w_au[idx_k][1], ema=w_au[idx_k][2], eps=1e-5)
                    scores_w = self.weight_gen_unet2(feats_).sigmoid()
                    #print(feats_.size(),  scores_w.size())
                    #print(scores_w)
                    #scores_w = 1-medpool_0(1-scores_w)
                    #scores_w = medpool_1(scores_w)

                    #scores_w = self.compute_sim(scores_w)
                    #print(scores_w)
                    #print(scores_w.max())
                    #mask_0 = (scores_w - scores_w0) > error_thresh_0
                    #mask_1 = (scores_w - scores_up) > error_thresh_1
                    #mask = (mask_0.float() * mask_1.float() > 0.1).float()
                    #scores_w = medpool_2(scores_w * mask)
                    #mask = medpool_2(mask)
                    #print((scores_w * mask).max())
                    mask_v0 = self.compute_sim_0718(scores_w, scores_w0)
                    mask_v1 = self.compute_sim_0718(scores_w, scores_up)
                    #print(mask_v0.max(), mask_v0.min())
                    mask_0 = mask_v0 > error_thresh_0
                    mask_1 = mask_v1 > error_thresh_1
                    mask = (mask_0.float() * mask_1.float() > 0.1).float()
                    #print(scores_w0.max(), scores_w0.min())
                    
                    mask_sc = F.interpolate(mask, size=(scores_up.size(-2), scores_up.size(-1)), mode='nearest') #bilinear
                    scores_up = scores_up * (1 - mask_sc) + scores_w * mask_sc
                    mask = F.interpolate(mask, size=(feats_.size(-2), feats_.size(-1)), mode='nearest') #bilinear
                    mask = (mask > 0.5).float()
                    feats_0 = feats_0 * (1 - mask) + feats_ * mask
                    final_mask = final_mask + mask
                    mid_masks.append(mask * feats_)
            '''

            ###draw
            output = feats_0
            self.iter_save = (self.iter_save + 1) % 2000
            if self.iter_save % 100 == 1:
            #if 0:
                vis_images_all = torch.zeros((1, output.size(1), 6 * output.size(-2), output.size(-1))).to(output.device)
                vis_images_all[:, :, 0:output.size(-2)] = save_im0[0:1]
                vis_images_all[:, :, output.size(-2):output.size(-2)*2] = feats_0[0:1]
                vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = output[0:1]
                vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = mid_masks[0]
                vis_images_all[:, :, 4*output.size(-2):output.size(-2)*5] = mid_masks[2]
                #vis_images_all[:, :, 5*output.size(-2):output.size(-2)*6] = mid_masks[4]
                
                #mark_ll = (scores_w0 > 0.8).float()
                mark_ll = (final_mask > 0.1).float()
                mark_ll = F.interpolate(mark_ll, size=(feats_0.size(-2), feats_0.size(-1)), mode='nearest') #bilinear
                mark_ll = mark_ll * feats_0
                #print(mark_ll.size())
                vis_images_all[:, :, 5*output.size(-2):output.size(-2)*6] = mark_ll
                
                import torchvision.utils as vutils
                name = str(self.iter_save).zfill(4)    
                vutils.save_image(vis_images_all.data, './samples/test/' + name +'.png', nrow=12, value_range=(-1, 1), normalize=True)
            
            output = feats_0 #self.final_conv(feats_0)
            output_dict = {'feature_dict':[feats_0], 'weight_dict':[scores_w0]}

            return output, output_dict
    


    def calc_std_em(self, feat, eps=1e-6):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]

        feat = feat ** 2
        feat = (self.pool(feat) + eps).sqrt() 
        feats = F.interpolate(feat, size=(size[-2], size[-1]), mode='bilinear')

        return feats
        
    
    def user_norm_v2(self, feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, eps=1e-6, ema=0.1):
        pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set)
        feats_pool = pool_func(feats)

        feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)), mode='nearest')
        pfeats = feats - feats_pool
        
        sfeats = pfeats ** 2
        sfeats = (pool_func(sfeats) + eps).sqrt() 

        sfeats = F.interpolate(sfeats, size=(feats.size(-2), feats.size(-1)), mode='nearest')
        alpha_ba = self.alpha_ba #.clone().detach()
        alpha_bu = self.alpha_bu #.clone().detach()

        feats_mean, feats_std = calc_mean_std(feats)
        ba_ema_rate = ema
        sfeats = sfeats * ba_ema_rate + (1 - ba_ema_rate) * feats_std
        alpha_bu = alpha_bu[None, :, None, None] #  * 0.7 + 0.3 * feats_mean

        nfeats = w_a * pfeats * alpha_ba[None, :, None, None] / sfeats + w_u * alpha_bu
        #nfeats = self.medpool(nfeats)

        #conv
        #nfeats = self.proj_norm_1x1(nfeats.float())
        #nfeats = self.proj_norm_1x1(nfeats.float())
        return nfeats.float()

    '''
    def update_norm_parameters(self, feats, is_update=False, loss_weights=None):
        #m5
        #update self.alpha
        if self.memory_index < self.memory_len:
            feats_mean, feats_std = calc_mean_std(feats)
            feats_mean = feats_mean.view(feats_mean.size(0), feats_mean.size(1))
            feats_std = feats_std.view(feats_std.size(0), feats_std.size(1))

            self.memory_aplpha_a[self.memory_index] = feats_std.mean(dim=0).cpu().detach().numpy()
            self.memory_aplpha_u[self.memory_index] = feats_mean.mean(dim=0).cpu().detach().numpy()
            self.memory_loss_weights[self.memory_index] = loss_weights.cpu().detach().numpy()
            self.memory_index += 1
            
            alpha_ba = torch.from_numpy(self.memory_aplpha_a[0:self.memory_index].sum(axis=0) / self.memory_index)
            alpha_bu = torch.from_numpy(self.memory_aplpha_u[0:self.memory_index].sum(axis=0) / self.memory_index)
            self.loss_weight_mem = self.memory_loss_weights[0:self.memory_index].sum(axis=0) / self.memory_index
            alpha_ba = alpha_ba.to(feats.device)
            alpha_bu = alpha_bu.to(feats.device)
            self.alpha_ba.data = self.alpha_ba.data.clone().detach() * 0.0 + alpha_ba.clone().detach()
            self.alpha_bu.data = self.alpha_bu.data.clone().detach() * 0.0 + alpha_bu.clone().detach()
        else:
            weights = self.loss_weight_mem / (loss_weights + self.loss_weight_mem)
            weights = 1 - weights

            feats_mean, feats_std = calc_mean_std(feats)
            feats_mean = feats_mean.view(feats_mean.size(0), feats_mean.size(1))
            feats_std = feats_std.view(feats_std.size(0), feats_std.size(1))

            self.alpha_ba.data = self.alpha_ba.data.clone().detach() * self.ema_rate * weights + \
                (1 - self.ema_rate) * feats_std.mean(dim=0).data.clone().detach() * (1 - weights)

            self.alpha_bu.data = self.alpha_bu.data.clone().detach() * self.ema_rate * weights + \
                (1 - self.ema_rate) * feats_mean.mean(dim=0).data.clone().detach() * (1 - weights)
            
            self.loss_weight_mem = self.loss_weight_mem * self.ema_rate * weights + \
                (1 - self.ema_rate) * loss_weights.cpu().detach().numpy() * (1 - weights)
            
            self.memory_index = self.memory_len
            #self.memory_index += 1
            

        return
        

    def compute_norm_parameters(self, device=None):
        alpha_ba = torch.from_numpy(self.memory_aplpha_a[0:self.memory_loc].sum(axis=0) / self.memory_loc)
        alpha_bu = torch.from_numpy(self.memory_aplpha_u[0:self.memory_loc].sum(axis=0) / self.memory_loc)
        alpha_ba = alpha_ba.to(device)
        alpha_bu = alpha_bu.to(device)
        self.alpha_ba.data = self.alpha_ba.data.clone().detach() * 0.0 + alpha_ba.clone().detach()
        self.alpha_bu.data = self.alpha_bu.data.clone().detach() * 0.0 + alpha_bu.clone().detach()
        return 
    '''
    
    def save_img(self, img_tensor_, labels=None, bboxes=None, idx='0', outputs='./WZ-Evals', use_norm=False):
        import os
        import cv2
        if not os.path.exists(outputs):
            os.makedirs(outputs)
        
        img_tensor = copy.deepcopy(img_tensor_)
        device = img_tensor.device
            
        lim = img_tensor[0].permute(1,2,0).contiguous()
        lim = lim * 0.5 + 0.5
        #lim -= lim.min()
        cv_im = lim.cpu().detach().numpy()
        cv_im = np.array(cv_im * 255, np.int32)
        if bboxes is not None:
            mbbs = bboxes.cpu().detach().numpy()
            for ii in range(mbbs.shape[0]):
                cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)

        cv2.imwrite(outputs+'/'+idx+'.jpg', cv_im)
    
    '''
    def compute_sim(self, scores_w, eps=1e-6):
        medpool_am = nn.AdaptiveAvgPool2d((3, 5))
        scores_mask = (scores_w > 0.3).float()

        scores_w = medpool_am(scores_w * scores_mask)
        scores_mask = medpool_am(scores_mask)
        
        scores_w = scores_w / (scores_mask + eps)

        #print(scores_w.max(), scores_w.min())
        return scores_w
    '''

    def compute_sim(self, scores_w, eps=1e-6):
        medpool_am = nn.AdaptiveAvgPool2d((3, 3))
        scores_mask = (scores_w > 0.3).float()

        #scores_w = medpool_am(scores_w * scores_mask)
        #scores_mask = medpool_am(scores_mask)
        
        #scores_w = scores_w / (scores_mask + eps)

        scores_w = medpool_am(scores_w)

        #print(scores_w.max(), scores_w.min())
        return scores_w
    
    def compute_sim_0718(self, scores_w, scores_wx, eps=1e-5):
        medpool_2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        scores_w_m = medpool_2(scores_w)
        scores_wx_m = medpool_2(scores_wx)

        #print(scores_w.size())
        mask_x0 = scores_w_m > 0.6
        mask_x1 = scores_wx_m > 0.6
        mask_xf = (mask_x0 | mask_x1).float()

        scores_er = scores_w_m - scores_wx_m
        medpool_am = nn.AdaptiveAvgPool2d((5, 7))
        scores_er = medpool_am(scores_er * mask_xf)
        scores_mask = medpool_am(mask_xf)
        scores_ef = scores_er / (scores_mask + eps)

        return scores_ef






def calc_mean_std(feat, eps=1e-8):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std_batch(feat, eps=1e-8):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.permute(1, 0, 2, 3).view(C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
