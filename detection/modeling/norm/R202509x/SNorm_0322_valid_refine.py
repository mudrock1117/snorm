import os
import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops

import numpy as np
from torchvision.ops import boxes as box_ops
from detection.modeling.unet.gate_moe import GatedMoE
from detection.modeling.unet.gate_score import *  # GatedScore, GatedScore_S8, GatedScore_S16
from detection.modeling.unet.Irender import *
from detection.modeling.unet.MedianPool2d import MedianPool2d
from detection.modeling.conV.tconV import tconV
from detection.modeling.HDR.OURS.OHDR import AHDR
from detection.modeling.norm.lib_pert import *

from torchvision import transforms as v2
import matplotlib.pyplot as plt

class SNorm(nn.Module):
    """
    0318-op 版本：
    - 继承 SNorm_0322_valid_refine_top 的整体结构
    - 在 eval 分支中增加更多候选局部变换，用于适配雾霾 / 夜晚等复杂光照
      * user_norm_v2 多尺度局部归一化（原有）
      * local_retinex: Retinex 风格局部照度归一化
      * local_gamma_night: 自适应 gamma 夜景增强
      * local_bright_contrast_v2: 局部亮度/对比度变换（原有，但纳入候选）
    由 weight_gen_unet2 对这些候选进行评分并融合。
    """

    def __init__(self):
        super().__init__()
        self.alpha_a = None
        self.alpha_u = None

        self.memory_len = 100
        self.memory_aplpha_a = np.zeros((self.memory_len, 3))
        self.memory_aplpha_u = np.zeros((self.memory_len, 3))
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
        )

        # down dims
        self.final_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
        )

        # alpha_ba: 通道 std 尺度；alpha_bu: 通道 mean 偏置（与 update_norm_parameters / calc_mean_std 一致）。
        # 从源域目录估计并写回本文件: python snorm_source_stats.py --config-file <yaml> --source-dirs <dir...> --patch-snorm
        self.alpha_ba = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.alpha_bu = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=True)

        self.gllc_rate = 0.1
        self.error_thresh = 0.05
        self.first_run = True

        ###########################################
        # self.weight_gen_unet2 = GatedScore_S16(3, 1)
        # self.weight_gen_unet2 = GatedScore_S8(3, 1)
        self.weight_gen_unet2 = GatedScore_SV8(3, 1)

        self.render_net = AHDR(3, 64)

        # NOTE: 不再使用可学习的 gamma_conv，夜景增强改为纯算法式（基于亮度/照度自适应）

        ###########################################
        self.output_index = 0
        self.medpool = MedianPool2d(kernel_size=3, stride=1, padding=1)

        self.iter_save = 0
        self.iter_test_save = 0
        self.traing_iter_save = 0
        self.traing_iter_scale_save = 0

    def forward(self, feats, use_moe=False, use_moe_fusion=False, moe_mode=0, moe_mark=None, targets=None):
        ################## segnet ####################
        if self.training:
            # global
            if moe_mode == 0:
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1, 1], w_a=1.0, w_u=1.0, ema=1.0)
                feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1) // 2),
                                          mode='bilinear', align_corners=False)
                scores_w0 = self.weight_gen_unet2(feats_0_w.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)),
                                          mode='bilinear', align_corners=False)
                output = feats_0
                output_dict = {'feature_dict': [output], 'weight_dict': [scores_w0]}
                self.iter_save = (self.iter_save + 1) % 2000
                return output, output_dict

            # per local
            elif moe_mode == 1:
                kernel_sz_set = [
                    [3, 5],
                    [5, 3],
                    [7, 11],
                    [11, 7],
                    [10, 20],
                    [20, 10],
                ]
                sz_index_set = random.sample(range(0, len(kernel_sz_set)), 1)
                kernel_sz_set_u = kernel_sz_set[sz_index_set[0]]
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=kernel_sz_set_u,
                                            w_a=1.0, w_u=1.0, ema=0.85)
                feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1) // 2),
                                          mode='bilinear', align_corners=False)
                scores_w0 = self.weight_gen_unet2(feats_0_w.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)),
                                          mode='bilinear', align_corners=False)
                output = feats_0
                output_dict = {'feature_dict': [output], 'weight_dict': [scores_w0]}
                return output, output_dict

            # local（训练阶段仍维持原结构）
            elif moe_mode == 2:
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1, 1], w_a=1.0, w_u=1.0, ema=1.0)
                scores_w0 = self.weight_gen_unet2(feats_0.detach()).softmax(dim=1)[:, 1:]
                save_im0 = feats_0.clone()

                kernel_sz_set = [
                    [3, 5],
                    [5, 3],
                    [7, 11],
                    [11, 7],
                ]

                w_au = [
                    [1.0, 1.0, 0.9],
                ]

                error_thresh_0 = 0.0
                error_thresh_1 = 0.0

                scores_up = scores_w0.clone() * 1.0
                final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
                medpool_0 = MedianPool2d(kernel_size=3, stride=1, padding=1)
                medpool_1 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5)
                medpool_2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
                scores_w0 = medpool_0(scores_w0)
                scores_up = medpool_0(scores_up)

                feats_mem = []
                scores_mem = []
                feats_g = feats_0.clone()

                for idx_k in range(len(w_au)):
                    for idx_ in range(len(kernel_sz_set)):
                        feats_ = self.user_norm_v2(
                            feats_g,
                            kernel_sz_set=kernel_sz_set[idx_],
                            w_a=w_au[idx_k][0],
                            w_u=w_au[idx_k][1],
                            ema=0.85,
                            eps=1e-5,
                        )
                        scores_w = self.weight_gen_unet2(feats_.detach()).softmax(dim=1)[:, 1:]
                        scores_w = medpool_0(scores_w)

                        mask_0 = scores_w - scores_w0 >= error_thresh_0
                        mask_1 = scores_w - scores_up >= error_thresh_1
                        mask = (mask_0.float() * mask_1.float()).float()
                        mask = medpool_1(mask)

                        scores_up = scores_up * (1 - mask) + scores_w * mask
                        mask = F.interpolate(mask, size=(feats_.size(-2), feats_.size(-1)),
                                             mode='bicubic', align_corners=False)
                        feats_0 = feats_0 * (1 - mask) + feats_ * mask

                        feats_mem.append(feats_0.detach().clone())
                        scores_mem.append(scores_up.detach().clone())

                scores_mem_t = torch.cat(scores_mem, dim=1)
                weights = scores_mem_t / (scores_mem_t.sum(dim=1, keepdim=True) + 1e-5)
                weights = F.interpolate(weights, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)),
                                         mode='bicubic', align_corners=False)

                final_feats = 0.0
                for f_idx in range(len(feats_mem)):
                    final_feats = final_feats + feats_mem[f_idx] * weights[:, f_idx:f_idx + 1]
                output = 0.1 * feats_g + 0.9 * final_feats
                output_dict = {'feature_dict': [final_feats], 'weight_dict': scores_mem}
                return output, output_dict

            elif moe_mode == 3:
                feats_0 = self.user_norm(feats, mode=0)
                scores_w0 = self.weight_gen_unet2(feats_0.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)),
                                          mode='bilinear', align_corners=False)
                output = self.final_conv(feats_0)

                feats_ss = feats_0 * 1
                scores_ws = (scores_w0 * 1).sigmoid()
                weight_dict = [scores_w0]
                index_list = [1, 2, 3, 4]
                for idx_ in range(len(index_list)):
                    mode_idx = index_list[idx_]
                    feats_ = self.user_norm(feats, mode=mode_idx)
                    scores_w = self.weight_gen_unet2(feats_).sigmoid().detach()

                    scores_w = F.interpolate(scores_w, size=(feats_.size(-2), feats_.size(-1)),
                                             mode='bilinear', align_corners=False)

                    mask = ((scores_w - scores_ws) > self.error_thresh).float()
                    feats_ss = feats_ss * (1 - mask) + feats_ * mask
                    scores_ws = scores_ws * (1 - mask) + scores_w * mask

                output = self.final_conv(feats_0)
                output_0 = self.final_conv(feats_ss)
                feature_dict = [output, output_0]
                output_dict = {'feature_dict': feature_dict, 'weight_dict': weight_dict, 'output_0': output_0}
                return output, output_dict

            elif moe_mode == 4:
                scores_w0 = self.weight_gen_unet2(feats.detach())
                output = feats
                scores_w0 = F.interpolate(scores_w0, size=(feats.size(-2), feats.size(-1)),
                                          mode='bilinear', align_corners=False)
                output_dict = {'feature_dict': [output], 'weight_dict': [scores_w0]}
                self.iter_save = (self.iter_save + 1) % 2000
                return output, output_dict

            ################## render ####################
            elif moe_mode == 5:
                index_list = [1, 2, 3, 4, 5]
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1, 1], w_a=1.0, w_u=1.0, ema=0.7)
                scores_w0 = self.weight_gen_unet2(feats_0).sigmoid()
                save_im0 = feats_0.clone()

                kernel_sz_set = [
                    [3, 5],
                    [5, 3],
                    [7, 11],
                    [11, 7],
                ]

                w_au = [
                    [1.0, 1.0, 0.5],
                ]

                error_thresh_0 = 0.0
                error_thresh_1 = 0.0

                scores_up = scores_w0.clone() * 1.0
                final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
                medpool_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                medpool_1 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5)

                scores_w0 = medpool_0(scores_w0)
                scores_up = medpool_0(scores_up)

                feats_mem = []
                scores_mem = []
                for idx_k in range(len(w_au)):
                    for idx_ in range(len(kernel_sz_set)):
                        feats_ = self.user_norm_v2(
                            feats,
                            kernel_sz_set=kernel_sz_set[idx_],
                            w_a=w_au[idx_k][0],
                            w_u=w_au[idx_k][1],
                            ema=0.7,
                            eps=1e-5,
                        )
                        scores_w = self.weight_gen_unet2(feats_).sigmoid()
                        scores_w = medpool_0(scores_w)

                        mask_0 = scores_w - scores_w0 > error_thresh_0
                        mask_1 = scores_w - scores_up > error_thresh_1
                        mask_2 = scores_w - scores_w0 > 0
                        mask = (mask_0.float() * mask_1.float() * mask_2.float() > 0.1).float()
                        mask = mask_0.float()
                        mask = medpool_1(mask)

                        scores_lup = scores_up * (1 - mask) + scores_w * mask
                        mask = F.interpolate(mask, size=(feats_.size(-2), feats_.size(-1)),
                                             mode='bilinear', align_corners=False)
                        feats_l0 = feats_0 * (1 - mask) + feats_ * mask

                        feats_mem.append(feats_l0)
                        scores_mem.append(scores_lup)

                scores_mem_t = torch.cat(scores_mem, dim=1)
                scores_mem_t = scores_mem_t - torch.topk(scores_mem_t, 1, dim=1)[0][:, 0:]
                scores_mem_t = torch.clip(scores_mem_t * 1.2, min=0.0, max=0.8)
                scores_mem_t = torch.exp(scores_mem_t)
                weights = scores_mem_t / (scores_mem_t.sum(dim=1, keepdim=True) + 1e-5)
                weights = F.interpolate(weights, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)),
                                         mode='bilinear', align_corners=False)

                final_feats = 0.0
                for f_idx in range(len(feats_mem)):
                    final_feats = final_feats + feats_mem[f_idx] * weights[:, f_idx:f_idx + 1]

                output = self.render_net(final_feats)
                output_dict = {'feature_dict': [final_feats], 'weight_dict': scores_mem}
                return output, output_dict

            elif moe_mode == 7:
                # 空间分区域退化策略：雾天退化 + 夜晚退化 + 保留干净区域
                # 单步渲染，不使用多步迭代
                if moe_mark == 0:
                    # 空间分区域退化：雾天偏向
                    feats_raw = self.norm_back(feats)
                    feats_deg0, feats_deg1, feats_guide = spatial_fog_night_degrade(feats_raw)
                    attn_masks = torch.ones_like(feats_raw)
                    output, out_dicts = self.render_net(feats_deg0, feats_deg1, feats_guide)

                elif moe_mark == 1:
                    # 空间分区域退化：随机交换两版本位置
                    feats_raw = self.norm_back(feats)
                    feats_deg0, feats_deg1, feats_guide = spatial_fog_night_degrade(feats_raw)
                    attn_masks = torch.ones_like(feats_raw)
                    loc_mark = random.randint(0, 1)
                    if loc_mark == 0:
                        output, out_dicts = self.render_net(feats_deg1, feats_deg0, feats_guide)
                    else:
                        output, out_dicts = self.render_net(feats_deg0, feats_deg1, feats_guide)

                elif moe_mark == 2:
                    # 更强退化：增加退化强度
                    feats_raw = self.norm_back(feats)
                    feats_deg0, feats_deg1, feats_guide = spatial_fog_night_degrade(feats_raw)
                    # 二次退化增强
                    feats_deg0, _, _ = spatial_fog_night_degrade(feats_deg0)
                    feats_deg1, _, _ = spatial_fog_night_degrade(feats_deg1)
                    attn_masks = torch.ones_like(feats_raw)
                    output, out_dicts = self.render_net(feats_deg0, feats_deg1, feats_guide)

                else:
                    # 轻微退化：仅施加单种退化
                    feats_raw = self.norm_back(feats)
                    feats_deg0, feats_deg1, feats_guide = spatial_fog_night_degrade(feats_raw)
                    # 用原始特征作为guide，减少退化干扰
                    output, out_dicts = self.render_net(feats_deg0, feats_deg1, feats_raw)
                    attn_masks = torch.ones_like(feats_raw)

                output_dict = {'feature_dict':[None], 'weight_dict':[None], 'attn':out_dicts['attn'], 'attn_masks':attn_masks}
                return output, output_dict

            ################## wo transformation ####################
            elif moe_mode == 100:
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1, 1], w_a=1.0, w_u=1.0, ema=1.0)
                feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1) // 2),
                                          mode='bilinear', align_corners=False)
                scores_w0 = self.weight_gen_unet2(feats_0_w.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)),
                                          mode='bilinear', align_corners=False)
                output = feats_0
                output_dict = {'feature_dict': [output], 'weight_dict': [scores_w0], 'targets_s': targets}
                return feats, output_dict

            elif moe_mode == 101:
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1, 1], w_a=1.0, w_u=1.0, ema=1.0)

                feats_s, targets_s = self.random_scale_center_crop(feats_0, targets)

                self.traing_iter_scale_save = (self.traing_iter_scale_save + 1) % 20
                name = str(self.traing_iter_scale_save).zfill(4)
                file_name = './samples/training/' + name + '.png'
                self.visualize_and_save(feats_s[0], targets_s[0], file_name)

                scores_w0 = self.weight_gen_unet2(feats_s.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats.size(-2), feats.size(-1)),
                                          mode='bilinear', align_corners=False)
                output = feats_s
                # 可视化训练阶段夜间模拟后的输出（周期保存，避免频繁IO）
                self.traing_iter_save = (self.traing_iter_save + 1) % 20
                if self.traing_iter_save == 0 and targets_s is not None and len(targets_s) > 0:
                    file_name = './samples/training/night_sim/moe200_output.png'
                    self.visualize_and_save(output[0], targets_s[0], file_name, title='moe200_output')
                output_dict = {'feature_dict': [output], 'weight_dict': [scores_w0], 'targets_s': targets_s}
                return output, output_dict

            ################## global+local ####################
            elif moe_mode == 200:
                feats_aug = self._snorm_random_full_bc_aug(feats, p=1.0)
                # 训练阶段显式注入夜间退化，避免仅学到雾霾/亮度扰动。
                if random.random() < 0.65:
                    feats_aug = self.local_night_degrade(feats_aug)
                
                feats_g = self.global_norm(feats_aug)
                kernel_sz_set_t = [
                    [1, 1, 1.0],
                    [3, 3, 1.0],
                    [5, 5, 1.0],
                    #[15, 15, 1.0],
                ]
                feats_restore, output_dict, feats_mem = self._snorm_restore_from_aug(
                    feats_g,
                    kernel_sz_set_t=kernel_sz_set_t,
                )
                
                feats_rec = self.render_net(feats_restore)
                #feats_l = feats_restore
                
                
                feats_s, targets_s = self.random_scale_center_crop(feats_g, targets)

                scores_w0 = self.weight_gen_unet2(feats_s.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats.size(-2), feats.size(-1)),
                                          mode='bilinear', align_corners=False)
                output = feats_s
                
                # 在该分支返回前可视化output，按周期保存避免IO过重
                '''
                self.traing_iter_save = (self.traing_iter_save + 1) % 2
                if self.traing_iter_save == 0 and targets_s is not None and len(targets_s) > 0:
                    self.output_index += 1
                    file_name = f'./samples/training/night_sim/moe200_output_{self.output_index:06d}.png'
                    vis_our = output[0] * 0.5 + 0.5
                    self.visualize_and_save(vis_our, targets_s[0], file_name, title='moe200_output')
                '''
                output_dict = {
                    'feature_dict': [output],
                    'weight_dict': [scores_w0],
                    'targets_s': targets_s,
                    'feats_rec': feats_rec,
                    'feats_g': feats_g,
                    'feats_restore': feats_restore,
                }
                return output, output_dict

            elif moe_mode == 300:
                kernel_sz_set = [
                    [1, 1],
                    [3, 3],
                    [5, 5],
                    #[7, 7],
                    #[11, 7],
                    #[10, 20],
                    #[20, 10],
                ]

                #if random.random() < 0.5:
                #    feats_g = self.global_norm(feats)
                #else:
                #    feats_g = feats
                #feats_g = feats
                feats_g = self.global_norm(feats)
                #feats_rec = self.render_net(feats_g)
                sz_index_set = random.sample(range(0, len(kernel_sz_set)), 1)
                kernel_sz_set_u = kernel_sz_set[sz_index_set[0]]
                if random.random() < 0.8:
                    '''
                    feats_l = self.user_norm_bright_contrast(feats_g, mode=0,
                                                            kernel_sz_set=kernel_sz_set_u, ema=1.0,
                                                            #brightness_range=(0.35, 0.95),
                                                            #contrast_range=(0.5, 1.2)
                                                            )
                    '''

                    feats_l = self.user_norm_bright_contrast(
                        feats_g,
                        mode=0,
                        kernel_sz_set=kernel_sz_set_u,
                        ema=1.0,
                        w_a=1.0,
                        w_u=0.75,
                        brightness_range=(-0.45, 0.25),
                        contrast_range=(0.55, 1.15),
                    )
                else:
                    feats_l = feats_g

                #if random.random() < 0.65:
                #    feats_l = self.local_night_degrade(feats_l)

                #feats_s, targets_s = self.random_scale_center_crop(feats_l, targets)
                if random.random() < 0.5:
                    feats_s, targets_s = self.random_scale_center_crop(feats_l, targets)
                else:
                    feats_s = feats_l
                    targets_s = targets

                #feats_s = feats
                #targets_s = targets

                scores_w0 = self.weight_gen_unet2(feats_s.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats.size(-2), feats.size(-1)),
                                          mode='bilinear', align_corners=False)
                output = feats_s
                output_dict = {
                    'feature_dict': [output],
                    'weight_dict': [scores_w0],
                    'targets_s': targets_s,
                    'feats_rec': feats_rec,
                    'feats_g': feats_g,
                }
                return output, output_dict

        else:
            _ablation = os.environ.get('SNORM_ABLATION', 'full')

            feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.0)
            save_im0 = feats_0.clone()

            _kernel_mode = os.environ.get('SNORM_KERNEL_MODE', 'default')
            if _kernel_mode == 'default':
                kernel_sz_set = [
                    [3,3],[7,7],[11,11],[15,15],[19,19]
                ]
            elif _kernel_mode == 'dense_small':
                kernel_sz_set = [
                    [3,3],[5,5],[9,9],[13,13],[17,17]
                ]
            elif _kernel_mode == 'no_small':
                kernel_sz_set = [
                    [5,5],[9,9],[13,13],[17,17],[21,21]
                ]
            elif _kernel_mode == 'four':
                kernel_sz_set = [
                    [3,3],[7,7],[11,11],[15,15]
                ]
            elif _kernel_mode == 'three':
                kernel_sz_set = [
                    [3,3],[11,11],[19,19]
                ]
            elif _kernel_mode == 'six':
                kernel_sz_set = [
                    [3,3],[5,5],[7,7],[11,11],[15,15],[19,19]
                ]
            elif _kernel_mode == 'asym':
                kernel_sz_set = [
                    [3,5],[5,3],[7,11],[11,7],[13,13]
                ]
            else:
                kernel_sz_set = [
                    [3,3],[7,7],[11,11],[15,15],[19,19]
                ]

            w_au = [
                    [1.0, 1.0, 0.5],
            ]

            feats_mem = []
            for idx_k in range(len(w_au)):
                for idx_ in range(len(kernel_sz_set)):
                    _ema_eval = float(os.environ.get('SNORM_EMA', '0.85'))
                    feats_ = self.user_norm_v2(feats_0, kernel_sz_set=kernel_sz_set[idx_],
                                               w_a=w_au[idx_k][0], w_u=w_au[idx_k][1],
                                               ema=_ema_eval, eps=1e-5)
                    feats_ = self.norm_back(feats_)
                    feats_mem.append(feats_)

            feats_mem.append(self.norm_back(feats))

            save_im0 = self.norm_back(save_im0)

            _res_w = float(os.environ.get('SNORM_RES_W', '0.9'))
            _steps = int(os.environ.get('SNORM_STEPS', '6'))

            if _ablation == 'passthrough':
                output = feats
                output_dict = {'feature_dict': feats_mem, 'weight_dict': [None]}
                return output, output_dict

            if _ablation == 'norm_only':
                output = save_im0
            elif _ablation == 'no_ahdr':
                output = save_im0
            elif _ablation == 'single_ahdr':
                output = self.render_net(feats_mem[0], save_im0, feats_mem[1])
                output = output * _res_w + save_im0 * (1 - _res_w)
            else:
                _debug_save = os.environ.get('SNORM_DEBUG_SAVE', '0') == '1'
                _saved = getattr(self, '_debug_saved', 0)
                _max_save = int(os.environ.get('SNORM_DEBUG_MAX', '10'))
                _do_debug = _debug_save and _saved < _max_save
                _all_steps = {}
                _all_refs = {}

                output = self.render_net(feats_mem[0], save_im0, feats_mem[1])
                output = output * _res_w + save_im0 * (1 - _res_w)
                if _do_debug:
                    _all_steps[1] = output.clone()
                    _all_refs[1] = feats_mem[0].clone()

                if _steps >= 2:
                    output = self.render_net(feats_mem[1], output, feats_mem[3])
                    output = output * _res_w + save_im0 * (1 - _res_w)
                    if _do_debug:
                        _all_steps[2] = output.clone()
                        _all_refs[2] = feats_mem[1].clone()

                if _steps >= 3:
                    output = self.render_net(feats_mem[2], output, feats_mem[3])
                    output = output * _res_w + save_im0 * (1 - _res_w)
                    if _do_debug:
                        _all_steps[3] = output.clone()
                        _all_refs[3] = feats_mem[2].clone()

                if _steps >= 4:
                    output = self.render_net(feats_mem[3], output, feats_mem[3])
                    output = output * _res_w + save_im0 * (1 - _res_w)
                    if _do_debug:
                        _all_steps[4] = output.clone()
                        _all_refs[4] = feats_mem[3].clone()

                if _steps >= 5:
                    output = self.render_net(feats_mem[4], output, feats_mem[3])
                    output = output * _res_w + save_im0 * (1 - _res_w)
                    if _do_debug:
                        _all_steps[5] = output.clone()
                        _all_refs[5] = feats_mem[4].clone()

                if _steps >= 6:
                    _step6_src = os.environ.get('SNORM_STEP6_SRC', '5')
                    _s6_idx = int(_step6_src)
                    _s6_ref = feats_mem[min(_s6_idx, len(feats_mem) - 1)]
                    if _do_debug:
                        _all_refs[6] = _s6_ref.clone()
                    output = self.render_net(_s6_ref, output, feats_mem[3])
                    output = output * _res_w + save_im0 * (1 - _res_w)
                    if _do_debug:
                        _all_steps[6] = output.clone()

                if _do_debug:
                    import pickle
                    _db_dir = os.environ.get('SNORM_DEBUG_DIR', './debug_step6')
                    os.makedirs(_db_dir, exist_ok=True)
                    def _stats(t):
                        tc = t.detach().cpu()
                        return {
                            'mean': float(tc.mean().item()),
                            'std': float(tc.std().item()),
                            'min': float(tc.min().item()),
                            'max': float(tc.max().item()),
                            'shape': list(tc.shape),
                        }
                    _db = {
                        'save_im0_stats': _stats(save_im0),
                        'kernel_sz_set': kernel_sz_set,
                        'step6_src_idx': int(os.environ.get('SNORM_STEP6_SRC', '5')),
                        'step_imgs': {},
                        'ref_imgs': {},
                    }
                    for _k, _v in _all_steps.items():
                        _db['step_imgs'][_k] = _v.detach().cpu()
                        _db[f'step{_k}_stats'] = _stats(_v)
                    for _k, _v in _all_refs.items():
                        _db['ref_imgs'][_k] = _v.detach().cpu()
                        _db[f'ref{_k}_stats'] = _stats(_v)
                    _all_mem_stats = {}
                    for _ki in range(len(feats_mem)):
                        _all_mem_stats[f'mem_{_ki}'] = _stats(feats_mem[_ki])
                    _db['mem_stats'] = _all_mem_stats
                    with open(os.path.join(_db_dir, f'step6_data_{_saved:04d}.pkl'), 'wb') as _f:
                        pickle.dump(_db, _f)
                    self._debug_saved = _saved + 1

            output = self.norm_forward(output)
            output_dict = {'feature_dict': feats_mem, 'weight_dict': [None]}
            return output, output_dict

    ############################
    # 下面是各种归一化 / 变换算子
    ############################
    def norm_back(self, feats):
        feats = feats * 0.5 + 0.5
        return feats
    
    def norm_forward(self, feats):
        feats = (feats - 0.5) / 0.5
        return feats
    
    def calc_std_em(self, feat, eps=1e-8):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]

        feat = feat ** 2
        feat = (self.pool(feat) + eps).sqrt()
        feats = F.interpolate(feat, size=(size[-2], size[-1]), mode='bicubic', align_corners=False)
        return feats

    def update_norm_parameters(self, feats_, is_update=False):
        #feats = feats_ * 0.5 + 0.5
        feats = feats_
        if self.memory_index < self.memory_len:
            feats_mean, feats_std = calc_mean_std(feats)
            feats_mean = feats_mean.view(feats_mean.size(0), feats_mean.size(1))
            feats_std = feats_std.view(feats_std.size(0), feats_std.size(1))

            self.memory_aplpha_a[self.memory_index] = feats_std.mean(dim=0).cpu().detach().numpy()
            self.memory_aplpha_u[self.memory_index] = feats_mean.mean(dim=0).cpu().detach().numpy()
            self.memory_index += 1

            alpha_ba = torch.from_numpy(self.memory_aplpha_a[0:self.memory_index].sum(axis=0) / self.memory_index)
            alpha_bu = torch.from_numpy(self.memory_aplpha_u[0:self.memory_index].sum(axis=0) / self.memory_index)
            alpha_ba = alpha_ba.to(feats.device)
            alpha_bu = alpha_bu.to(feats.device)
            self.alpha_ba.data = self.alpha_ba.data.clone().detach() * 0.0 + alpha_ba.clone().detach()
            self.alpha_bu.data = self.alpha_bu.data.clone().detach() * 0.0 + alpha_bu.clone().detach()
        else:
            feats_mean, feats_std = calc_mean_std(feats)
            feats_mean = feats_mean.view(feats_mean.size(0), feats_mean.size(1))
            feats_std = feats_std.view(feats_std.size(0), feats_std.size(1))

            self.alpha_ba.data = self.alpha_ba.data.clone().detach() * self.ema_rate + \
                (1 - self.ema_rate) * feats_std.mean(dim=0).data.clone().detach()
            self.alpha_bu.data = self.alpha_bu.data.clone().detach() * self.ema_rate + \
                (1 - self.ema_rate) * feats_mean.mean(dim=0).data.clone().detach()
            self.memory_index = self.memory_len - 1
            self.memory_index += 1

        return

    def compute_norm_parameters(self, device=None):
        mem_count = max(self.memory_index, 1)
        alpha_ba = torch.from_numpy(self.memory_aplpha_a[0:mem_count].sum(axis=0) / mem_count)
        alpha_bu = torch.from_numpy(self.memory_aplpha_u[0:mem_count].sum(axis=0) / mem_count)
        alpha_ba = alpha_ba.to(device)
        alpha_bu = alpha_bu.to(device)
        self.alpha_ba.data = self.alpha_ba.data.clone().detach() * 0.0 + alpha_ba.clone().detach()
        self.alpha_bu.data = self.alpha_bu.data.clone().detach() * 0.0 + alpha_bu.clone().detach()
        return

    def save_img(self, img_tensor_, labels=None, bboxes=None, idx='0', outputs='./WZ-Evals', use_norm=False):
        import os
        import cv2
        if not os.path.exists(outputs):
            os.makedirs(outputs)

        img_tensor = copy.deepcopy(img_tensor_)
        device = img_tensor.device

        lim = img_tensor[0].permute(1, 2, 0).contiguous()
        lim = lim * 0.5 + 0.5
        cv_im = lim.cpu().detach().numpy()
        cv_im = np.array(cv_im * 255, np.int32)
        if bboxes is not None:
            mbbs = bboxes.cpu().detach().numpy()
            for ii in range(mbbs.shape[0]):
                cv2.rectangle(cv_im, (int(mbbs[ii, 0]), int(mbbs[ii, 1])),
                              (int(mbbs[ii, 2]), int(mbbs[ii, 3])), (0, 0, 255), 2)

        cv2.imwrite(outputs + '/' + idx + '.jpg', cv_im)

    ########################################
    def user_norm_v2(self, feats, mode=0, kernel_sz_set=[1, 1], w_a=1.0, w_u=1.0, eps=1e-8, ema=0.1):
        pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set[0])
        feats_pool = pool_func(feats)

        feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)),
                                   mode='bicubic', align_corners=False)
        pfeats = feats - feats_pool

        sfeats = pfeats ** 2
        sfeats = (pool_func(sfeats) + eps).sqrt()

        sfeats = F.interpolate(sfeats, size=(feats.size(-2), feats.size(-1)),
                               mode='bicubic', align_corners=False)
        alpha_ba = self.alpha_ba.clone().detach()
        alpha_bu = self.alpha_bu.clone().detach()

        feats_mean, feats_std = calc_mean_std(feats)
        ba_ema_rate = ema
        sfeats = sfeats * ba_ema_rate + (1 - ba_ema_rate) * feats_std
        alpha_bu = alpha_bu[None, :, None, None]

        nfeats = w_a * pfeats * alpha_ba[None, :, None, None] / (sfeats + 1e-3) + w_u * alpha_bu
        nfeats = torch.clip(nfeats, min=-1.0, max=1.0)
        return nfeats.float()

    def global_norm(self, feats_):
        #feats = feats_ * 0.5 + 0.5
        feats = feats_
        feats_mean, feats_std = calc_mean_std(feats)
        alpha_ba = self.alpha_ba.clone().detach()
        alpha_bu = self.alpha_bu.clone().detach()

        nfeats = (feats - feats_mean) * alpha_ba[None, :, None, None] / (feats_std + 1e-5) + \
            alpha_bu[None, :, None, None]
        
        #nfeats = (nfeats - 0.5) / 0.5
        nfeats = torch.clip(nfeats, min=-1.0, max=1.0)
        return nfeats.float()

    def _snorm_random_full_bc_aug(self, feats_g, p=0.6):
        """
        基于 user_norm_bright_contrast 的整图随机增强（kernel=[1,1]）。
        """
        kernel_sz_set = [
            [1, 1],
            [3, 3],
            [5, 5],
            [7, 7],
            #[11, 7],
            #[10, 20],
            [20, 10],
        ]
        if random.random() < p:
            sz_index_set = random.sample(range(0, len(kernel_sz_set)), 1)
            kernel_sz_set_u = kernel_sz_set[sz_index_set[0]]

            return self.user_norm_bright_contrast(
                feats_g,
                mode=0,
                kernel_sz_set=kernel_sz_set_u,
                ema=0.8,
                w_a=1.0,
                w_u=1.0,
                brightness_range=(-0.55, 0.35),
                contrast_range=(0.55, 1.45),
            )
        return feats_g

    def _snorm_restore_from_aug(
        self,
        feats,
        beta_0=2.5,
        mask_soft_temp=8.0,
        error_thresh_1=0.14,
        topk_num=2,
        kernel_sz_set_t=None,
        draw_flag=False,
    ):
        """
        参考 481-644 的融合流程，将增强特征与基准特征融合得到复原结果。
        """

        #SNORM_ERROR_THRESH_1=0.14 | full tag b02.5_b10.5_mst8.0_tss5.0_tk8_ema0.92_et0-0.02_et10.14
        # 推理融合超参：默认与下述一致；optimize_snorm_params_032900.py 通过环境变量覆盖（控制变量法扫参）
        beta_0 = float(os.getenv("SNORM_BETA_0", "6"))
        mask_soft_temp = float(os.getenv("SNORM_MASK_SOFT_TEMP", "20.0"))
        #topk_num = int(float(os.getenv("SNORM_TOPK_NUM", "8")))
        topk_num = int(float(os.getenv("SNORM_TOPK_NUM", "4")))
        norm_ema = float(os.getenv("SNORM_NORM_EMA", "0.92"))

        #error_thresh_0 = float(os.getenv("SNORM_ERROR_THRESH_0", "0.02"))
        #error_thresh_1 = float(os.getenv("SNORM_ERROR_THRESH_1", "0.14"))
        error_thresh_1 = float(os.getenv("SNORM_ERROR_THRESH_1", "0.1"))

        medpool_0 = MedianPool2d(kernel_size=3, stride=1, padding=1)
        medpool_1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        feats_mem = []
        scores_mem = []
        feats_all = [feats]
        masks_all = []

        # 1) 原有的多尺度 user_norm_v2 变换
        for idx_ in range(len(kernel_sz_set_t)):
            kernel_sz = [kernel_sz_set_t[idx_][0], kernel_sz_set_t[idx_][1]]
            w_u = kernel_sz_set_t[idx_][2]
            feats_ = self.user_norm_v2(
                feats,
                kernel_sz_set=kernel_sz,
                w_a=1.0,
                w_u=w_u,
                ema=norm_ema,
                eps=1e-5,
            )
            feats_all.append(feats_)

        # 夜景增强：放在 norm 循环外，只对 feats 算一次；原先写在循环内会对同一 feats 重复 append 4 份相同结果
        kernel_sz_set_gm = [91]
        gamma_min = 0.3
        gamma_max = 2.0
        for kernel_sz in kernel_sz_set_gm:
            feats_gamma = self.local_gamma_night(
                feats,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                kernel_sz=kernel_sz,
            )
            feats_all.append(feats_gamma)

        medpool_1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        feats_mem = []
        scores_mem = []

        B, C, H, W = feats.size()
        feats_all = torch.stack(feats_all, dim=1)
        feats_all = feats_all.view(-1, C, H, W)
        scores_all = self.weight_gen_unet2(feats_all).softmax(dim=1)[:, 0:1]
        Hs, Ws = int(scores_all.size(-2)), int(scores_all.size(-1))

        feats_all = feats_all.view(B, -1, C, H, W)
        scores_all = scores_all.view(B, -1, 1, Hs, Ws)

        scores_w0 = scores_all[:, 0]
        feats_0 = feats_all[:, 0]

        # feats_all / scores_all: [B, V, ...]，循环维度应是视图维 V（不是 batch 维 B）
        n_views_all = int(feats_all.size(1))
        for idx_ in range(n_views_all):
            scores_w = scores_all[:, idx_]
            feats_ = feats_all[:, idx_]

            delta = (scores_w - scores_w0) #* (scores_w - error_thresh_1)
            mask = torch.sigmoid(mask_soft_temp * delta)
            scores_w_l = medpool_1(mask * scores_w)
            mask = medpool_1(mask)
            scores_w = scores_w_l * mask + (1 - mask) * scores_w

            scores_lup = scores_w0 * (1 - mask) + scores_w * mask
            mask_f = F.interpolate(
                mask,
                size=(feats_.size(-2), feats_.size(-1)),
                mode='bicubic',
                align_corners=False,
            )
            feats_l0 = feats_0 * (1 - mask_f) + feats_ * mask_f

            feats_mem.append(feats_l0.clone())
            scores_mem.append(scores_lup.clone())

        scores_mem_t = torch.cat(scores_mem, dim=1)  # [B, V, H, W]
        n_views = int(scores_mem_t.size(1))
        topk_sel = min(topk_num, max(1, n_views))
        top_scores, top_idx = torch.topk(scores_mem_t, k=topk_sel, dim=1)
        top_scores = top_scores - top_scores[:, -1:]
        top_weights = torch.softmax(top_scores * beta_0, dim=1)
        top_weights = F.interpolate(
            top_weights,
            size=(feats_mem[0].size(-2), feats_mem[0].size(-1)),
            mode='bicubic',
            align_corners=False,
        )
        top_weights = top_weights / (top_weights.sum(dim=1, keepdim=True) + 1e-6)
        top_idx = F.interpolate(
            top_idx.float(),
            size=(feats_mem[0].size(-2), feats_mem[0].size(-1)),
            mode='nearest',
        ).long().clamp(0, n_views - 1)

        feats_mem_t = torch.stack(feats_mem, dim=1)  # [B, V, C, H, W]
        c_dim = int(feats_mem_t.size(2))
        top_idx_feat = top_idx.unsqueeze(2).expand(-1, -1, c_dim, -1, -1)
        top_feats = torch.gather(feats_mem_t, dim=1, index=top_idx_feat)
        feats_restore = (top_feats * top_weights.unsqueeze(2)).sum(dim=1)

        output_dict = {'feature_dict': [feats_restore], 'weight_dict': scores_mem}
        
        return feats_restore, output_dict, feats_mem

    def user_norm_bright_contrast(self, feats_, mode=0, kernel_sz_set=[1, 1],
                                  w_a=1.0, w_u=1.0, eps=1e-8, ema=1.0,
                                  brightness_range=(-1, 1.0), contrast_range=(0.2, 1.5)):
        #feats = feats_ * 0.5 + 0.5
        feats = feats_
        pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set)
        feats_pool = pool_func(feats)

        feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)),
                                   mode='bicubic', align_corners=False)
        pfeats = feats - feats_pool

        sfeats = pfeats ** 2
        sfeats = (pool_func(sfeats) + eps).sqrt()

        sfeats = F.interpolate(sfeats, size=(feats.size(-2), feats.size(-1)),
                               mode='bicubic', align_corners=False)
        alpha_ba = self.alpha_ba.clone().detach()
        alpha_bu = self.alpha_bu.clone().detach()

        _, feats_std = calc_mean_std(feats)
        ba_ema_rate = ema
        sfeats = sfeats * ba_ema_rate + (1 - ba_ema_rate) * feats_std
        alpha_bu_bc = alpha_bu[None, :, None, None]
        alpha_ba_bc = alpha_ba[None, :, None, None]

        device = feats.device
        B, C = feats.size(0), feats.size(1)
        brightness = torch.empty(B, 1, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*brightness_range)
        contrast = torch.empty(B, 1, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*contrast_range)
        brightness = brightness.expand(B, C, kernel_sz_set[0], kernel_sz_set[1])
        contrast = contrast.expand(B, C, kernel_sz_set[0], kernel_sz_set[1])
        brightness = F.interpolate(brightness, size=(feats.size(-2), feats.size(-1)),
                                   mode='bicubic', align_corners=False)
        contrast = F.interpolate(contrast, size=(feats.size(-2), feats.size(-1)),
                                 mode='bicubic', align_corners=False)
        contrast = torch.clip(contrast, min=0.1, max=2.0)

        # mode==1: 旧版仅对比度 + 低频，便于对照；默认 mode==0 走完整亮度/对比度/自适应尺度
        if mode == 1:
            nfeats = contrast * pfeats + feats_pool
        else:
            high = w_a * contrast * pfeats * alpha_ba_bc / (sfeats + 1e-3)
            nfeats = high + feats_pool + w_u * brightness * alpha_bu_bc

        nfeats = torch.clip(nfeats, min=-1.0, max=1.0)
        return nfeats.float()

    def random_scale_center_crop(
        self,
        images,
        targets,
        scale_range=(0.6, 1.6),
    ):
        B, C, H, W = images.shape
        device = images.device

        scale = torch.empty(1).uniform_(*scale_range).item()
        new_h = int(H * scale)
        new_w = int(W * scale)

        images_scaled = F.interpolate(
            images, size=(new_h, new_w),
            mode="bilinear", align_corners=False
        )

        pad_h = max(H - new_h, 0)
        pad_w = max(W - new_w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            images_scaled = F.pad(
                images_scaled,
                (pad_left, pad_right, pad_top, pad_bottom),
                value=0.0
            )

        padded_h, padded_w = images_scaled.shape[-2:]

        y0 = (padded_h - H) // 2
        x0 = (padded_w - W) // 2
        y1 = y0 + H
        x1 = x0 + W

        images_crop = images_scaled[:, :, y0:y1, x0:x1]

        new_targets = []
        images_crops = []
        for tgt_idx in range(len(targets)):
            tgt_o = targets[tgt_idx]
            tgt = copy.deepcopy(tgt_o)

            if "boxes" in tgt:
                boxes = tgt["boxes"].clone()

                boxes[:, [0, 2]] *= scale
                boxes[:, [1, 3]] *= scale

                boxes[:, [0, 2]] += pad_left
                boxes[:, [1, 3]] += pad_top

                boxes[:, [0, 2]] -= x0
                boxes[:, [1, 3]] -= y0

                boxes[:, 0::2].clamp_(0, W)
                boxes[:, 1::2].clamp_(0, H)

                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                tgt["boxes"] = boxes[keep]
                tgt["labels"] = tgt["labels"][keep]

            if "masks" in tgt:
                masks = tgt["masks"].float()

                masks = F.interpolate(
                    masks.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="nearest"
                ).squeeze(1)

                if pad_h > 0 or pad_w > 0:
                    masks = F.pad(
                        masks,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        value=0.0
                    )

                masks = masks[:, y0:y1, x0:x1]
                tgt["masks"] = masks

            if tgt["boxes"].size(0) > 0:
                images_crops.append(images_crop[tgt_idx])
                new_targets.append(tgt)
            else:
                images_crops.append(images[tgt_idx])
                new_targets.append(tgt_o)

        images_crops = torch.stack(images_crops, dim=0)
        return images_crops, new_targets


    def local_night_degrade(
        self,
        feats,
        gamma_range=(1.3, 2.4),
        brightness_range=(0.4, 0.85),
        noise_std_range=(0.0, 0.03),
        color_shift_range=(-0.06, 0.03),
    ):
        """
        夜间退化模拟（训练增强）：
        - gamma 压暗中高亮区域；
        - brightness 进一步下压整体亮度；
        - 轻量颜色偏移 + 传感器噪声，贴近夜间成像分布。
        """
        x = feats * 0.5 + 0.5
        x = torch.clamp(x, 0.0, 1.0)

        b = x.size(0)
        device = x.device

        gamma = torch.empty(b, 1, 1, 1, device=device).uniform_(*gamma_range)
        brightness = torch.empty(b, 1, 1, 1, device=device).uniform_(*brightness_range)
        x = torch.clamp((x ** gamma) * brightness, 0.0, 1.0)

        color_shift = torch.empty(b, 3, 1, 1, device=device).uniform_(*color_shift_range)
        blue_boost = torch.empty(b, 1, 1, 1, device=device).uniform_(0.0, 0.05)
        color_shift[:, 2:3] = color_shift[:, 2:3] + blue_boost
        x = torch.clamp(x + color_shift, 0.0, 1.0)

        noise_std = torch.empty(b, 1, 1, 1, device=device).uniform_(*noise_std_range)
        x = torch.clamp(x + torch.randn_like(x) * noise_std, 0.0, 1.0)

        y = x * 2.0 - 1.0
        return torch.clamp(y, -1.0, 1.0).float()

    ############################
    # 新增：Retinex 风格局部变换 + 自适应 gamma 夜景增强
    ############################

    def local_retinex(self, feats, kernel_sz=31, eps=1e-3):
        """
        简化 Retinex：
        - 输入 feats ~ [-1,1]，先映射到 [0,1]
        - 估计局部照度 I（avg pool）
        - 在 log 域做 R = log(1+x) - log(1+I)
        - 再做一次 mean/std 归一化，映射回 [-1,1]
        """
        B, C, H, W = feats.size()
        x = feats * 0.5 + 0.5
        x = torch.clamp(x, 0.0, 1.0)

        pool = nn.AvgPool2d(kernel_size=kernel_sz, stride=1, padding=kernel_sz // 2)
        illum = pool(x)
        illum = torch.clamp(illum, eps, 1.0)

        r = torch.log1p(x) - torch.log1p(illum)

        mean, std = calc_mean_std(r)
        r_norm = (r - mean) / (std + 1e-3)
        r_norm = torch.tanh(r_norm)
        return r_norm.float()

    def local_gamma_night(self, feats, gamma_min=0.4, gamma_max=1.4, eps=1e-3, kernel_sz = 15):
        """
        夜景增强（替代纯 gamma）：
        - 先用局部平均估计照度 illum
        - 计算“重照明”增益 gain = (target / illum) ** alpha，并做上限约束，避免过曝
        - 仅对暗部更强地应用增益（由 illum 生成的暗部权重）
        其中 alpha 由照度推导出的强度图控制（纯算法式，不使用可学习参数）。

        备注：为保持接口兼容，沿用 gamma_min/gamma_max 作为 alpha 的取值范围（并不再直接做幂次 gamma）。
        """
        x = feats * 0.5 + 0.5
        x = torch.clamp(x, 0.0, 1.0)

        # 局部照度估计（更稳：对噪声不敏感、不会像幂次那样拉爆高光）
        pool = nn.AvgPool2d(kernel_size=kernel_sz, stride=1, padding=kernel_sz // 2)
        illum = torch.clamp(pool(x), eps, 1.0)

        # 由照度生成“增强强度图”：越暗越强（不使用可学习参数）
        # strength in [0,1]
        strength = torch.clamp((0.65 - illum) / 0.65, 0.0, 1.0)
        alpha = gamma_min + (gamma_max - gamma_min) * strength  # [B,1,H,W]

        # 暗部权重：illum 越暗，权重越大
        dark_w = torch.clamp(1.0 - illum, 0.0, 1.0) ** 1.5

        target = 0.60
        gain = (target / illum) ** alpha
        gain = torch.clamp(gain, 1.0, 4.0)

        x_relit = x * (1.0 + dark_w * (gain - 1.0))
        x_relit = torch.clamp(x_relit, 0.0, 1.0)

        # 轻量对比度提升（增强后压一压灰雾感）
        contrast = 1.0 + 0.25 * strength
        x_relit = torch.clamp((x_relit - 0.5) * contrast + 0.5, 0.0, 1.0)

        y = x_relit * 2.0 - 1.0
        return torch.clamp(y, -1.0, 1.0).float()

    def _snorm_restore_from_aug(
        self,
        feats,
        beta_0=2.5,
        mask_soft_temp=8.0,
        error_thresh_1=0.14,
        topk_num=2,
        kernel_sz_set_t=None,
        draw_flag=False,
    ):
        """
        多尺度特征融合函数：
        1) 多尺度 user_norm_v2 变换
        2) local_gamma_night 夜景增强
        所有候选通过 weight_gen_unet2 打分，Top-K Softmax 加权融合。
        """

        beta_0 = float(os.getenv("SNORM_BETA_0", "6"))
        mask_soft_temp = float(os.getenv("SNORM_MASK_SOFT_TEMP", "20.0"))
        topk_num = int(float(os.getenv("SNORM_TOPK_NUM", "4")))
        norm_ema = float(os.getenv("SNORM_NORM_EMA", "0.92"))
        error_thresh_1 = float(os.getenv("SNORM_ERROR_THRESH_1", "0.1"))

        medpool_0 = MedianPool2d(kernel_size=3, stride=1, padding=1)
        medpool_1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        feats_mem = []
        scores_mem = []
        feats_all = [feats]
        masks_all = []

        # 1) 多尺度 user_norm_v2 变换
        for idx_ in range(len(kernel_sz_set_t)):
            kernel_sz = [kernel_sz_set_t[idx_][0], kernel_sz_set_t[idx_][1]]
            w_u = kernel_sz_set_t[idx_][2]
            feats_ = self.user_norm_v2(
                feats,
                kernel_sz_set=kernel_sz,
                w_a=1.0,
                w_u=w_u,
                ema=norm_ema,
                eps=1e-5,
            )
            feats_all.append(feats_)

        # 2) local_gamma_night 夜景增强
        kernel_sz_set_gm = [91]
        gamma_min = 0.3
        gamma_max = 2.0
        for kernel_sz in kernel_sz_set_gm:
            feats_gamma = self.local_gamma_night(
                feats,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                kernel_sz=kernel_sz,
            )
            feats_all.append(feats_gamma)

        medpool_1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        feats_mem = []
        scores_mem = []

        B, C, H, W = feats.size()
        feats_all = torch.stack(feats_all, dim=1)
        feats_all = feats_all.view(-1, C, H, W)
        scores_all = self.weight_gen_unet2(feats_all).softmax(dim=1)[:, 0:1]
        Hs, Ws = int(scores_all.size(-2)), int(scores_all.size(-1))

        feats_all = feats_all.view(B, -1, C, H, W)
        scores_all = scores_all.view(B, -1, 1, Hs, Ws)

        scores_w0 = scores_all[:, 0]
        feats_0 = feats_all[:, 0]

        n_views_all = int(feats_all.size(1))
        for idx_ in range(n_views_all):
            scores_w = scores_all[:, idx_]
            feats_ = feats_all[:, idx_]

            delta = (scores_w - scores_w0)
            mask = torch.sigmoid(mask_soft_temp * delta)
            scores_w_l = medpool_1(mask * scores_w)
            mask = medpool_1(mask)
            scores_w = scores_w_l * mask + (1 - mask) * scores_w

            scores_lup = scores_w0 * (1 - mask) + scores_w * mask
            mask_f = F.interpolate(
                mask,
                size=(feats_.size(-2), feats_.size(-1)),
                mode='bicubic',
                align_corners=False,
            )
            feats_l0 = feats_0 * (1 - mask_f) + feats_ * mask_f

            feats_mem.append(feats_l0.clone())
            scores_mem.append(scores_lup.clone())

        scores_mem_t = torch.cat(scores_mem, dim=1)
        n_views = int(scores_mem_t.size(1))
        topk_sel = min(topk_num, max(1, n_views))
        top_scores, top_idx = torch.topk(scores_mem_t, k=topk_sel, dim=1)
        top_scores = top_scores - top_scores[:, -1:]
        top_weights = torch.softmax(top_scores * beta_0, dim=1)
        top_weights = F.interpolate(
            top_weights,
            size=(feats_mem[0].size(-2), feats_mem[0].size(-1)),
            mode='bicubic',
            align_corners=False,
        )
        top_weights = top_weights / (top_weights.sum(dim=1, keepdim=True) + 1e-6)
        top_idx = F.interpolate(
            top_idx.float(),
            size=(feats_mem[0].size(-2), feats_mem[0].size(-1)),
            mode='nearest',
        ).long().clamp(0, n_views - 1)

        feats_mem_t = torch.stack(feats_mem, dim=1)
        c_dim = int(feats_mem_t.size(2))
        top_idx_feat = top_idx.unsqueeze(2).expand(-1, -1, c_dim, -1, -1)
        top_feats = torch.gather(feats_mem_t, dim=1, index=top_idx_feat)
        feats_restore = (top_feats * top_weights.unsqueeze(2)).sum(dim=1)

        output_dict = {'feature_dict': [feats_restore], 'weight_dict': scores_mem}

        return feats_restore, output_dict, feats_mem

    def visualize_and_save(
        self,
        image_x,
        target,
        save_path,
        title=None,
        dpi=150
    ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        image = image_x * 0.5 + 0.5
        img = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis("off")

        if title is not None:
            ax.set_title(title)

        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=2
                )
            )
            ax.text(
                x1, y1 - 2,
                str(int(label)),
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
        plt.close(fig)


def calc_mean_std(feat, eps=1e-8):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std_batch(feat, eps=1e-8):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.permute(1, 0, 2, 3).view(C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std