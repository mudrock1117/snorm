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
from detection.modeling.unet.gate_score import * #GatedScore, GatedScore_S8, GatedScore_S16
from detection.modeling.unet.Irender import * 
from detection.modeling.unet.MedianPool2d import MedianPool2d
from detection.modeling.conV.tconV import tconV

from detection.modeling.HDR.OURS.OHDR import AHDR
from detection.modeling.norm.lib_pert import *    
from detection.modeling.render.integrated_adaptive_render import IntegratedAdaptiveRender

from detection.modeling.render.adaptive_render import AdaptiveRenderNetwork
# 增强版自适应渲染lazy import（避免mamba_ssm兼容性问题）
EnhancedAdaptiveRenderNetwork = None

# ExpoMamba算子lazy import（避免mamba_ssm兼容性问题）
_expmamba_operator = None
def get_expmamba_operator_lazy():
    global _expmamba_operator
    if _expmamba_operator is None:
        try:
            from expmamba_operator import get_expmamba_operator
            _expmamba_operator = get_expmamba_operator()
        except Exception as e:
            print(f"⚠️ ExpoMamba加载失败: {e}")
            _expmamba_operator = None
    return _expmamba_operator

#from torchvision.transforms import v2
from torchvision import transforms as v2

class SNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_a = None #nn.Parameter(torch.ones(1), requires_grad=True)
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
        #self.weight_gen_unet2 = GatedScore_S16(3, 1) #GatedScore(1024, 1)#GatedScore_S8(512, 1)
        #self.weight_gen_unet2 = GatedScore_S8(3, 1)
        self.weight_gen_unet2 = GatedScore_SV8(3, 1)

        #self.render_net = RenderNet(3, 1)
        #self.render_net = AHDR(3, 3, 128, 32)
        #self.render_net = AHDR(3,64)

        # 使用整合版自适应渲染网络
        # 三输入模式：兼容原项目 + 局部最优选择
        self.render_net = AHDR(in_ch=3, base_ch=64)

        ###########################################
        self.output_index = 0
        self.medpool = MedianPool2d(kernel_size=3, stride=1, padding=1)
        
        self.iter_save = 0
        self.iter_training_save = 0
        self.iter_global_save = 0
        self.iter_test_save = 0
        
        # ExpoMamba算子初始化
        self.expmamba_operator = None
        
        # 自适应渲染网络初始化（多渲染效果生成）
        self.adaptive_render_net = None
        self.use_adaptive_render = False
        
        # 增强版自适应渲染网络（夜间+雾霾操作）
        self.enhanced_adaptive_render_net = None
        self.use_enhanced_render = False

    def forward(self, feats, use_moe=False, use_moe_fusion=False, moe_mode=0, moe_mark=0):
        # 优先检查是否使用ExpoMamba渲染
        if moe_mode == 7 and moe_mark == 8:
            # ExpoMamba渲染模式
            if self.expmamba_operator is None:
                self.expmamba_operator = get_expmamba_operator_lazy()

            # 空间转换：标准化空间 -> 原始空间
            input_tensor = self.norm_back(feats)

            # ExpoMamba输出
            expomamba_out = self.expmamba_operator.enhance_tensor(input_tensor)
            expomamba_output = self.norm_forward(expomamba_out)

            # 原始渲染输出
            original_output = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=1.0)

            # 加权融合（alpha=0.55：ExpoMamba权重0.55，原始渲染权重0.45）最佳配置
            alpha = 0.55
            output = alpha * expomamba_output + (1 - alpha) * original_output

            attn_masks = torch.ones_like(feats)
            output_dict = {'feature_dict':[None], 'weight_dict':[None],
                          'attn':attn_masks, 'attn_masks':attn_masks}
            return output, output_dict
            
        # 增强版自适应渲染模式（暂时禁用，避免mamba_ssm兼容性问题）
        if moe_mode == 7 and moe_mark == 10:
            # 使用原始渲染方式作为替代
            print("⚠️ 增强版自适应渲染暂时不可用，使用原始渲染方式")
            output = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=1.0)
            output_dict = {'feature_dict':[output], 'weight_dict':[torch.ones_like(output)]}
            return output, output_dict
            
        ##################segnet####################
        if self.training:
            #global
            if moe_mode == 0:
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=1.0)
                feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
                scores_w0 = self.weight_gen_unet2(feats_0_w.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)), mode='bilinear')
                output = feats_0
                output_dict = {'feature_dict':[output], 'weight_dict':[scores_w0]}
                
                #self.iter_save = (self.iter_save + 1) % 2000
                ###draw
                self.iter_global_save = (self.iter_global_save + 1) % 2000
                if self.iter_global_save % 40 == 0:
                    vis_images_all = torch.zeros((1, output.size(1), 4 * output.size(-2), output.size(-1))).to(output.device)
                    vis_images_all[:, :, 0:output.size(-2)] = feats[0:1]
                    vis_images_all[:, :, output.size(-2):output.size(-2)*2] = feats_0[0:1]
                    #vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = (output[0:1] - 0.5) / 0.5
                    #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = final_mask[0:1] 
                    #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output * mask)[0:1]

                    import torchvision.utils as vutils
                
                    name = str(self.iter_global_save).zfill(4)    
                    vutils.save_image(vis_images_all.data, './samples/global/' + name +'.jpg', nrow=12, value_range=(-1, 1), normalize=True)
                    
                return output, output_dict
            
            #per local
            elif moe_mode == 1:
                kernel_sz_set = [
                    [3,5],
                    [5,3],
                    [7,11],
                    [11,7],
                    [10, 20],
                    [20, 10],
                ]
                sz_index_set = random.sample(range(0, len(kernel_sz_set)), 1)
                kernel_sz_set_u = kernel_sz_set[sz_index_set[0]]
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=kernel_sz_set_u, w_a=1.0, w_u=1.0, ema=1.0)
                feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
                scores_w0 = self.weight_gen_unet2(feats_0_w.detach())
                scores_w0 = F.interpolate(scores_w0, size=(feats_0.size(-2), feats_0.size(-1)), mode='bilinear')
                output = feats_0
                output_dict = {'feature_dict':[output], 'weight_dict':[scores_w0]}
                #self.iter_save = (self.iter_save + 1) % 2000
                return output, output_dict

            #local
            elif moe_mode == 2:
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=1.0)
                scores_w0 = self.weight_gen_unet2(feats_0.detach()).softmax(dim=1)[:, 1:]
                save_im0 = copy.deepcopy(feats_0)

                kernel_sz_set = [
                    [3,5],
                    [5,3],
                    [7,11],
                    [11,7],
                    [10, 20],
                    [20, 10],
                ]

                w_au = [
                    [1.0, 1.0, 0.9],
                ]
                
                #005
                error_thresh_0 = 0.0
                error_thresh_1 = 0.0

                ##########
                scores_up = scores_w0.clone() * 1.0
                final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
                medpool_0 =  MedianPool2d(kernel_size=3, stride=1, padding=1) #40.0
                medpool_1 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5) #41.2
                medpool_2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2) 
                scores_w0 = medpool_0(scores_w0)
                scores_up = medpool_0(scores_up)
                
                feats_mem = []
                scores_mem = []
                feats_g = copy.deepcopy(feats_0)
                for idx_k in range(len(w_au)):
                    for idx_ in range(len(kernel_sz_set)):
                        feats_ = self.user_norm_v2(feats_g,  kernel_sz_set=kernel_sz_set[idx_], w_a=w_au[idx_k][0], w_u=w_au[idx_k][1], ema=0.85, eps=1e-5)
                        scores_w = self.weight_gen_unet2(feats_.detach()).softmax(dim=1)[:, 1:]
                        scores_w = medpool_0(scores_w)

                        mask_0 = scores_w - scores_w0 >= error_thresh_0
                        mask_1 = scores_w - scores_up >= error_thresh_1
                        mask = (mask_0.float() * mask_1.float()).float()
                        mask = medpool_1(mask)

                        scores_up = scores_up * (1 - mask) + scores_w *  mask
                        mask = F.interpolate(mask, size=(feats_.size(-2), feats_.size(-1)), mode='bicubic')
                        feats_0 = feats_0 *  (1 - mask) + feats_ *  mask
                        
                        feats_mem.append(copy.deepcopy(feats_0))
                        scores_mem.append(copy.deepcopy(scores_up))
            
                scores_mem_t = torch.cat(scores_mem, dim=1)
                '''
                #print(scores_mem_t.size())
                #scores_mem_t = scores_mem_t - torch.topk(scores_mem_t, 3, dim=1)[0][:,2:]
                scores_mem_t = scores_mem_t - torch.topk(scores_mem_t, 1, dim=1)[0][:,0:]
                #scores_mem_t = torch.exp(scores_mem_t * 1.0) * (scores_mem_t >= 0.0).float()


                #scores_mem_t = scores_mem_t - scores_mem_t.min(dim=1, keepdim=True)[0]
                #scores_mem_t = scores_mem_t * 0.1 #torch.clip(scores_mem_t * 0.2, min=0.0, max=1.0)
                scores_mem_t = torch.clip(scores_mem_t * 1.2, min=0.0, max=0.8)
                scores_mem_t = torch.exp(scores_mem_t)
                weights = scores_mem_t / (scores_mem_t.sum(dim=1, keepdim=True) + 1e-5)
                weights = F.interpolate(weights, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)), mode='bicubic')
                '''

                #weights = F.softmax(scores_mem_t * 1.5, dim=1)
                weights = scores_mem_t / (scores_mem_t.sum(dim=1, keepdim=True) + 1e-5)
                weights = F.interpolate(weights, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)), mode='bicubic')
                
                final_feats = 0.0
                for f_idx in range(len(feats_mem)):
                    final_feats = final_feats + feats_mem[f_idx] * weights[:, f_idx:f_idx+1]
                output = 0.1 * feats_g + 0.9 * final_feats
                output_dict = {'feature_dict':[final_feats], 'weight_dict':scores_mem}
                '''
                ###draw
                self.iter_save = (self.iter_save + 1) % 2000
                if self.iter_save % 100 == 0:
                    vis_images_all = torch.zeros((1, output.size(1), 4 * output.size(-2), output.size(-1))).to(output.device)
                    vis_images_all[:, :, 0:output.size(-2)] = feats[0:1]
                    vis_images_all[:, :, output.size(-2):output.size(-2)*2] = save_im0[0:1]
                    vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = output[0:1]
                    vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = final_mask[0:1] 
                    #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output * mask)[0:1]

                    import torchvision.utils as vutils
                    
                    name = str(self.iter_save).zfill(4)    
                    vutils.save_image(vis_images_all.data, './samples/test/' + name +'.png', nrow=12, value_range=(-1, 1), normalize=True)
                '''
                return output, output_dict

            elif moe_mode == 3:
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
            elif moe_mode == 4:
                #feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.7)
                #feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
                scores_w0 = self.weight_gen_unet2(feats.detach())
                output = feats
                scores_w0 = F.interpolate(scores_w0, size=(feats.size(-2), feats.size(-1)), mode='bilinear')
                output_dict = {'feature_dict':[output], 'weight_dict':[scores_w0]}
                self.iter_save = (self.iter_save + 1) % 2000
                return output, output_dict
            
            ##################render####################
            elif moe_mode == 5:
                index_list = [1, 2, 3, 4, 5]
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.7)
                #feats_0_w = F.interpolate(feats_0, size=(feats_0.size(-2) // 2, feats_0.size(-1)//2), mode='bilinear')
                scores_w0 = self.weight_gen_unet2(feats_0).sigmoid()
                #scores_w0 = F.interpolate(scores_w0, size=(3, 5), mode='bilinear')
                save_im0 = feats_0.clone()

                kernel_sz_set = [
                    [3,5],
                    [5,3],
                    [7,11],
                    [11,7],
                    #[11, 21],
                    #[21, 11],
                    #[10, 20],
                    #[20, 10],
                ]

                w_au = [
                    [1.0, 1.0, 0.5],
                    #[1.0, 1.0, 0.5],
                    #[1.0, 1.0, 0.9],
                    #[0.9, 1.0, 0.7],
                    #[1.1, 1.0, 0.7],
                    #[1.0, 0.8, 0.7],
                    #[1.0, 1.2, 0.7],
                ]

                
                #005
                error_thresh_0 = 0.0
                error_thresh_1 = 0.0

                ##########
                scores_up = scores_w0.clone() * 1.0
                final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
                medpool_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #40.9
                medpool_1 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5) #41.2

                scores_w0 = medpool_0(scores_w0)
                scores_up = medpool_0(scores_up)
                
                feats_mem = []
                scores_mem = []
                for idx_k in range(len(w_au)):
                    for idx_ in range(len(kernel_sz_set)):

                        feats_ = self.user_norm_v2(feats,  kernel_sz_set=kernel_sz_set[idx_], w_a=w_au[idx_k][0], w_u=w_au[idx_k][1], ema=0.7, eps=1e-5)
                        scores_w = self.weight_gen_unet2(feats_).sigmoid()
                        scores_w = medpool_0(scores_w)

                        mask_0 = scores_w - scores_w0 > error_thresh_0
                        mask_1 = scores_w - scores_up > error_thresh_1
                        mask_2 = scores_w - scores_w0 > 0
                        mask = (mask_0.float() * mask_1.float() * mask_2.float() > 0.1).float()
                        #mask = (mask_0.float() * mask_1.float()).float()
                        mask = mask_0.float()
                        mask = medpool_1(mask)
                        #mask = mask  * (1 - final_mask)

                        scores_lup = scores_up * (1 - mask) + scores_w *  mask
                        mask = F.interpolate(mask, size=(feats_.size(-2), feats_.size(-1)), mode='bilinear')
                        feats_l0 = feats_0 *  (1 - mask) + feats_ *  mask
                        

                        feats_mem.append(feats_l0)
                        scores_mem.append(scores_lup)
                        #final_mask = final_mask + weight[:,0]
            
                scores_mem_t = torch.cat(scores_mem, dim=1)
                scores_mem_t = scores_mem_t - torch.topk(scores_mem_t, 1, dim=1)[0][:,0:]
                scores_mem_t = torch.clip(scores_mem_t * 1.2, min=0.0, max=0.8)
                scores_mem_t = torch.exp(scores_mem_t)
                weights = scores_mem_t / (scores_mem_t.sum(dim=1, keepdim=True) + 1e-5)
                weights = F.interpolate(weights, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)), mode='bilinear')
                
                
                final_feats = 0.0
                for f_idx in range(len(feats_mem)):
                    final_feats = final_feats + feats_mem[f_idx] * weights[:, f_idx:f_idx+1]
                    
                #output = 0.4 * feats_0 + 0.6 * final_feats #41.27
                output = self.render_net(final_feats) #0.4 * feats_0 + 0.6 * final_feats
                
                output_dict = {'feature_dict':[final_feats], 'weight_dict':scores_mem}
                return output, output_dict
                
            elif moe_mode == 6:
                #dis
                #feats = local_color_jitter_v2(feats)
                feats = user_domain_shift(feats)
                
                feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.9)
                scores_w0 = self.weight_gen_unet2(feats_0).sigmoid()
                save_im0 = feats_0.clone()
                #save_im0 = self.norm_back(feats_0.clone())

                kernel_sz_set = [
                    [3,5],
                    [5,3],
                    [7,11],
                    [11,7],
                    #[11, 21],
                    #[21, 11],
                    #[10, 20],
                    #[20, 10],
                ]

                w_au = [
                    [1.0, 1.0, 0.5],
                    #[1.0, 1.0, 0.5],
                    #[1.0, 1.0, 0.9],
                    #[0.9, 1.0, 0.7],
                    #[1.1, 1.0, 0.7],
                    #[1.0, 0.8, 0.7],
                    #[1.0, 1.2, 0.7],
                ]

                
                #005
                error_thresh_0 = 0.0
                error_thresh_1 = 0.0

                ##########
                scores_up = scores_w0.clone() * 1.0
                final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
                medpool_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #40.9
                medpool_1 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5) #41.2

                scores_w0 = medpool_0(scores_w0)
                scores_up = medpool_0(scores_up)
                
                feats_mem = []
                scores_mem = []
                for idx_k in range(len(w_au)):
                    for idx_ in range(len(kernel_sz_set)):

                        feats_ = self.user_norm_v2(feats,  kernel_sz_set=kernel_sz_set[idx_], w_a=w_au[idx_k][0], w_u=w_au[idx_k][1], ema=0.9, eps=1e-5)
                        feats_ = self.norm_back(feats_)
                        feats_mem.append(feats_)
                
                
                save_im0 = self.norm_back(save_im0)
                save_im0 = random_block_mask(save_im0)
                output = self.render_net(feats_mem[0], save_im0, feats_mem[1])
                output = self.render_net(feats_mem[2], output, feats_mem[3])
                output_dict = {'feature_dict':feats_mem, 'weight_dict':[None]}
                
                return output, output_dict
            elif moe_mode == 7:
                
                #print(moe_mark)
                #mark = random.randint(0, 2)
                if moe_mark == 0:            #random mask
                     feats = self.norm_back(feats)
                     #feats_pert = random_block_mask(feats, p=1.0, min_ratio=0.4, max_ratio=0.8, value='random_v')
                     feats_x0, feats_x1, attn_masks = random_block_mask(feats)
                     #feats_pert = self.norm_back(feats_pert)
                     output, out_dicts = self.render_net(feats_x1, feats_x0, feats_x0)

                elif moe_mark == 1:         #random mask + norm
                     kernel_sz_set = [
                       [3,5],
                       [5,3],
                       [7,11],
                       [11,7],
                       #[11, 21],
                       #[21, 11],
                       #[10, 20],
                       #[20, 10],
                       [13, 13]
                     ]
                     kernel_index_0 = random.randint(0, len(kernel_sz_set)-1)
                     feats_0 = self.user_norm_v2(feats,  kernel_sz_set=kernel_sz_set[kernel_index_0], w_a=1.0, w_u=1.0, ema=0.9, eps=1e-5)
                    
                     feats = self.norm_back(feats)
                     #feats_pert = random_block_mask(feats, p=1.0, min_ratio=0.4, max_ratio=0.8, value='random_v')
                     #feats_pert = self.norm_back(feats_pert)
                     feats_0 = self.norm_back(feats_0)
                     feats_x0, feats_x1, attn_masks = random_block_mask(feats_0)
                     output, out_dicts = self.render_net(feats_x1, feats_x0, feats_x0)
                     
                
                elif moe_mark == 2: #norm + clear
                     
                     kernel_sz_set = [
                       [3,5],
                       [5,3],
                       [7,11],
                       [11,7],
                       [13, 13]
                     ]
                     kernel_index_0 = random.randint(0, len(kernel_sz_set)-1)
                     feats = local_transform(feats, kernel_sz=kernel_sz_set[kernel_index_0])
                     feats_pert = self.user_norm_v2(feats,  kernel_sz_set=kernel_sz_set[kernel_index_0], w_a=1.0, w_u=1.0, ema=0.9, eps=1e-5)
                    
                     feats = self.norm_back(feats)
                     feats_pert = self.norm_back(feats_pert)
                     feats_x0 = copy.deepcopy(feats)
                     feats_x1 = copy.deepcopy(feats_pert)
                     
                     loc_mark = random.randint(0, 1)
                     if loc_mark == 0:
                         output, out_dicts = self.render_net(feats_x1, feats_x0, feats_pert)
                     elif loc_mark == 1:
                         output, out_dicts = self.render_net(feats_x0, feats_x1, feats_pert)
                     elif loc_mark == 2:
                         output, out_dicts = self.render_net(feats, feats_pert, feats_pert)
                     
                     attn_masks = torch.ones_like(feats)
                
                elif moe_mark == 3:  #domain simulator
                     feats_pert = user_domain_shift(feats)
                     feats = self.norm_back(feats)
                     feats_pert = self.norm_back(feats_pert)
                     
                     loc_mark = random.randint(0, 2)
                     if loc_mark == 0:
                         output = self.render_net(feats_pert, feats, feats_pert)
                     elif loc_mark == 1:
                         output = self.render_net(feats, feats_pert, feats_pert)
                     elif loc_mark == 2:
                         output = self.render_net(feats_pert, feats_pert, feats)
                         
                elif moe_mark == 4: #multi- norm
                    kernel_sz_set = [
                       [3,5],
                       [5,3],
                       [7,11],
                       [11,7],
                       #[11, 21],
                       #[21, 11],
                       #[10, 20],
                       #[20, 10],
                    ]
                    kernel_index_0 = random.randint(0, len(kernel_sz_set)-1)
                    kernel_index_1 = random.randint(0, len(kernel_sz_set)-1)
                    feats_0 = self.user_norm_v2(feats,  kernel_sz_set=kernel_sz_set[kernel_index_0], w_a=1.0, w_u=1.0, ema=0.9, eps=1e-5)
                    feats_1 = self.user_norm_v2(feats,  kernel_sz_set=kernel_sz_set[kernel_index_1], w_a=1.0, w_u=1.0, ema=0.9, eps=1e-5)
                    feats_pert = user_domain_shift(feats)
                    feats = self.norm_back(feats)
                    feats_0 = self.norm_back(feats_0)
                    feats_1 = self.norm_back(feats_1)
                    
                    loc_mark = random.randint(0, 2)
                    if loc_mark == 0:
                        output = self.render_net(feats_0, feats_1, feats_pert)
                    elif loc_mark == 1:
                        output = self.render_net(feats_0, feats_pert, feats_1)
                    elif loc_mark == 2:
                        output = self.render_net(feats_pert, feats_0, feats_1)
		    
		    
                elif moe_mark == 9:  # 自适应多渲染效果生成（变换元操作集合）
                    if self.adaptive_render_net is None:
                        # 6种渲染方法：HDR增强、对比度增强、亮度调整、色彩饱和度调整、局部注意力增强、ExpoMamba
                        self.adaptive_render_net = AdaptiveRenderNetwork(n_channel=3, n_render_methods=6, patch_size=32)
                        self.adaptive_render_net.to(feats.device)
                    
                    feats_back = self.norm_back(feats)
                    
                    # 使用自适应渲染网络生成多个渲染效果（对所有图片都应用各种渲染方法）
                    render_results = self.adaptive_render_net(feats_back)
                    
                    # 获取最终渲染结果
                    output = self.norm_forward(render_results['final_image'])
                    
                    # 构建输出字典
                    output_dict = {
                        'feature_dict': [render_results['final_image']],
                        'weight_dict': [render_results['selection_map']],
                        'attn': render_results['fusion_weights'],
                        'attn_masks': torch.ones_like(feats),
                        'render_results': render_results['render_results'],
                        'selection_map': render_results['selection_map'],
                        'selection_mask': render_results['selection_mask']
                    }
                    
                    # 可视化
                    self.iter_save = (self.iter_save + 1) % 2000
                    if self.iter_save % 40 == 0:
                        vis_images = [feats_back]
                        for render in render_results['render_results']:
                            vis_images.append(render)
                        vis_images.append(render_results['final_image'])
                        
                        n_images = len(vis_images)
                        vis_images_all = torch.zeros((1, 3, n_images * feats_back.size(-2), feats_back.size(-1))).to(feats_back.device)
                        for i, img in enumerate(vis_images):
                            vis_images_all[:, :, i*feats_back.size(-2):(i+1)*feats_back.size(-2)] = (img[0:1] - 0.5) / 0.5
                        
                        import torchvision.utils as vutils
                        os.makedirs('./samples/adaptive_render/', exist_ok=True)
                        name = str(self.iter_save).zfill(4)    
                        vutils.save_image(vis_images_all.data, './samples/adaptive_render/' + name +'.jpg', 
                                         nrow=12, value_range=(-1, 1), normalize=True)
                    
                    return output, output_dict
		    
		    
                ###draw
                
                self.iter_save = (self.iter_save + 1) % 2000
                if self.iter_save % 40 == 0:
                    attn_masks_v =attn_masks.expand(attn_masks.size(0), 3, attn_masks.size(-2), attn_masks.size(-1))
                    vis_images_all = torch.zeros((1, output.size(1), 5 * output.size(-2), output.size(-1))).to(output.device)
                    vis_images_all[:, :, 0:output.size(-2)] = (feats[0:1] - 0.5) / 0.5
                    vis_images_all[:, :, output.size(-2):output.size(-2)*2] = (output[0:1] - 0.5) / 0.5
                    vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = (feats_x0[0:1] - 0.5) / 0.5 
                    vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (feats_x1[0:1] - 0.5) / 0.5
                    vis_images_all[:, :, 4*output.size(-2):output.size(-2)*5] = (attn_masks_v[0:1] - 0.5) / 0.5

                    import torchvision.utils as vutils
                
                    name = str(self.iter_save).zfill(4)    
                    vutils.save_image(vis_images_all.data, './samples/changes/' + name +'.jpg', nrow=12, value_range=(-1, 1), normalize=True)
               
                output_dict = {'feature_dict':[None], 'weight_dict':[None], 'attn':out_dicts['attn'], 'attn_masks':attn_masks}
                
                return output, output_dict
            
        #else:
        #    feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.9)
        #    #output = feats
        #    output = feats_0
        #    output_dict = None
        #    return output, output_dict
        else:
        #if 0:
            '''
            index_list = [1, 2, 3, 4, 5]
            feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=1.0)
            scores_w0 = self.weight_gen_unet2(feats_0).sigmoid()
            save_im0 = copy.deepcopy(feats_0)

            kernel_sz_set = [
                [3,5],
                [5,3],
                [7,11],
                [11,7],
                #[11, 21],
                #[21, 11],
                [10, 20],
                [20, 10],
            ]

            w_au = [
                [1.0, 1.0, 0.5],
                #[1.0, 1.0, 0.5],
                #[1.0, 1.0, 0.9],
                #[0.9, 1.0, 0.7],
                #[1.1, 1.0, 0.7],
                #[1.0, 0.8, 0.7],
                #[1.0, 1.2, 0.7],
            ]

            from torchvision.transforms import functional as tF
            transform = v2.Compose([
                v2.ColorJitter(brightness=0.1)
            ])
            
            ##########
            #001:30.54
            #error_thresh_0 = 0.01
            #error_thresh_1 = 0.001
            
            #002: 30.28
            #error_thresh_0 = 0.02
            #error_thresh_1 = 0.001
            
            #003: 29.91
            #error_thresh_0 = 0.03
            #error_thresh_1 = 0.001

            #004: 30.66
            #error_thresh_0 = 0.005
            #error_thresh_1 = 0.001
            
            #005
            error_thresh_0 = 0.0
            error_thresh_1 = 0.0

            ##########
            scores_up = scores_w0.clone() * 1.0
            final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
            #medpool_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #40.9
            #medpool_0 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            medpool_0 =  MedianPool2d(kernel_size=3, stride=1, padding=1) #40.0
            #medpool_0 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
            #medpool_0 = nn.AvgPool2d(kernel_size=7, stride=3, padding=3)

            #medpool_1 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4) #41.24
            #medpool_1 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3) #41.2
            medpool_1 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5) #41.2
            #medpool_1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6) #40.82
            #medpool_1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2) #41.21
            #medpool_1 = nn.MaxPool2d(kernel_size=17, stride=1, padding=8)

            #medpool_1 = nn.MaxPool2d(kernel_size=15, stride=1, padding=7)
            #medpool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            medpool_2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2) 

            scores_w0 = medpool_0(scores_w0)
            scores_up = medpool_0(scores_up)

            #
            #medpool = MedianPool2d(kernel_size=3, stride=1, padding=1)
            #medpool = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
            #medpool = nn.MaxPool2d(kernel_size=11, stride=1, padding=5) #25.76
            #medpool = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)  #26.26
            #medpool = nn.MaxPool2d(kernel_size=15, stride=1, padding=7)  #26.62
            #medpool = nn.MaxPool2d(kernel_size=17, stride=1, padding=8)  #26.52
            #medpool = nn.MaxPool2d(kernel_size=19, stride=1, padding=9) #26.80
            #medpool = nn.MaxPool2d(kernel_size=21, stride=1, padding=10) #27.0
            #for idx_ in range(len(kernel_sz_set)):
            #    for idx_k in range(len(w_au)):
            medpool_30 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
            medpool_31 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
            
            feats_mem = []
            scores_mem = []
            #feats = copy.deepcopy(feats_0)
            feats_g = copy.deepcopy(feats_0)
            for idx_k in range(len(w_au)):
                for idx_ in range(len(kernel_sz_set)):
                    #feats_ = feats * 2
                    #feats_ = transform(feats)

                    #feats_lx = feats * 0.5 + 0.5
                    #feats_lx = tF.adjust_brightness(feats_lx, brightness_factor=1.5)
                    #feats_lx = (feats_lx - 0.5) / 0.5

                    feats_ = self.user_norm_v2(feats_g,  kernel_sz_set=kernel_sz_set[idx_], w_a=w_au[idx_k][0], w_u=w_au[idx_k][1], ema=0.85, eps=1e-5)
                    #print(feats_.max(), feats_.min())
                    #feats_lx = feats_ * 0.5 + 0.5
                    #feats_lx = tF.adjust_brightness(feats_lx, brightness_factor=1.5)
                    #feats_lx = tF.adjust_contrast(feats_lx, 1.5)
                    #feats_lx = (feats_lx - 0.5) / 0.5
                    #feats_ = feats_lx
                    
                    scores_w = self.weight_gen_unet2(feats_).sigmoid()
                    scores_w = medpool_0(scores_w)
                    #scores_w = 1-medpool_0(1-scores_w)
                    #mask_0 = (scores_w - scores_w0) / (scores_w0 + 1e-6) > error_thresh_0
                    #mask_1 = (scores_w - scores_up) / (scores_up + 1e-6) > error_thresh_1
                    mask_0 = scores_w - scores_w0 >= error_thresh_0
                    mask_1 = scores_w - scores_up >= error_thresh_1
                    mask_2 = scores_w - scores_w0 > 0
                    #mask = (mask_0.float() * mask_1.float() * mask_2.float() > 0.1).float()
                    mask = (mask_0.float() * mask_1.float()).float()
                    mask = medpool_1(mask)
                    #mask = mask  * (1 - final_mask)

                    scores_up = scores_up * (1 - mask) + scores_w *  mask
                    #bicubic, bilinear
                    mask = F.interpolate(mask, size=(feats_.size(-2), feats_.size(-1)), mode='bicubic')
                    feats_0 = feats_0 *  (1 - mask) + feats_ *  mask
                    

                    feats_mem.append(copy.deepcopy(feats_0))
                    scores_mem.append(copy.deepcopy(scores_up))
                    
                    #feats_mem.append(feats_)
                    #scores_mem.append(scores_w)
                    
                    #final_mask = final_mask + weight[:,0]
        
            scores_mem_t = torch.cat(scores_mem, dim=1)
            '''
            '''
            #print(scores_mem_t.size())
            #scores_mem_t = scores_mem_t - torch.topk(scores_mem_t, 3, dim=1)[0][:,2:]
            scores_mem_t = scores_mem_t - torch.topk(scores_mem_t, 1, dim=1)[0][:,0:]
            #scores_mem_t = torch.exp(scores_mem_t * 1.0) * (scores_mem_t >= 0.0).float()


            #scores_mem_t = scores_mem_t - scores_mem_t.min(dim=1, keepdim=True)[0]
            #scores_mem_t = scores_mem_t * 0.1 #torch.clip(scores_mem_t * 0.2, min=0.0, max=1.0)
            scores_mem_t = torch.clip(scores_mem_t * 1.2, min=0.0, max=0.8)
            scores_mem_t = torch.exp(scores_mem_t)
            weights = scores_mem_t / (scores_mem_t.sum(dim=1, keepdim=True) + 1e-5)
            weights = F.interpolate(weights, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)), mode='bicubic')
            '''
            '''

            #weights = F.softmax(scores_mem_t * 1.5, dim=1)
            weights = scores_mem_t / (scores_mem_t.sum(dim=1, keepdim=True) + 1e-5)
            weights = F.interpolate(weights, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)), mode='bicubic')
            
            final_feats = 0.0
            for f_idx in range(len(feats_mem)):
                final_feats = final_feats + feats_mem[f_idx] * weights[:, f_idx:f_idx+1]
                

            #output = final_feats #0.5* save_im0 + 0.5*feats_f0 #0.5* save_im0 + 0.5*feats_0 #self.final_conv(feats_0)
            #output = 0.4 * feats_0 + 0.6 * final_feats #41.27
            output = 0.1 * feats_g + 0.9 * final_feats
            
            
            #output=feats_0
            #final_feats = feats_0
            #scores_mem = []
            medpool_f = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            #output = medpool_f(output)
            output_dict = {'feature_dict':[final_feats], 'weight_dict':scores_mem}

            final_mask = [scores_w, scores_w, scores_w]
            final_mask= torch.cat(final_mask, dim=1)
            final_mask = F.interpolate(final_mask, size=(feats_mem[0].size(-2), feats_mem[0].size(-1)), mode='bicubic')
            final_mask = (final_mask - 0.5) / 0.5
            

            ###draw
            self.iter_save = (self.iter_save + 1) % 2000
            if self.iter_save % 100 == 0:
                vis_images_all = torch.zeros((1, output.size(1), 4 * output.size(-2), output.size(-1))).to(output.device)
                vis_images_all[:, :, 0:output.size(-2)] = feats[0:1]
                vis_images_all[:, :, output.size(-2):output.size(-2)*2] = save_im0[0:1]
                vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = output[0:1]
                vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = final_mask[0:1] 
                #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output * mask)[0:1]

                import torchvision.utils as vutils
                
                name = str(self.iter_save).zfill(4)    
                vutils.save_image(vis_images_all.data, './samples/test/' + name +'.png', nrow=12, value_range=(-1, 1), normalize=True)
            return output, output_dict
            '''
            
            feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.0)
            #scores_w0 = self.weight_gen_unet2(feats_0).sigmoid()
            save_im0 = feats_0.clone()

            kernel_sz_set0 = [
                    [3,5],
                    [5,3],
                    [7,11],
                    [11,7],
                    #[11, 21],
                    #[21, 11],
                    #[10, 20],
                    #[20, 10],
                    [13, 13]
            ]
            kernel_sz_set =[
                    [3,3],
                    [7,7],
                    [11,11],
                    [15, 15],
                    [19, 19]
                    
            ]
            

            w_au = [
                    [1.0, 1.0, 0.5],
                    #[1.0, 1.0, 0.5],
                    #[1.0, 1.0, 0.9],
                    #[0.9, 1.0, 0.7],
                    #[1.1, 1.0, 0.7],
                    #[1.0, 0.8, 0.7],
                    #[1.0, 1.2, 0.7],
            ]

                
            #005
            error_thresh_0 = 0.0
            error_thresh_1 = 0.0

            ##########
            #scores_up = scores_w0.clone() * 1.0
            final_mask = torch.zeros((1, feats_0.size(1), feats_0.size(-2), feats_0.size(-1))).to(feats_0.device)
            medpool_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #40.9
            medpool_1 = nn.MaxPool2d(kernel_size=11, stride=1, padding=5) #41.2

            #scores_w0 = medpool_0(scores_w0)
            #scores_up = medpool_0(scores_up)
       
            feats_mem = []
            scores_mem = []
            for idx_k in range(len(w_au)):
                    for idx_ in range(len(kernel_sz_set)):
                        #ema=0.85: 40.5
                        feats_ = self.user_norm_v2(feats_0,  kernel_sz_set=kernel_sz_set[idx_], w_a=w_au[idx_k][0], w_u=w_au[idx_k][1], ema=0.85, eps=1e-5)
                        #feats_ = self.local_bright_contrast(feats_0,  kernel_sz_set=kernel_sz_set[idx_])
                        feats_ = self.norm_back(feats_)
                        feats_mem.append(feats_)
                
            save_im0 = self.norm_back(save_im0)
            output_0 = self.render_net(feats_mem[0],  save_im0, feats_mem[1])
            output_0 = output_0 * 0.9 + save_im0 * 0.1
            output = self.render_net(feats_mem[1],  output_0, feats_mem[3])
            output = output * 0.9 + save_im0 * 0.1
            output = self.render_net(feats_mem[2], output, feats_mem[3])
            output = output * 0.9 + save_im0 * 0.1
            output = self.render_net(feats_mem[3], output, feats_mem[3])
            output = output * 0.9 + save_im0 * 0.1
            output = self.render_net(feats_mem[4], output, feats_mem[3])
            #medpool_0 =  MedianPool2d(kernel_size=3, stride=1, padding=1)
            #output = medpool_0(output)
            output = self.norm_forward(output)
            #output = self.user_norm_v2(output, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.9)
            output_dict = {'feature_dict':feats_mem, 'weight_dict':[None]}
                
            ###draw
            self.iter_test_save = (self.iter_test_save + 1) % 3000
            if self.iter_test_save % 50 == 0:
                
                #vis_images_all = torch.zeros((1, output.size(1), 5 * output.size(-2), output.size(-1))).to(output.device)
                #vis_images_all[:, :, 0:output.size(-2)] = feats[0:1]
                #vis_images_all[:, :, output.size(-2):output.size(-2)*2] = (save_im0[0:1] - 0.5) / 0.5
                #vis_images_all[:, :, 2*output.size(-2):output.size(-2)*3] = output[0:1]
                #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (feats_mem[3] - 0.5) / 0.5 
                #vis_images_all[:, :, 4*output.size(-2):output.size(-2)*5] = (output_0 - 0.5) / 0.5
                #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = final_mask[0:1] 
                #vis_images_all[:, :, 3*output.size(-2):output.size(-2)*4] = (output * mask)[0:1]
                
                
                vis_images_all = torch.zeros((1, output.size(1), 2 * output.size(-2), output.size(-1))).to(output.device)
                vis_images_all[:, :, 0:output.size(-2)] = feats[0:1]
                vis_images_all[:, :, 1*output.size(-2):output.size(-2)*2] = output[0:1]

                import torchvision.utils as vutils
                
                name = str(self.iter_test_save).zfill(4)    
                vutils.save_image(vis_images_all.data, './samples/test/' + name +'.png', nrow=12, value_range=(-1, 1), normalize=True)
            
            #output = self.norm_forward(output)
            save_im0 = self.norm_forward(save_im0)
            #output = output * 0.1 + 0.9 * save_im0
            
            
            #feats_0 = self.user_norm_v2(feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, ema=0.9)
            #output = feats
            #output_dict = None
            #print(feats.mean(), output.mean())
            #output = output * 0.8 + 0.2 * feats
            
            #output = output * 1.0 + 0.0 * save_im0
            
            
            return output, output_dict
    


    def calc_std_em(self, feat, eps=1e-8):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]

        feat = feat ** 2
        feat = (self.pool(feat) + eps).sqrt() 
        feats = F.interpolate(feat, size=(size[-2], size[-1]), mode='bicubic')

        return feats
        
    
    def user_norm_v2(self, feats, mode=0, kernel_sz_set=[1,1], w_a=1.0, w_u=1.0, eps=1e-8, ema=0.1):
        pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set)
        feats_pool = pool_func(feats)

        feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        pfeats = feats - feats_pool
        
        sfeats = pfeats ** 2
        sfeats = (pool_func(sfeats) + eps).sqrt() 

        sfeats = F.interpolate(sfeats, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        alpha_ba = self.alpha_ba.clone().detach()
        alpha_bu = self.alpha_bu.clone().detach()

        feats_mean, feats_std = calc_mean_std(feats)
        ba_ema_rate = ema
        sfeats = sfeats * ba_ema_rate + (1 - ba_ema_rate) * feats_std
        alpha_bu = alpha_bu[None, :, None, None] #  * 0.7 + 0.3 * feats_mean

        #nfeats = w_a * pfeats * alpha_ba[None, :, None, None] / (sfeats + 1e-6) + w_u * alpha_bu
        nfeats = w_a * pfeats * alpha_ba[None, :, None, None] / (sfeats + 1e-3)  + w_u * alpha_bu
        nfeats = torch.clip(nfeats, min=-1.0, max=1.0)
        #nfeats = self.medpool(nfeats)

        #conv
        #nfeats = self.proj_norm_1x1(nfeats.float())
        #nfeats = self.proj_norm_1x1(nfeats.float())
        return nfeats.float()

    
    def update_norm_parameters(self, feats, is_update=False):
        #m5
        #update self.alpha
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

            self.alpha_ba.data = self.alpha_ba.data.clone().detach() * self.ema_rate + (1 - self.ema_rate) * feats_std.mean(dim=0).data.clone().detach()
            self.alpha_bu.data = self.alpha_bu.data.clone().detach() * self.ema_rate + (1 - self.ema_rate) * feats_mean.mean(dim=0).data.clone().detach()
            self.memory_index = self.memory_len - 1
            self.memory_index += 1
            

        return
        


    
    def compute_norm_parameters(self, device=None):
        alpha_ba = torch.from_numpy(self.memory_aplpha_a[0:self.memory_loc].sum(axis=0) / self.memory_loc)
        alpha_bu = torch.from_numpy(self.memory_aplpha_u[0:self.memory_loc].sum(axis=0) / self.memory_loc)
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
    
    def norm_back(self, feats):
        feats = feats * 0.5 + 0.5
        return feats
    
    def norm_forward(self, feats):
        feats = (feats - 0.5) / 0.5
        return feats
    
    def local_bright_contrast(self, feats, mode=0, kernel_sz_set=[1,1], w_a=(0.8, 1.2), w_u=(0.8, 1.2)):
        B,C, H, W = feats.size()
        device = feats.device
        pool_func = nn.AdaptiveAvgPool2d(kernel_sz_set)
        feats_pool = pool_func(feats)

        feats_pool = F.interpolate(feats_pool, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        pfeats = feats - feats_pool

        ###
        
        brightness = torch.empty(B, C, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*w_a)
        contrast = torch.empty(B, C, kernel_sz_set[0], kernel_sz_set[1], device=device).uniform_(*w_u)
        brightness = F.interpolate(brightness, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        contrast = F.interpolate(contrast, size=(feats.size(-2), feats.size(-1)), mode='bicubic')
        ###
        
        nfeats = contrast * pfeats   + brightness * feats_pool
        nfeats = torch.clip(nfeats, min=-1.0, max=1.0)
        return nfeats.float()


    

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
