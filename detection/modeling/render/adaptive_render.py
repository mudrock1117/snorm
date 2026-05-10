import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from detection.modeling.HDR.AHDR.AHDR import AHDR

# 简化版局部注意力模块，专门用于RGB图像
class SimpleLocalAttention(nn.Module):
    def __init__(self, dim=3, window_size=7):
        super().__init__()
        self.dim = dim
        self.window = window_size
        
        # 简化为卷积增强，避免显存问题
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        # 使用多尺度卷积替代注意力机制
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        out = self.conv3(feat1 + feat2)
        return x + out

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


class MultiRenderTransform(nn.Module):
    """
    变换元操作集合 - 对所有输入图片都应用各种渲染方法
    包括：HDR增强、对比度增强、亮度调整、色彩饱和度调整、局部注意力增强、ExpoMamba
    """
    def __init__(self, n_channel=3, n_render_methods=6):
        super(MultiRenderTransform, self).__init__()
        self.n_channel = n_channel
        self.n_render_methods = n_render_methods
        
        # 渲染方法1: HDR增强 (基于现有AHDR)
        self.ahdr = AHDR(n_channel, nDenselayer=4, nFeat=64, growthRate=32)
        
        # 渲染方法2: 对比度增强
        self.contrast_enhance = nn.Sequential(
            nn.Conv2d(n_channel, n_channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channel * 2, n_channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 渲染方法3: 亮度调整
        self.brightness_adjust = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 渲染方法4: 色彩饱和度调整
        self.saturation_adjust = nn.Sequential(
            nn.Conv2d(n_channel, n_channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channel * 2, n_channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 渲染方法5: 局部注意力增强
        self.local_attn = SimpleLocalAttention(dim=n_channel, window_size=7)
        
        # 渲染方法6: ExpoMamba算子
        self.expmamba_operator = None
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(n_channel * n_render_methods, n_channel * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channel * 4, n_channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channel * 2, n_channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        对输入图片应用所有渲染方法，返回每个方法的渲染结果
        """
        batch_size, channels, height, width = x.shape
        render_results = []
        
        # 方法1: HDR增强 (需要3个输入，这里使用同一个图像的3个版本)
        x_hdr = self.ahdr(x, x, x)
        render_results.append(x_hdr)
        
        # 方法2: 对比度增强
        contrast_weight = self.contrast_enhance(x)
        x_contrast = x * contrast_weight
        render_results.append(x_contrast)
        
        # 方法3: 亮度调整
        brightness_weight = 0.5 + 0.5 * self.brightness_adjust(x)
        x_brightness = x * brightness_weight
        render_results.append(x_brightness)
        
        # 方法4: 色彩饱和度调整
        saturation_weight = self.saturation_adjust(x)
        x_saturation = x * saturation_weight
        render_results.append(x_saturation)
        
        # 方法5: 局部注意力增强
        x_local_attn = self.local_attn(x)
        x_local_attn = x + 0.3 * x_local_attn
        render_results.append(x_local_attn)
        
        # 方法6: ExpoMamba算子增强
        if self.expmamba_operator is None:
            self.expmamba_operator = get_expmamba_operator_lazy()
        
        if self.expmamba_operator is not None:
            expomamba_out = self.expmamba_operator.enhance_tensor(x)
            render_results.append(expomamba_out)
        else:
            # 如果ExpoMamba不可用，使用原始图像作为第6种方法
            render_results.append(x)
        
        # 确保所有渲染结果具有相同的形状
        for i in range(len(render_results)):
            if render_results[i].shape != x.shape:
                render_results[i] = F.interpolate(render_results[i], size=(height, width), mode='bilinear')
        
        # 堆叠所有渲染结果
        all_renders = torch.cat(render_results, dim=1)
        
        # 特征融合
        fusion_weights = self.feature_fusion(all_renders)
        
        return render_results, fusion_weights


class LocalRenderSelector(nn.Module):
    """
    局部最优选择器 - 对每个局部patch选择最优的渲染方法
    """
    def __init__(self, n_channel=3, n_render_methods=6, patch_size=32):
        super(LocalRenderSelector, self).__init__()
        self.n_channel = n_channel
        self.n_render_methods = n_render_methods
        self.patch_size = patch_size
        
        # 局部质量评估网络 - 输出单个质量分数
        self.quality_net = nn.Sequential(
            nn.Conv2d(n_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),  # 输出单个质量分数
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
        # 全局优化网络
        self.global_opt = nn.Sequential(
            nn.Conv2d(n_channel * n_render_methods, n_channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channel * 2, n_channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 平滑过渡网络（用于相邻patch之间的平滑）
        self.smooth_transition = nn.Conv2d(n_channel, n_channel, kernel_size=5, padding=2, groups=n_channel)
        
    def forward(self, render_results, fusion_weights):
        batch_size, channels, height, width = render_results[0].shape
        
        # 计算每个patch的最优渲染方法
        n_patches_h = (height + self.patch_size - 1) // self.patch_size
        n_patches_w = (width + self.patch_size - 1) // self.patch_size
        
        selected_image = torch.zeros_like(render_results[0])
        selection_map = torch.zeros(batch_size, self.n_render_methods, n_patches_h, n_patches_w, 
                                    device=render_results[0].device)
        selection_mask = torch.zeros(batch_size, height, width, device=render_results[0].device)
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                h_start = i * self.patch_size
                h_end = min((i + 1) * self.patch_size, height)
                w_start = j * self.patch_size
                w_end = min((j + 1) * self.patch_size, width)
                
                # 评估每个渲染方法在当前patch上的质量
                quality_scores = []
                for render in render_results:
                    patch = render[:, :, h_start:h_end, w_start:w_end]
                    # 如果patch太小，进行上采样
                    if patch.size(-2) < 8 or patch.size(-1) < 8:
                        patch = F.interpolate(patch, size=(16, 16), mode='bilinear')
                    score = self.quality_net(patch)  # (batch_size, 1)
                    quality_scores.append(score)
                
                # 堆叠质量分数 (batch_size, n_render_methods)
                quality_scores = torch.cat(quality_scores, dim=1)
                
                # 选择最优的渲染方法
                best_method = quality_scores.argmax(dim=1)  # (batch_size,)
                
                # 使用最优方法渲染的patch
                for b in range(batch_size):
                    method_idx = int(best_method[b].item())  # 转换为整数索引
                    selected_image[b, :, h_start:h_end, w_start:w_end] = render_results[method_idx][b, :, h_start:h_end, w_start:w_end]
                    selection_map[b, :, i, j] = quality_scores[b]
                    selection_mask[b, h_start:h_end, w_start:w_end] = best_method[b]
        
        # 全局优化
        all_renders = torch.cat(render_results, dim=1)
        global_weights = self.global_opt(all_renders)
        
        # 结合局部选择和全局优化
        final_image = selected_image * global_weights
        
        # 应用平滑过渡
        final_image = self.smooth_transition(final_image)
        
        return final_image, selection_map, selection_mask


class AdaptiveRenderNetwork(nn.Module):
    """
    自适应渲染网络 - 整合变换元操作和局部最优选择
    """
    def __init__(self, n_channel=3, n_render_methods=6, patch_size=32):
        super(AdaptiveRenderNetwork, self).__init__()
        self.n_channel = n_channel
        self.n_render_methods = n_render_methods
        self.patch_size = patch_size
        
        # 变换元操作集合（对所有图片应用各种渲染方法）
        self.transform_set = MultiRenderTransform(n_channel, n_render_methods)
        
        # 局部最优选择器
        self.selector = LocalRenderSelector(n_channel, n_render_methods, patch_size)
        
        # 最终整合模块
        self.final_integration = nn.Sequential(
            nn.Conv2d(n_channel, n_channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channel * 2, n_channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        完整的自适应渲染流程：
        1. 对输入图片应用所有渲染方法
        2. 选择每个局部最优的渲染结果
        3. 整合输出
        """
        # 生成多个渲染效果（对所有图片都应用各种渲染方法）
        render_results, fusion_weights = self.transform_set(x)
        
        # 选择每个局部最优的渲染图片
        selected_image, selection_map, selection_mask = self.selector(render_results, fusion_weights)
        
        # 最终整合
        final_image = self.final_integration(selected_image)
        
        return {
            'final_image': final_image,
            'render_results': render_results,
            'selection_map': selection_map,
            'selection_mask': selection_mask,
            'fusion_weights': fusion_weights
        }
