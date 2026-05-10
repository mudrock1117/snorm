import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from detection.modeling.HDR.OURS.OHDR import AHDR

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


class LocalPatchRenderer(nn.Module):
    """
    局部Patch渲染器 - 对每个局部patch应用不同的渲染方法
    渲染方法：AHDR（原始方法）+ ExpoMamba
    """
    def __init__(self, n_channel=3, patch_size=32):
        super(LocalPatchRenderer, self).__init__()
        self.n_channel = n_channel
        self.patch_size = patch_size
        
        # 渲染方法1: AHDR原始渲染（使用OHDR中的AHDR实现，与原始项目一致）
        self.ahdr = AHDR(in_ch=n_channel, base_ch=64)
        
        # 渲染方法2: ExpoMamba算子
        self.expmamba_operator = None
        
        # 记录渲染方法数量
        self.n_methods = 2
    
    def load_ahdr_weights(self, checkpoint_path):
        """加载AHDR预训练权重"""
        import os
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_state_dict = checkpoint['model']
            ahdr_state_dict = {k.replace('teacher.proj_norm.render_net.', ''): v
                              for k, v in model_state_dict.items()
                              if 'teacher.proj_norm.render_net' in k}
            self.ahdr.load_state_dict(ahdr_state_dict, strict=False)
            self.ahdr = self.ahdr.to(next(self.parameters()).device)
            print(f"✅ LocalPatchRenderer: AHDR权重加载成功 ({len(ahdr_state_dict)}个参数)")
        else:
            print(f"⚠️ LocalPatchRenderer: 未找到权重文件: {checkpoint_path}")
        
    def forward(self, x):
        """
        对输入图像应用所有渲染方法
        返回每个方法的渲染结果列表
        
        输入: x - [0, 1]范围的图像张量
        输出: [ahdr_result, expomamba_result] - 两种渲染方法的结果
              - AHDR输出范围: [0, 1]
              - ExpoMamba输出范围: [-1, 1]
        """
        batch_size, channels, height, width = x.shape
        
        # 方法1: AHDR渲染（需要3个输入，模拟原始项目中的调用方式）
        # 在原始项目中，AHDR接收的是经过user_norm_v2和norm_back处理的特征图
        # 这里我们使用简化的模拟：原始图像 + 经过不同扰动处理的图像
        from detection.modeling.norm.lib_pert import random_block_mask, local_color_jitter_v2
        
        # 创建三个不同的输入
        x1 = x.clone()  # 原始图像
        
        # 经过random_block_mask处理的图像
        x_masked1, x_masked2, _ = random_block_mask(x.clone())
        x2 = x_masked1
        
        # 经过local_color_jitter_v2处理的图像（模拟局部归一化效果）
        x_jitter = local_color_jitter_v2((x - 0.5) / 0.5)  # 需要[-1, 1]输入
        x3 = (x_jitter * 0.5 + 0.5).clamp(0, 1)  # 转换回[0, 1]
        
        # AHDR接收三个不同的输入
        ahdr_output = self.ahdr(x1, x2, x3)  # 输出范围[0, 1]
        # AHDR可能返回元组 (output_tensor, dict)，只取张量部分
        if isinstance(ahdr_output, tuple):
            ahdr_result = ahdr_output[0]
        else:
            ahdr_result = ahdr_output
        
        # 方法2: ExpoMamba渲染（直接使用[0, 1]范围输入，不需要转换）
        if self.expmamba_operator is None:
            self.expmamba_operator = get_expmamba_operator_lazy()
        if self.expmamba_operator is not None:
            # ExpoMamba直接接收[0, 1]范围
            expomamba_result = self.expmamba_operator.enhance_tensor(x)
        else:
            # 如果ExpoMamba不可用，使用原始图像作为第2种方法
            expomamba_result = x
        
        # 确保ExpoMamba输出范围在[0, 1]内
        expomamba_result = torch.clamp(expomamba_result, 0, 1)
        
        # 确保渲染结果尺寸一致
        if ahdr_result.shape != x.shape:
            ahdr_result = F.interpolate(ahdr_result, size=(height, width), mode='bilinear')
        if expomamba_result.shape != x.shape:
            expomamba_result = F.interpolate(expomamba_result, size=(height, width), mode='bilinear')
        
        return [ahdr_result, expomamba_result]


class LocalQualityEvaluator(nn.Module):
    """
    局部质量评估网络 - 评估每个渲染方法在局部patch上的质量
    输出质量分数用于选择最优渲染
    """
    def __init__(self, n_channel=3):
        super(LocalQualityEvaluator, self).__init__()
        self.n_channel = n_channel
        
        # 质量评估网络 - 输出单个质量分数
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
            nn.Linear(256, 1),
            nn.Sigmoid()  # 归一化到[0,1]
        )
        
    def forward(self, x):
        """
        评估输入图像的质量分数
        """
        return self.quality_net(x)


class LocalOptimalSelector(nn.Module):
    """
    局部最优选择器 - 对每个局部patch选择最优的渲染方法
    """
    def __init__(self, n_channel=3, n_methods=2, patch_size=32, trained_model_path=None, 
                 selection_threshold=0.1, temperature=1.0):
        super(LocalOptimalSelector, self).__init__()
        self.n_channel = n_channel
        self.n_methods = n_methods
        self.patch_size = patch_size
        self.selection_threshold = selection_threshold  # 选择阈值
        self.temperature = temperature  # softmax温度系数

        # 质量评估器
        self.quality_evaluator = LocalQualityEvaluator(n_channel)

        # 加载训练好的质量评估器权重
        if trained_model_path is not None:
            self.load_trained_quality_evaluator(trained_model_path)

        # 平滑过渡网络（用于相邻patch之间的平滑）
        self.smooth_transition = nn.Conv2d(n_channel, n_channel,
                                          kernel_size=5, padding=2, groups=n_channel)

    def _gaussian_blur(self, x, kernel_size=5, sigma=2.0):
        """应用高斯模糊以平滑图像，消除方块边界"""
        import math
        batch, channels, height, width = x.shape
        device = x.device

        kernel = self._create_gaussian_kernel(kernel_size, sigma, channels, device)
        padding = kernel_size // 2

        output = F.conv2d(x, kernel, padding=padding, groups=channels)
        return output

    def _create_gaussian_kernel(self, kernel_size, sigma, channels, device):
        """创建高斯核"""
        import math
        kernel = torch.zeros(kernel_size, kernel_size, device=device)
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = math.exp(-(x*x + y*y) / (2 * sigma * sigma))

        kernel = kernel / kernel.sum()

        return kernel.repeat(channels, 1, 1, 1)

    def load_trained_quality_evaluator(self, model_path):
        """加载训练好的质量评估器权重"""
        import os
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'quality_evaluator_state_dict' in checkpoint:
                state_dict = checkpoint['quality_evaluator_state_dict']
            else:
                state_dict = checkpoint
            self.quality_evaluator.load_state_dict(state_dict)
            print(f"✅ 成功加载训练好的质量评估器: {model_path}")
            if 'loss' in checkpoint:
                print(f"   训练损失: {checkpoint['loss']:.6f}")
        else:
            print(f"⚠️ 训练好的模型不存在: {model_path}，使用随机初始化的评估器")
        
    def forward(self, render_results):
        """
        对每个局部patch选择最优渲染方法
        
        Args:
            render_results: 列表，包含每个渲染方法的结果
                          [0]: AHDR结果 (范围[0,1])
                          [1]: ExpoMamba结果 (范围[-1,1])
            
        Returns:
            selected_image: 选择最优渲染后的图像
            selection_map: 每个patch选择的渲染方法概率图
            selection_mask: 像素级选择掩码
        """
        batch_size, channels, height, width = render_results[0].shape
        
        # 计算patch数量
        n_patches_h = (height + self.patch_size - 1) // self.patch_size
        n_patches_w = (width + self.patch_size - 1) // self.patch_size
        
        # 初始化输出
        selected_image = torch.zeros_like(render_results[0])
        selection_map = torch.zeros(batch_size, self.n_methods, n_patches_h, n_patches_w, 
                                    device=render_results[0].device)
        selection_mask = torch.zeros(batch_size, height, width, device=render_results[0].device)
        
        # 标准化ExpoMamba结果到[0,1]用于质量评估
        # AHDR已经是[0,1]
        # ExpoMamba需要从[-1,1]转换，但使用自适应归一化
        normalized_results = []
        for idx, render in enumerate(render_results):
            if idx == 1:  # ExpoMamba (现在输出已经是[0,1]范围)
                # 检测ExpoMamba输出是否正常
                dark_ratio = (render < 0.01).float().mean()
                
                # 如果超过50%的像素变黑，使用AHDR的结果
                if dark_ratio > 0.5:
                    normalized_results.append(render_results[0].clone())
                else:
                    # 正常情况，直接使用
                    normalized_results.append(render)
            else:  # AHDR
                normalized_results.append(render)
        
        # 使用非重叠patch选择，但最后用均值滤波平滑边界
        # 遍历每个patch
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # 计算patch边界（非重叠）
                h_start = i * self.patch_size
                h_end = min((i + 1) * self.patch_size, height)
                w_start = j * self.patch_size
                w_end = min((j + 1) * self.patch_size, width)
                
                # 评估每个渲染方法在当前patch上的质量
                quality_scores = []
                for normalized in normalized_results:
                    patch = normalized[:, :, h_start:h_end, w_start:w_end]
                    if patch.size(-2) < 8 or patch.size(-1) < 8:
                        patch = F.interpolate(patch, size=(16, 16), mode='bilinear')
                    score = self.quality_evaluator(patch)
                    quality_scores.append(score)
                
                quality_scores = torch.cat(quality_scores, dim=1)
                
                # 使用温度系数的softmax进行概率选择
                if self.temperature > 0:
                    probs = F.softmax(quality_scores / self.temperature, dim=1)
                    if self.selection_threshold > 0:
                        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                        max_prob_diff = sorted_probs[:, 0] - sorted_probs[:, 1]
                        for b in range(batch_size):
                            if max_prob_diff[b] < self.selection_threshold:
                                probs[b] = torch.tensor([0.5, 0.5], device=probs.device)
                    best_method = torch.multinomial(probs, 1).squeeze(1)
                else:
                    best_method = quality_scores.argmax(dim=1)
                
                # 使用最优方法渲染的patch填充到输出
                for b in range(batch_size):
                    method_idx = int(best_method[b].item())
                    patch_result = render_results[method_idx][b, :, h_start:h_end, w_start:w_end]
                    # ExpoMamba现在输出已经是[0,1]范围，直接使用
                    selected_image[b, :, h_start:h_end, w_start:w_end] = patch_result
                    selection_map[b, :, i, j] = quality_scores[b]
                    selection_mask[b, h_start:h_end, w_start:w_end] = method_idx
        
        # 对整个图像应用高斯模糊以消除所有方块边界
        kernel_size = 5
        sigma = 2.0
        smoothed = self._gaussian_blur(selected_image, kernel_size, sigma)
        
        return smoothed, selection_map, selection_mask


class EnhancedAdaptiveRenderNetwork(nn.Module):
    """
    增强版自适应渲染网络 - 实现用户要求的局部自适应渲染流程

    完整流程：
    1. 输入图像 → 分成局部patch
    2. 对每个patch应用两种渲染方法（AHDR + ExpoMamba）
    3. 使用质量评估网络选择每个patch的最佳渲染结果
    4. 通过平滑过渡整合所有patch
    5. 输出最终渲染图像

    渲染方法：
    1. AHDR（原始HDR增强方法）
    2. ExpoMamba（基于Mamba架构的图像增强）
    """
    def __init__(self, n_channel=3, patch_size=32, trained_quality_model_path=None, 
                 ahdr_checkpoint_path=None, selection_threshold=0.1, temperature=0.5):
        super(EnhancedAdaptiveRenderNetwork, self).__init__()
        self.n_channel = n_channel
        self.patch_size = patch_size

        # 1. 局部Patch渲染器（应用所有渲染方法）
        self.patch_renderer = LocalPatchRenderer(n_channel, patch_size)

        # 加载AHDR权重
        if ahdr_checkpoint_path is not None:
            self.patch_renderer.load_ahdr_weights(ahdr_checkpoint_path)

        # 2. 局部最优选择器（选择每个patch的最佳渲染，可选加载训练好的模型）
        self.local_selector = LocalOptimalSelector(
            n_channel, n_methods=2, patch_size=patch_size,
            trained_model_path=trained_quality_model_path,
            selection_threshold=selection_threshold,
            temperature=temperature
        )

        # 3. 最终整合网络
        self.final_integration = nn.Sequential(
            nn.Conv2d(n_channel, n_channel * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channel * 2, n_channel, kernel_size=3, padding=1)
        )

        # 渲染方法名称
        self.render_method_names = ['AHDR', 'ExpoMamba']

        # 保存训练模型路径
        self.trained_quality_model_path = trained_quality_model_path
    
    def forward(self, x):
        """
        完整的局部自适应渲染流程：
        
        Step 1: 对整个图像应用两种渲染方法
        Step 2: 对每个局部patch选择最优渲染结果
        Step 3: 整合输出最终图像
        """
        batch_size, channels, height, width = x.shape
        
        # ========== Step 1: 渲染阶段 ==========
        # 对整个图像应用所有渲染方法
        render_results = self.patch_renderer(x)  # [ahdr_result, expomamba_result]
        
        # ========== Step 2: 局部最优选择阶段 ==========
        # 对每个局部patch选择最优渲染方法
        selected_image, selection_map, selection_mask = self.local_selector(render_results)
        
        # 直接返回选择后的图像，不经过整合模块（避免输出被压缩）
        final_image = selected_image
        
        # 构建输出字典
        output_dict = {
            'ahdr_result': render_results[0],           # AHDR渲染结果
            'expomamba_result': render_results[1],      # ExpoMamba渲染结果
            'selected_image': selected_image,           # 局部选择后的图像
            'final_image': final_image,                 # 最终整合图像（输入SNorm）
            'selection_map': selection_map,             # patch级选择概率图
            'selection_mask': selection_mask,           # 像素级选择掩码
            'render_method_names': self.render_method_names  # 渲染方法名称
        }
        
        return final_image, output_dict