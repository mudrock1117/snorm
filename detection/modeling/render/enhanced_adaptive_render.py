import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from detection.modeling.HDR.OURS.OHDR import AHDR

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
        self.selection_threshold = selection_threshold
        self.temperature = temperature

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
            nn.Sigmoid()
        )
        
        if trained_model_path is not None:
            self.load_trained_quality_evaluator(trained_model_path)

        self.smooth_transition = nn.Conv2d(n_channel, n_channel,
                                          kernel_size=5, padding=2, groups=n_channel)

    def _gaussian_blur(self, x, kernel_size=5, sigma=2.0):
        batch, channels, height, width = x.shape
        device = x.device
        kernel = self._create_gaussian_kernel(kernel_size, sigma, channels, device)
        padding = kernel_size // 2
        output = F.conv2d(x, kernel, padding=padding, groups=channels)
        return output

    def _create_gaussian_kernel(self, kernel_size, sigma, channels, device):
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
        import os
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'quality_evaluator_state_dict' in checkpoint:
                state_dict = checkpoint['quality_evaluator_state_dict']
            else:
                state_dict = checkpoint
            self.quality_net.load_state_dict(state_dict)
            print(f"✅ 成功加载训练好的质量评估器: {model_path}")
            if 'loss' in checkpoint:
                print(f"   训练损失: {checkpoint['loss']:.6f}")
        else:
            print(f"⚠️ 训练好的模型不存在: {model_path}，使用随机初始化的评估器")
        
    def forward(self, render_results):
        batch_size, channels, height, width = render_results[0].shape
        
        n_patches_h = (height + self.patch_size - 1) // self.patch_size
        n_patches_w = (width + self.patch_size - 1) // self.patch_size
        
        selected_image = torch.zeros_like(render_results[0])
        selection_map = torch.zeros(batch_size, self.n_methods, n_patches_h, n_patches_w, 
                                    device=render_results[0].device)
        selection_mask = torch.zeros(batch_size, height, width, device=render_results[0].device)
        
        normalized_results = []
        for idx, render in enumerate(render_results):
            if idx == 1:
                dark_ratio = (render < 0.01).float().mean()
                if dark_ratio > 0.5:
                    normalized_results.append(render_results[0].clone())
                else:
                    normalized_results.append(render)
            else:
                normalized_results.append(render)
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                h_start = i * self.patch_size
                h_end = min((i + 1) * self.patch_size, height)
                w_start = j * self.patch_size
                w_end = min((j + 1) * self.patch_size, width)
                
                quality_scores = []
                for normalized in normalized_results:
                    patch = normalized[:, :, h_start:h_end, w_start:w_end]
                    if patch.size(-2) < 8 or patch.size(-1) < 8:
                        patch = F.interpolate(patch, size=(16, 16), mode='bilinear')
                    score = self.quality_net(patch)
                    quality_scores.append(score)
                
                quality_scores = torch.cat(quality_scores, dim=1)
                
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
                
                for b in range(batch_size):
                    method_idx = int(best_method[b].item())
                    patch_result = render_results[method_idx][b, :, h_start:h_end, w_start:w_end]
                    selected_image[b, :, h_start:h_end, w_start:w_end] = patch_result
                    selection_map[b, :, i, j] = quality_scores[b]
                    selection_mask[b, h_start:h_end, w_start:w_end] = method_idx
        
        kernel_size = 5
        sigma = 2.0
        smoothed = self._gaussian_blur(selected_image, kernel_size, sigma)
        
        return smoothed, selection_map, selection_mask