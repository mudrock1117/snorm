import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from detection.modeling.HDR.OURS.OHDR import AHDR

class IntegratedAdaptiveRender(nn.Module):
    """
    整合版自适应渲染网络
    - 完全采用三输入模式，兼容原项目中AHDR的调用方式
    - 暂时禁用增强版自适应渲染以避免兼容性问题
    """

    def __init__(self, in_ch=3, base_ch=64):
        super(IntegratedAdaptiveRender, self).__init__()

        self.ahdr = AHDR(in_ch=in_ch, base_ch=base_ch)

        self.expmamba_operator = None

    def _get_expmamba_operator(self):
        if self.expmamba_operator is None:
            try:
                from expmamba_operator import get_expmamba_operator
                self.expmamba_operator = get_expmamba_operator()
            except Exception as e:
                print(f"⚠️ ExpoMamba加载失败: {e}")
                self.expmamba_operator = None
        return self.expmamba_operator

    def load_weights(self, checkpoint_path):
        """加载预训练权重到AHDR"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_state_dict = checkpoint['model']

            render_net_state_dict = {k.replace('teacher.proj_norm.render_net.', ''): v
                                    for k, v in model_state_dict.items()
                                    if 'teacher.proj_norm.render_net' in k}

            if render_net_state_dict:
                self.ahdr.load_state_dict(render_net_state_dict, strict=False)
                print(f"✅ 成功加载AHDR权重 ({len(render_net_state_dict)}个参数)")

    def forward(self, x1, x2, x3):
        """
        三输入模式：
        1. 先使用原始AHDR处理三个输入特征图

        Args:
            x1: 低层特征（提供细节信息）
            x2: 中间特征（提供上下文信息）
            x3: 高层特征（提供全局结构信息）

        Returns:
            AHDR处理后的结果
        """
        ahdr_output = self.ahdr(x1, x2, x3)

        if isinstance(ahdr_output, tuple):
            ahdr_result = ahdr_output[0]
        else:
            ahdr_result = ahdr_output

        return ahdr_result
