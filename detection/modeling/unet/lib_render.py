import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocalAttention(nn.Module):
    def __init__(self, dim, window_size=7, heads=8):
        super().__init__()
        self.dim = dim
        self.window = window_size
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)   # → B, H, W, C

        # === 1. 按 window 划分 ===
        # pad 保证 H, W 是 window_size 的整数倍
        pad_h = (self.window - H % self.window) % self.window
        pad_w = (self.window - W % self.window) % self.window

        x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = x.shape[1], x.shape[2]

        # 划分窗口 B * num_windows, window, window, C
        x_windows = x.view(
            B,
            H_pad // self.window, self.window,
            W_pad // self.window, self.window,
            C
        )
        x_windows = x_windows.permute(0,1,3,2,4,5).contiguous().view(
            -1, self.window * self.window, C
        )

        # === 2. QKV ===
        qkv = self.qkv(x_windows).reshape(-1, self.window*self.window, 3, self.heads, C//self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # 3, B*nwin, heads, tokens, dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        # === 3. 注意力 ===
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)   # B*nwin, heads, tokens, dim
        out = out.transpose(1, 2).reshape(-1, self.window*self.window, C)

        # === 4. 恢复窗口到图像 ===
        out = out.view(
            B,
            H_pad // self.window, W_pad // self.window,
            self.window, self.window, C
        ).permute(0,1,3,2,4,5).contiguous().view(
            B, H_pad, W_pad, C
        )

        # remove padding
        out = out[:, :H, :W, :]

        return self.proj(out.permute(0, 3, 1, 2))   # → B,C,H,W
        