import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian(window_size, sigma):
    atx = [np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)]
    gauss = torch.from_numpy(np.array(atx))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2, weights=None):
        # img1, img2: [B,C,H,W], range [0,1]
        (_, channel, _, _) = img1.size()
        window = create_window(self.window_size, channel).to(img1.device)
        window = window.float()

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        '''
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)
        '''

        loss_ssim_fg = (ssim_map * weights).sum() / (weights.sum() + 1e-5)
        loss_ssim_bg = (ssim_map * (1-weights)).sum() / ((1-weights).sum() + 1e-5)
        loss_ssim = loss_ssim_fg + loss_ssim_bg
        return loss_ssim


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        # Sobel kernels
        self.kernel_x = torch.tensor([[-1,0,1],
                                      [-2,0,2],
                                      [-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.tensor([[-1,-2,-1],
                                      [0,0,0],
                                      [1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, img1, img2, weights=None):
        """
        img1, img2: [B,C,H,W], range [0,1]
        """
        B, C, H, W = img1.shape
        device = img1.device

        # repeat kernel for multi-channel
        kernel_x = self.kernel_x.to(device).repeat(C,1,1,1)
        kernel_y = self.kernel_y.to(device).repeat(C,1,1,1)

        grad_x1 = F.conv2d(img1, kernel_x, padding=1, groups=C)
        grad_y1 = F.conv2d(img1, kernel_y, padding=1, groups=C)
        grad_x2 = F.conv2d(img2, kernel_x, padding=1, groups=C)
        grad_y2 = F.conv2d(img2, kernel_y, padding=1, groups=C)

        edge1 = torch.sqrt(grad_x1**2 + grad_y1**2 + 1e-6)
        edge2 = torch.sqrt(grad_x2**2 + grad_y2**2 + 1e-6)

        loss_mask = F.l1_loss(edge1, edge2, reduction='none')
        loss_edge_fg = (loss_mask * weights).sum() / (weights.sum() + 1e-5)
        loss_edge_bg = (loss_mask * (1-weights)).sum() / ((1-weights).sum() + 1e-5)
        loss_edge = loss_edge_fg + loss_edge_bg
        return loss_edge
        