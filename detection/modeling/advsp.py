"""Adversarial spectral perturbation for CDFKD (no persistent parameters).

The implementation prefers the modern torch.fft API and falls back to the
legacy torch.rfft/irfft API used by older server PyTorch builds.

Controls: SNORM_ADVSP_PRESET, SNORM_ADVSP_CSG/TSG/SCG, _R, _STEPS,
_TAU, _MU, _LAMBDA, _FREQ_TAU, _OBJECT_PROTECT, _OBJECT_RHO, and
_RANDOM_ONLY (prefix each suffix with SNORM_ADVSP_). Preset defaults are
resolved in cdfkd.py.
"""
import math
import torch
from torch import nn


def _fft2_real(image):
    if hasattr(torch, 'fft') and hasattr(torch.fft, 'fft2'):
        spectrum = torch.fft.fft2(image, dim=(-2, -1), norm='backward')
        return spectrum, True
    if hasattr(torch, 'rfft'):
        return torch.rfft(image, signal_ndim=2, normalized=False,
                          onesided=False), False
    raise RuntimeError('AdvSP requires torch.fft.fft2 or legacy torch.rfft')


def _ifft2_real(spectrum, modern, height, width):
    if modern:
        return torch.fft.ifft2(spectrum, dim=(-2, -1),
                               norm='backward').real
    return torch.irfft(spectrum, signal_ndim=2, normalized=False,
                       onesided=False, signal_sizes=(height, width))


def _legacy_components(spectrum):
    real = spectrum[..., 0]
    imag = spectrum[..., 1]
    amplitude = torch.sqrt(real * real + imag * imag)
    phase = torch.atan2(imag, real)
    return amplitude, phase


def _factor_view(factor, channels, channelwise):
    if factor is None:
        return None
    if factor.dim() == 1:
        factor = factor.view(-1, 1, 1)
    if factor.dim() == 2:
        factor = factor.view(factor.size(0), factor.size(1), 1)
    if factor.size(0) == 1 and channelwise:
        factor = factor.expand(channels, -1, -1)
    return factor


class SpectralPerturbationModule(nn.Module):
    """Apply CSG/TSG/SCG to each valid image extent independently.

    Factors are caller-owned tensors; this module intentionally has no
    Parameters or buffers, so AdvSP state never enters a checkpoint.
    """

    def __init__(self, csg=True, tsg=True, scg=True, tau=0.10,
                 object_protect=False, object_rho=0.5):
        super(SpectralPerturbationModule, self).__init__()
        self.csg = bool(csg)
        self.tsg = bool(tsg)
        self.scg = bool(scg)
        self.tau = float(tau)
        self.object_protect = bool(object_protect)
        self.object_rho = float(object_rho)
        if not 0.0 < self.tau <= 0.5:
            raise ValueError('SNORM_ADVSP_TAU must be in (0, 0.5]')
        if not 0.0 <= self.object_rho <= 1.0:
            raise ValueError('SNORM_ADVSP_OBJECT_RHO must be in [0, 1]')

    @staticmethod
    def object_mask_from_boxes(boxes, height, width, device, dtype):
        """Build a binary Bx1xHxW mask; boxes may be per-image tensors/lists."""
        batch = len(boxes)
        mask = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
        for batch_index, image_boxes in enumerate(boxes):
            if image_boxes is None:
                continue
            if hasattr(image_boxes, 'bbox'):
                image_boxes = image_boxes.bbox
            elif isinstance(image_boxes, dict):
                image_boxes = image_boxes.get('boxes', image_boxes.get('bbox'))
            if image_boxes is None:
                continue
            if not torch.is_tensor(image_boxes):
                image_boxes = torch.as_tensor(image_boxes, device=device)
            image_boxes = image_boxes.detach().to(device=device)
            if image_boxes.numel() == 0:
                continue
            image_boxes = image_boxes.reshape(-1, 4)
            for box in image_boxes:
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))
                if x2 > x1 and y2 > y1:
                    mask[batch_index, 0, y1:y2, x1:x2] = 1.0
        return mask

    def _one(self, image, alpha, beta, gamma):
        channels, height, width = image.shape
        spectrum, modern = _fft2_real(image)
        if modern:
            amplitude = torch.abs(spectrum)
            phase = torch.angle(spectrum)
        else:
            amplitude, phase = _legacy_components(spectrum)

        fy = torch.minimum(
            torch.arange(height, device=image.device, dtype=image.dtype),
            height - torch.arange(height, device=image.device, dtype=image.dtype)
        ) / float(height)
        fx = torch.minimum(
            torch.arange(width, device=image.device, dtype=image.dtype),
            width - torch.arange(width, device=image.device, dtype=image.dtype)
        ) / float(width)
        radius = torch.sqrt(fy[:, None] * fy[:, None] + fx[None, :] * fx[None, :])
        low = (radius <= self.tau).to(dtype=image.dtype).view(1, height, width)
        high = (1.0 - low)

        if self.csg and alpha is not None:
            a = _factor_view(alpha, channels, True)
            amplitude = amplitude * (1.0 + a * low)
        if self.tsg and beta is not None:
            b = _factor_view(beta, channels, True)
            amplitude = amplitude * (1.0 + b * high)
        if self.scg and gamma is not None:
            g = _factor_view(gamma, channels, True)
            phase = phase * (1.0 + g)

        if modern:
            if hasattr(torch, 'polar'):
                perturbed = torch.polar(amplitude, phase)
            else:
                perturbed = torch.complex(
                    amplitude * torch.cos(phase),
                    amplitude * torch.sin(phase),
                )
        else:
            perturbed = torch.stack(
                (amplitude * torch.cos(phase), amplitude * torch.sin(phase)),
                dim=-1
            )
        return _ifft2_real(perturbed, modern, height, width)

    def forward(self, images, alpha=None, beta=None, gamma=None,
                image_sizes=None, boxes=None):
        if images.dim() != 4:
            raise ValueError('AdvSP expects BCHW images')
        batch, channels, padded_h, padded_w = images.shape
        if image_sizes is None:
            image_sizes = [(padded_h, padded_w)] * batch
        if len(image_sizes) != batch:
            raise ValueError('image_sizes batch does not match images')
        outputs = []
        for index, size in enumerate(image_sizes):
            height, width = int(size[0]), int(size[1])
            if height <= 0 or width <= 0 or height > padded_h or width > padded_w:
                raise ValueError('invalid AdvSP valid image extent: %s' % (size,))
            crop = images[index, :, :height, :width]
            a = None if alpha is None else alpha[index]
            b = None if beta is None else beta[index]
            g = None if gamma is None else gamma[index]
            perturbed = self._one(crop, a, b, g)
            if self.object_protect:
                if boxes is None:
                    raise ValueError('object-protect AdvSP requires GT boxes')
                mask = self.object_mask_from_boxes(
                    [boxes[index]], height, width, images.device, images.dtype
                )
                mask = mask[0]
                perturbed = mask * (
                    self.object_rho * crop + (1.0 - self.object_rho) * perturbed
                ) + (1.0 - mask) * perturbed
            if width < padded_w or height < padded_h:
                padded = images[index].new_zeros(
                    (channels, padded_h, padded_w)
                )
                padded[:, :height, :width] = perturbed
                perturbed = padded
            outputs.append(perturbed)
        return torch.stack(outputs, dim=0)


def make_spectral_factors(batch, device, dtype, radius, csg, tsg, scg):
    """Create temporary per-rank factors (never model Parameters)."""
    alpha = (torch.empty(batch, 3, device=device, dtype=dtype)
             .uniform_(-radius, radius).requires_grad_(True)
             if csg else None)
    beta = (torch.empty(batch, 1, device=device, dtype=dtype)
            .uniform_(-radius, radius).requires_grad_(True)
            if tsg else None)
    gamma = (torch.empty(batch, 3, device=device, dtype=dtype)
             .uniform_(-radius, radius).requires_grad_(True)
             if scg else None)
    return alpha, beta, gamma


def update_spectral_factors(factors, gradients, radius, tau):
    """One or more detached projected-gradient-ascent updates."""
    updated = []
    for factor, gradient in zip(factors, gradients):
        if factor is None:
            updated.append(None)
        elif not factor.requires_grad or gradient is None:
            updated.append(None if factor is None else factor.detach())
        else:
            updated.append((factor + float(tau) * gradient).clamp(
                -float(radius), float(radius)
            ).detach())
    return tuple(updated)

