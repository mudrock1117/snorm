"""Non-mutating tensor metrics for SNorm/AHDR diagnostics."""
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _number(value: torch.Tensor) -> float:
    value = value.detach().float()
    value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    return float(value.cpu().item())


@torch.no_grad()
def high_frequency_energy(value: torch.Tensor) -> float:
    x = value.detach().float()
    if x.ndim < 2 or x.shape[-2] < 2 or x.shape[-1] < 2:
        return 0.0
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return _number((dx + dy) * 0.5 / x.abs().mean().clamp_min(1e-12))


@torch.no_grad()
def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("cosine inputs must have identical shapes")
    a = left.detach().float().reshape(1, -1)
    b = right.detach().float().reshape(1, -1)
    return _number(F.cosine_similarity(a, b, dim=1, eps=1e-12).mean())


@torch.no_grad()
def tensor_metrics(
    value: torch.Tensor, reference: Optional[torch.Tensor] = None
) -> Dict[str, object]:
    x = value.detach().float()
    result: Dict[str, object] = {
        "shape": list(x.shape),
        "mean": _number(x.mean()) if x.numel() else 0.0,
        "std": _number(x.std(unbiased=False)) if x.numel() else 0.0,
        "l2": _number(torch.norm(x)) if x.numel() else 0.0,
        "abs_max": _number(x.abs().max()) if x.numel() else 0.0,
        "high_frequency_energy": high_frequency_energy(x),
    }
    if reference is not None:
        if reference.shape != value.shape:
            raise ValueError("reference must have the same shape as value")
        ref = reference.detach().float()
        delta = x - ref
        result.update(
            cosine_to_reference=cosine_similarity(x, ref),
            change_l2=_number(torch.norm(delta)),
            change_rate=_number(
                torch.norm(delta)
                / torch.norm(ref).clamp_min(1e-12)
            ),
        )
    return result


@torch.no_grad()
def region_response(feature: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, float]:
    """Compare absolute target and background response with a binary mask."""
    if feature.ndim != 4:
        raise ValueError("feature must be NCHW")
    mask = target_mask.detach()
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[0] not in (1, feature.shape[0]):
        raise ValueError("target_mask must be HW, NHW, or N1HW")
    mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
    if mask.shape[0] == 1 and feature.shape[0] != 1:
        mask = mask.expand(feature.shape[0], -1, -1, -1)
    mask = mask.clamp(0.0, 1.0)
    response = feature.detach().float().abs().mean(dim=1, keepdim=True)
    target_mean = (response * mask).sum() / mask.sum().clamp_min(1.0)
    background = 1.0 - mask
    background_mean = (response * background).sum() / background.sum().clamp_min(1.0)
    return {
        "target_mean": _number(target_mean),
        "background_mean": _number(background_mean),
        "target_background_ratio": _number(
            target_mean / background_mean.clamp_min(1e-12)
        ),
    }
