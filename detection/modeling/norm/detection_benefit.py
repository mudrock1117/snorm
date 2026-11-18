import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionBenefitPredictor(nn.Module):
    """Sample-level AHDR benefit predictor using inference-time statistics."""

    def __init__(self, channels=3, hidden_channels=16, initial_alpha=0.9):
        super().__init__()
        if initial_alpha <= 0.0 or initial_alpha >= 1.0:
            raise ValueError('initial_alpha must be strictly between 0 and 1')
        self.encoder = nn.Linear(channels * 6, hidden_channels)
        self.predictor = nn.Linear(hidden_channels, 1)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.predictor.weight)
        initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
        nn.init.constant_(self.predictor.bias, initial_logit)

    @staticmethod
    def _mean_std(x):
        mean = x.mean(dim=(2, 3))
        variance = (x - mean[:, :, None, None]).pow(2).mean(dim=(2, 3))
        return mean, (variance + 1e-6).sqrt()

    def forward(self, base, ahdr):
        base = base.detach()
        ahdr = ahdr.detach()
        delta = ahdr - base
        base_mean, base_std = self._mean_std(base)
        ahdr_mean, ahdr_std = self._mean_std(ahdr)
        delta_mean, delta_std = self._mean_std(delta)
        descriptor = torch.cat([
            base_mean, base_std, ahdr_mean, ahdr_std,
            delta_mean, delta_std,
        ], dim=1)
        hidden = F.relu(self.encoder(descriptor), inplace=False)
        logits = self.predictor(hidden)
        alpha = torch.sigmoid(logits)
        return alpha[:, :, None, None], logits[:, 0], descriptor

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class DetectionSemanticBenefitPredictor(nn.Module):
    """C3-V2 selector built from existing detector semantic features."""

    def __init__(self, in_channels=1024, projection_channels=32,
                 hidden_channels=32, initial_alpha=0.9):
        super().__init__()
        if initial_alpha <= 0.0 or initial_alpha >= 1.0:
            raise ValueError('initial_alpha must be strictly between 0 and 1')
        self.projection = nn.Linear(in_channels, projection_channels)
        self.encoder = nn.Linear(projection_channels * 3, hidden_channels)
        self.predictor = nn.Linear(hidden_channels, 1)
        nn.init.kaiming_uniform_(self.projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.projection.bias)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.predictor.weight, mean=0.0, std=1e-3)
        initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
        nn.init.constant_(self.predictor.bias, initial_logit)

    def _project(self, feature):
        pooled = F.adaptive_avg_pool2d(feature.detach(), 1).flatten(1)
        return F.relu(self.projection(pooled), inplace=False)

    def forward(self, base_feature, ahdr_feature):
        base_projected = self._project(base_feature)
        ahdr_projected = self._project(ahdr_feature)
        delta_projected = ahdr_projected - base_projected
        descriptor = torch.cat([
            base_projected, ahdr_projected, delta_projected
        ], dim=1)
        hidden = F.relu(self.encoder(descriptor), inplace=False)
        logits = self.predictor(hidden)
        alpha = torch.sigmoid(logits)
        return alpha[:, :, None, None], logits[:, 0], descriptor

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def detection_benefit_loss(logits, detached_benefit, margin=0.0):
    target = (detached_benefit > float(margin)).to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    prediction = (logits >= 0.0).to(dtype=target.dtype)
    accuracy = (prediction == target).to(dtype=logits.dtype).mean()
    return loss, target, accuracy
