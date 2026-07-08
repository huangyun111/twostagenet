"""Direct U-Net++ baseline for RGB/S0 -> polarization encoding.

This baseline is intentionally simpler than Stage1 PolarPriorNet: it predicts
only [DoLP, cos2AoLP, sin2AoLP] and has no uncertainty/confidence head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DirectUnetPlusPlusBaseline(nn.Module):
    """Plain deterministic U-Net++ baseline with physical output projection."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.eps = eps
        self.net = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=3,
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        raw = self.net(rgb)
        dolp = torch.sigmoid(raw[:, 0:1])
        raw_cos_sin = raw[:, 1:3]
        norm = torch.sqrt((raw_cos_sin * raw_cos_sin).sum(dim=1, keepdim=True)).clamp_min(self.eps)
        cos_sin = raw_cos_sin / norm
        return torch.cat((dolp, cos_sin), dim=1).contiguous()
