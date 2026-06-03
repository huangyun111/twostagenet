"""Confidence-guided gated residual refiner for Stage 2 polarization."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class ConfidenceGuidedResidualRefiner(nn.Module):
    """Refine a Stage 1 polarization prior with confidence-gated residuals."""

    def __init__(
        self,
        in_channels_rgb: int = 3,
        in_channels_prior: int = 3,
        in_channels_confidence: int = 3,
        base_channels: int = 64,
        residual_scale: float = 0.3,
        min_gate: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if not 0.0 <= min_gate <= 1.0:
            raise ValueError("min_gate must be in [0, 1].")

        self.residual_scale = residual_scale
        self.min_gate = min_gate
        self.eps = eps

        self.rgb_stem = nn.Sequential(
            ConvGNAct(in_channels_rgb, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.prior_stem = nn.Sequential(
            ConvGNAct(in_channels_prior, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.confidence_stem = nn.Sequential(
            ConvGNAct(in_channels_confidence, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.fuse_conv = nn.Sequential(
            ConvGNAct(base_channels * 3, base_channels),
            ResidualBlock(base_channels, base_channels),
        )

        self.down1 = ConvGNAct(base_channels, base_channels * 2, stride=2)
        self.enc1 = ResidualBlock(base_channels * 2, base_channels * 2)
        self.down2 = ConvGNAct(base_channels * 2, base_channels * 4, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec1 = ResidualBlock(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec2 = ResidualBlock(base_channels * 2, base_channels)

        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.gate_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(
        self,
        rgb: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rgb_feat = self.rgb_stem(rgb)
        prior_feat = self.prior_stem(prior)
        confidence_feat = self.confidence_stem(confidence)
        fused = self.fuse_conv(torch.cat([rgb_feat, prior_feat, confidence_feat], dim=1))

        enc0 = fused
        enc1 = self.enc1(self.down1(enc0))
        bottleneck = self.bottleneck(self.down2(enc1))

        dec1 = self.up1(bottleneck)
        dec1 = self._match_size(dec1, enc1)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        dec2 = self.up2(dec1)
        dec2 = self._match_size(dec2, enc0)
        features = self.dec2(torch.cat([dec2, enc0], dim=1))

        raw_residual = self.residual_head(features)
        bounded_residual = self.residual_scale * torch.tanh(raw_residual)
        learned_gate = torch.sigmoid(self.gate_head(features))
        confidence_gate = self.min_gate + (1.0 - self.min_gate) * (1.0 - confidence)
        gate = learned_gate * confidence_gate
        residual = bounded_residual * gate

        dolp_final = (prior[:, 0:1] + residual[:, 0:1]).clamp(0.0, 1.0)
        cos_sin_raw = prior[:, 1:3] + residual[:, 1:3]
        norm = torch.sqrt((cos_sin_raw * cos_sin_raw).sum(dim=1, keepdim=True) + self.eps)
        cos_sin_final = cos_sin_raw / norm
        refined = torch.cat([dolp_final, cos_sin_final], dim=1).contiguous()

        return {
            "refined": refined,
            "residual": residual,
            "raw_residual": raw_residual,
            "gate": gate,
        }

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
