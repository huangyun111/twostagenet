"""Version E asymmetric prior-guided Restormer polarization refiner."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.direct_restormer_baseline import Downsample, Upsample, make_blocks


class Stage2AsymmetricRestormerRefiner(nn.Module):
    """Refine scalar DoLP and circular AoLP with geometry-specific branches.

    The AoLP path preserves Version D's gated circular residual.  The DoLP path
    uses a zero-initialized, ungated residual so its gradients cannot disappear
    when a learned gate closes.  DoLP and angle priors are encoded separately at
    every scale, and a light high-resolution adapter restores local DoLP detail.
    """

    def __init__(
        self,
        in_channels_rgb: int = 3,
        dim: int = 32,
        num_blocks: tuple[int, int, int] = (2, 2, 4),
        num_heads: tuple[int, int, int] = (1, 2, 4),
        expansion: float = 2.66,
        dolp_residual_scale: float = 0.25,
        angle_residual_scale: float = math.pi / 2.0,
        min_angle_gate: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 3 or len(num_heads) != 3:
            raise ValueError("num_blocks and num_heads must each have 3 entries.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if dolp_residual_scale <= 0.0 or angle_residual_scale <= 0.0:
            raise ValueError("residual scales must be positive.")
        if not 0.0 <= min_angle_gate <= 1.0:
            raise ValueError("min_angle_gate must be in [0, 1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.dolp_residual_scale = dolp_residual_scale
        self.angle_residual_scale = angle_residual_scale
        self.min_angle_gate = min_angle_gate
        self.eps = eps

        self.rgb_stem = nn.Conv2d(in_channels_rgb, dim, kernel_size=3, padding=1)
        # DoLP is mapped from [0, 1] to [-1, 1] before this two-channel stem.
        self.dolp_prior_stem = nn.Conv2d(2, dim, kernel_size=3, padding=1)
        # Angle keeps its native unit-vector representation plus confidence.
        self.angle_prior_stem = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        self.dolp_down1 = Downsample(dim)
        self.dolp_down2 = Downsample(dim * 2)
        self.angle_down1 = Downsample(dim)
        self.angle_down2 = Downsample(dim * 2)

        self.dolp_guidance = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
                nn.Conv2d(dim * 4, dim * 4, kernel_size=1),
            ]
        )
        self.angle_guidance = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
                nn.Conv2d(dim * 4, dim * 4, kernel_size=1),
            ]
        )

        self.encoder1 = make_blocks(dim, num_blocks[0], num_heads[0], expansion)
        self.down1 = Downsample(dim)
        self.encoder2 = make_blocks(dim * 2, num_blocks[1], num_heads[1], expansion)
        self.down2 = Downsample(dim * 2)
        self.latent = make_blocks(dim * 4, num_blocks[2], num_heads[2], expansion)

        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder2 = make_blocks(dim * 2, num_blocks[1], num_heads[1], expansion)
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder1 = make_blocks(dim, num_blocks[0], num_heads[0], expansion)

        # Full-resolution DoLP evidence: decoder, RGB, DoLP prior features,
        # scaled raw DoLP, and its confidence.
        self.dolp_adapter = nn.Sequential(
            nn.Conv2d(dim * 3 + 2, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.delta_dolp_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.delta_angle_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.gate_angle_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

        for layer in (*self.dolp_guidance, *self.angle_guidance):
            self._zero_init(layer)
        # Identity at initialization without a DoLP gate.
        self._zero_init(self.delta_dolp_head)
        self._zero_init(self.delta_angle_head)
        self._zero_init(self.gate_angle_head)

    def forward(
        self,
        rgb: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = rgb.shape[-2:]
        rgb, prior, confidence = self._pad_inputs(rgb, prior, confidence)

        prior_dolp = prior[:, 0:1].clamp(0.0, 1.0)
        confidence_dolp = confidence[:, 0:1].clamp(0.0, 1.0)
        confidence_angle = confidence[:, 1:3].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        scaled_dolp = 2.0 * prior_dolp - 1.0
        scaled_confidence_dolp = 2.0 * confidence_dolp - 1.0
        scaled_confidence_angle = 2.0 * confidence_angle - 1.0

        rgb1 = self.rgb_stem(rgb)
        dolp1 = self.dolp_prior_stem(torch.cat([scaled_dolp, scaled_confidence_dolp], dim=1))
        angle1 = self.angle_prior_stem(
            torch.cat([prior[:, 1:3], scaled_confidence_angle], dim=1)
        )
        x1 = self._guided_fusion(
            rgb1,
            dolp1,
            angle1,
            confidence_dolp,
            confidence_angle,
            self.dolp_guidance[0],
            self.angle_guidance[0],
        )
        x1 = self.encoder1(x1)

        dolp2 = self.dolp_down1(dolp1)
        angle2 = self.angle_down1(angle1)
        x2 = self._guided_fusion(
            self.down1(x1),
            dolp2,
            angle2,
            confidence_dolp,
            confidence_angle,
            self.dolp_guidance[1],
            self.angle_guidance[1],
        )
        x2 = self.encoder2(x2)

        dolp3 = self.dolp_down2(dolp2)
        angle3 = self.angle_down2(angle2)
        latent = self._guided_fusion(
            self.down2(x2),
            dolp3,
            angle3,
            confidence_dolp,
            confidence_angle,
            self.dolp_guidance[2],
            self.angle_guidance[2],
        )
        latent = self.latent(latent)

        y = self.up2(latent)
        y = self.decoder2(self.reduce2(torch.cat([y, x2], dim=1)))
        y = self.up1(y)
        y = self.decoder1(self.reduce1(torch.cat([y, x1], dim=1)))

        y = y[..., :height, :width]
        rgb1 = rgb1[..., :height, :width]
        dolp1 = dolp1[..., :height, :width]
        prior = prior[..., :height, :width]
        prior_dolp = prior_dolp[..., :height, :width]
        confidence_dolp = confidence_dolp[..., :height, :width]
        confidence_angle = confidence_angle[..., :height, :width]
        scaled_dolp = scaled_dolp[..., :height, :width]

        dolp_features = y + self.dolp_adapter(
            torch.cat([y, rgb1, dolp1, scaled_dolp, confidence_dolp], dim=1)
        )
        raw_delta_dolp = self.delta_dolp_head(dolp_features)
        raw_delta_angle = self.delta_angle_head(y)
        delta_dolp = self.dolp_residual_scale * torch.tanh(raw_delta_dolp)
        delta_angle = self.angle_residual_scale * torch.tanh(raw_delta_angle)

        gate_angle_logits = self.gate_angle_head(y)
        angle_demand = self.min_angle_gate + (1.0 - self.min_angle_gate) * (
            1.0 - confidence_angle
        )
        gate_angle = torch.sigmoid(gate_angle_logits) * angle_demand

        dolp = torch.clamp(prior_dolp + delta_dolp, 0.0, 1.0)
        theta_prior = 0.5 * torch.atan2(prior[:, 2:3], prior[:, 1:2])
        theta_candidate = theta_prior + delta_angle
        theta_final = theta_prior + gate_angle * delta_angle
        refined = torch.cat(
            [dolp, torch.cos(2.0 * theta_final), torch.sin(2.0 * theta_final)],
            dim=1,
        )
        candidate = torch.cat(
            [dolp, torch.cos(2.0 * theta_candidate), torch.sin(2.0 * theta_candidate)],
            dim=1,
        )

        return {
            "refined": refined.contiguous(),
            "candidate": candidate.contiguous(),
            "delta_dolp": delta_dolp.contiguous(),
            "delta_angle": delta_angle.contiguous(),
            "gate_angle": gate_angle.contiguous(),
            "refinement_gate": gate_angle.contiguous(),
            "raw_delta_dolp": raw_delta_dolp.contiguous(),
            "raw_delta_angle": raw_delta_angle.contiguous(),
            "gate_angle_logits": gate_angle_logits.contiguous(),
        }

    @staticmethod
    def _zero_init(layer: nn.Conv2d) -> None:
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _guided_fusion(
        image_features: torch.Tensor,
        dolp_features: torch.Tensor,
        angle_features: torch.Tensor,
        confidence_dolp: torch.Tensor,
        confidence_angle: torch.Tensor,
        dolp_gate: nn.Conv2d,
        angle_gate: nn.Conv2d,
    ) -> torch.Tensor:
        size = image_features.shape[-2:]
        reliability_dolp = F.interpolate(
            confidence_dolp, size=size, mode="bilinear", align_corners=False
        )
        reliability_angle = F.interpolate(
            confidence_angle, size=size, mode="bilinear", align_corners=False
        )
        modulation_dolp = (
            reliability_dolp * (2.0 * torch.sigmoid(dolp_gate(dolp_features)))
        ).clamp(0.0, 1.0)
        modulation_angle = (
            reliability_angle * (2.0 * torch.sigmoid(angle_gate(angle_features)))
        ).clamp(0.0, 1.0)
        return (
            image_features
            + modulation_dolp * dolp_features
            + modulation_angle * angle_features
        )

    @staticmethod
    def _pad_inputs(
        rgb: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[-2:]
        pad_h = (4 - height % 4) % 4
        pad_w = (4 - width % 4) % 4
        if not pad_h and not pad_w:
            return rgb, prior, confidence
        padding = (0, pad_w, 0, pad_h)
        return (
            F.pad(rgb, padding, mode="reflect"),
            F.pad(prior, padding, mode="reflect"),
            F.pad(confidence, padding, mode="reflect"),
        )
