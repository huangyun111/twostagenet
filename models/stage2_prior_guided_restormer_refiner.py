"""Prior-guided Restormer refiner for Stage 2 polarization."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.direct_restormer_baseline import Downsample, Upsample, make_blocks


class Stage2PriorGuidedRestormerRefiner(nn.Module):
    """Version D: reliability-gated coarse-to-fine polarization refinement.

    RGB, prior, and confidence use separate stems. The prior is injected at all
    three Restormer scales through confidence-guided feature modulation. Output
    refinement uses independent DoLP/AoLP gates and applies the AoLP residual
    directly on the circular angle manifold.
    """

    def __init__(
        self,
        in_channels_rgb: int = 3,
        in_channels_prior: int = 3,
        in_channels_confidence: int = 3,
        dim: int = 32,
        num_blocks: tuple[int, int, int] = (2, 2, 4),
        num_heads: tuple[int, int, int] = (1, 2, 4),
        expansion: float = 2.66,
        residual_scale: float = 0.5,
        angle_residual_scale: float = math.pi / 2.0,
        min_gate: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 3 or len(num_heads) != 3:
            raise ValueError("num_blocks and num_heads must each have 3 entries.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if residual_scale <= 0.0 or angle_residual_scale <= 0.0:
            raise ValueError("residual scales must be positive.")
        if not 0.0 <= min_gate <= 1.0:
            raise ValueError("min_gate must be in [0, 1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.residual_scale = residual_scale
        self.angle_residual_scale = angle_residual_scale
        self.min_gate = min_gate
        self.eps = eps
        self.rgb_stem = nn.Conv2d(in_channels_rgb, dim, kernel_size=3, padding=1)
        self.prior_stem = nn.Conv2d(in_channels_prior, dim, kernel_size=3, padding=1)
        self.confidence_stem = nn.Conv2d(
            in_channels_confidence,
            dim,
            kernel_size=3,
            padding=1,
        )
        self.prior_down1 = Downsample(dim)
        self.prior_down2 = Downsample(dim * 2)
        self.confidence_down1 = Downsample(dim)
        self.confidence_down2 = Downsample(dim * 2)
        self.guidance_gate1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.guidance_gate2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        self.guidance_gate3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=1)

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

        self.delta_dolp_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.delta_angle_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.gate_dolp_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.gate_angle_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

        for layer in (self.guidance_gate1, self.guidance_gate2, self.guidance_gate3):
            self._zero_init(layer)
        # The initial network is exactly the Stage1 identity mapping. This keeps
        # a newly initialized refiner from damaging a useful prior.
        self._zero_init(self.delta_dolp_head)
        self._zero_init(self.delta_angle_head)
        self._zero_init(self.gate_dolp_head)
        self._zero_init(self.gate_angle_head)

    def forward(
        self,
        rgb: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = rgb.shape[-2:]
        rgb, prior, confidence = self._pad_inputs(rgb, prior, confidence)

        prior_reliability = confidence.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        prior1 = self.prior_stem(prior)
        confidence1 = self.confidence_stem(confidence)
        x1 = self._guided_fusion(
            self.rgb_stem(rgb),
            prior1,
            confidence1,
            prior_reliability,
            self.guidance_gate1,
        )
        x1 = self.encoder1(x1)

        prior2 = self.prior_down1(prior1)
        confidence2 = self.confidence_down1(confidence1)
        x2 = self._guided_fusion(
            self.down1(x1),
            prior2,
            confidence2,
            prior_reliability,
            self.guidance_gate2,
        )
        x2 = self.encoder2(x2)

        prior3 = self.prior_down2(prior2)
        confidence3 = self.confidence_down2(confidence2)
        latent = self._guided_fusion(
            self.down2(x2),
            prior3,
            confidence3,
            prior_reliability,
            self.guidance_gate3,
        )
        latent = self.latent(latent)

        y = self.up2(latent)
        y = self.decoder2(self.reduce2(torch.cat([y, x2], dim=1)))
        y = self.up1(y)
        y = self.decoder1(self.reduce1(torch.cat([y, x1], dim=1)))
        y = y[..., :height, :width]
        prior = prior[..., :height, :width]
        confidence = confidence[..., :height, :width]

        raw_delta_dolp = self.delta_dolp_head(y)
        raw_delta_angle = self.delta_angle_head(y)
        delta_dolp = self.residual_scale * torch.tanh(raw_delta_dolp)
        delta_angle = self.angle_residual_scale * torch.tanh(raw_delta_angle)

        gate_dolp_logits = self.gate_dolp_head(y)
        gate_angle_logits = self.gate_angle_head(y)
        confidence_dolp = confidence[:, 0:1].clamp(0.0, 1.0)
        confidence_angle = confidence[:, 1:3].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        demand_dolp = self.min_gate + (1.0 - self.min_gate) * (1.0 - confidence_dolp)
        demand_angle = self.min_gate + (1.0 - self.min_gate) * (1.0 - confidence_angle)
        gate_dolp = torch.sigmoid(gate_dolp_logits) * demand_dolp
        gate_angle = torch.sigmoid(gate_angle_logits) * demand_angle

        candidate_dolp = torch.clamp(prior[:, 0:1] + delta_dolp, 0.0, 1.0)
        theta_prior = 0.5 * torch.atan2(prior[:, 2:3], prior[:, 1:2])
        theta_candidate = theta_prior + delta_angle
        candidate_cos = torch.cos(2.0 * theta_candidate)
        candidate_sin = torch.sin(2.0 * theta_candidate)

        dolp = torch.clamp(prior[:, 0:1] + gate_dolp * delta_dolp, 0.0, 1.0)
        theta_final = theta_prior + gate_angle * delta_angle
        cos2 = torch.cos(2.0 * theta_final)
        sin2 = torch.sin(2.0 * theta_final)
        refined = torch.cat([dolp, cos2, sin2], dim=1)
        refinement_gate = 0.5 * (gate_dolp + gate_angle)

        return {
            "refined": refined.contiguous(),
            "candidate": torch.cat([candidate_dolp, candidate_cos, candidate_sin], dim=1).contiguous(),
            "delta_dolp": delta_dolp.contiguous(),
            "delta_angle": delta_angle.contiguous(),
            "gate_dolp": gate_dolp.contiguous(),
            "gate_angle": gate_angle.contiguous(),
            "refinement_gate": refinement_gate.contiguous(),
            "raw_delta_dolp": raw_delta_dolp.contiguous(),
            "raw_delta_angle": raw_delta_angle.contiguous(),
            "gate_dolp_logits": gate_dolp_logits.contiguous(),
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
        prior_features: torch.Tensor,
        confidence_features: torch.Tensor,
        prior_reliability: torch.Tensor,
        guidance_gate: nn.Conv2d,
    ) -> torch.Tensor:
        reliability = F.interpolate(
            prior_reliability,
            size=image_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        learned_scale = 2.0 * torch.sigmoid(guidance_gate(confidence_features))
        modulation = (reliability * learned_scale).clamp(0.0, 1.0)
        return image_features + modulation * prior_features

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
