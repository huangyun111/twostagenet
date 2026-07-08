"""Prior-guided Restormer refiner for Stage 2 polarization."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.direct_restormer_baseline import Downsample, Upsample, make_blocks


class Stage2PriorGuidedRestormerRefiner(nn.Module):
    """Version C: refine a coarse polarization prior with a Restormer backbone.

    The network consumes image evidence, Stage1 prior, and Stage1 confidence.
    It predicts explicit DoLP and AoLP residuals plus a single refinement gate,
    then fuses the residual-updated candidate with the original prior.
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
        angle_residual_scale: float = math.pi,
        min_gate: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 3 or len(num_heads) != 3:
            raise ValueError("num_blocks and num_heads must each have 3 entries.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if not 0.0 <= min_gate <= 1.0:
            raise ValueError("min_gate must be in [0, 1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.residual_scale = residual_scale
        self.angle_residual_scale = angle_residual_scale
        self.min_gate = min_gate
        self.eps = eps
        in_channels = in_channels_rgb + in_channels_prior + in_channels_confidence

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
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
        self.gate_head = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

    def forward(
        self,
        rgb: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = rgb.shape[-2:]
        rgb, prior, confidence = self._pad_inputs(rgb, prior, confidence)

        x = torch.cat([rgb, prior, confidence], dim=1)
        x1 = self.encoder1(self.patch_embed(x))
        x2 = self.encoder2(self.down1(x1))
        latent = self.latent(self.down2(x2))

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

        gate_logits = self.gate_head(y)
        learned_gate = torch.sigmoid(gate_logits)
        confidence_mean = confidence.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        uncertainty_gate = self.min_gate + (1.0 - self.min_gate) * (1.0 - confidence_mean)
        refinement_gate = learned_gate * uncertainty_gate

        candidate_dolp = torch.clamp(prior[:, 0:1] + delta_dolp, 0.0, 1.0)
        theta_prior = 0.5 * torch.atan2(prior[:, 2:3], prior[:, 1:2])
        theta_candidate = theta_prior + delta_angle
        candidate_cos = torch.cos(2.0 * theta_candidate)
        candidate_sin = torch.sin(2.0 * theta_candidate)

        dolp = (1.0 - refinement_gate) * prior[:, 0:1] + refinement_gate * candidate_dolp
        cos2 = (1.0 - refinement_gate) * prior[:, 1:2] + refinement_gate * candidate_cos
        sin2 = (1.0 - refinement_gate) * prior[:, 2:3] + refinement_gate * candidate_sin
        norm = torch.sqrt(cos2.square() + sin2.square() + self.eps)
        refined = torch.cat([dolp.clamp(0.0, 1.0), cos2 / norm, sin2 / norm], dim=1)

        return {
            "refined": refined.contiguous(),
            "candidate": torch.cat([candidate_dolp, candidate_cos, candidate_sin], dim=1).contiguous(),
            "delta_dolp": delta_dolp.contiguous(),
            "delta_angle": delta_angle.contiguous(),
            "refinement_gate": refinement_gate.contiguous(),
            "raw_delta_dolp": raw_delta_dolp.contiguous(),
            "raw_delta_angle": raw_delta_angle.contiguous(),
            "gate_logits": gate_logits.contiguous(),
        }

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
