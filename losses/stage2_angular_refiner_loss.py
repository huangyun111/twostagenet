"""Losses for the Stage 2 angular residual refiner."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class Stage2AngularRefinerLoss(nn.Module):
    def __init__(
        self,
        lambda_dolp_final: float = 0.5,
        lambda_vector_final: float = 1.0,
        lambda_aolp_final: float = 2.0,
        lambda_delta_dolp: float = 0.3,
        lambda_delta_angle: float = 1.0,
        lambda_lowfreq: float = 0.1,
        lambda_edge: float = 0.2,
        lambda_residual_reg: float = 0.05,
        lambda_gate_reg: float = 0.02,
        dolp_low: float = 0.03,
        dolp_high: float = 0.15,
        lowfreq_kernel_size: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_dolp_final = lambda_dolp_final
        self.lambda_vector_final = lambda_vector_final
        self.lambda_aolp_final = lambda_aolp_final
        self.lambda_delta_dolp = lambda_delta_dolp
        self.lambda_delta_angle = lambda_delta_angle
        self.lambda_lowfreq = lambda_lowfreq
        self.lambda_edge = lambda_edge
        self.lambda_residual_reg = lambda_residual_reg
        self.lambda_gate_reg = lambda_gate_reg
        self.dolp_low = dolp_low
        self.dolp_high = dolp_high
        self.lowfreq_kernel_size = lowfreq_kernel_size
        self.eps = eps

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        refined = pred["refined"]
        reliability = self._aolp_reliability(target[:, 0:1])

        loss_dolp_final = F.smooth_l1_loss(refined[:, 0:1], target[:, 0:1])

        vector_final = F.smooth_l1_loss(refined[:, 1:3], target[:, 1:3], reduction="none")
        loss_vector_final = self._weighted_mean(vector_final, reliability)

        dot = refined[:, 1:2] * target[:, 1:2] + refined[:, 2:3] * target[:, 2:3]
        cross = refined[:, 2:3] * target[:, 1:2] - refined[:, 1:2] * target[:, 2:3]
        two_aolp_error = torch.atan2(cross, dot.clamp(-1.0, 1.0)).abs()
        loss_aolp_final = self._weighted_mean(two_aolp_error, reliability)

        delta_dolp_gt = target[:, 0:1] - prior[:, 0:1]
        loss_delta_dolp = F.smooth_l1_loss(pred["delta_dolp"], delta_dolp_gt)

        delta_angle_gt = self._target_delta_angle(prior, target)
        angle_diff = self._wrap_angle(pred["delta_angle"] - delta_angle_gt)
        loss_delta_angle = self._weighted_mean(angle_diff.abs(), reliability)

        loss_lowfreq = self._lowfreq_loss(refined, target)
        loss_edge = self._edge_loss(refined, target)

        conf_dolp = confidence[:, 0:1]
        conf_angle = confidence[:, 1:3].mean(dim=1, keepdim=True)
        loss_residual_dolp_reg = torch.mean(pred["delta_dolp"].abs() * (0.2 + conf_dolp))
        loss_residual_angle_reg = torch.mean(pred["delta_angle"].abs() * (0.2 + conf_angle))
        loss_residual_reg = loss_residual_dolp_reg + loss_residual_angle_reg
        loss_gate_reg = torch.mean(pred["gate_dolp"] * conf_dolp) + torch.mean(
            pred["gate_angle"] * conf_angle
        )

        total = (
            self.lambda_dolp_final * loss_dolp_final
            + self.lambda_vector_final * loss_vector_final
            + self.lambda_aolp_final * loss_aolp_final
            + self.lambda_delta_dolp * loss_delta_dolp
            + self.lambda_delta_angle * loss_delta_angle
            + self.lambda_lowfreq * loss_lowfreq
            + self.lambda_edge * loss_edge
            + self.lambda_residual_reg * loss_residual_reg
            + self.lambda_gate_reg * loss_gate_reg
        )

        return {
            "loss": total,
            "loss_dolp_final": loss_dolp_final.detach(),
            "loss_vector_final": loss_vector_final.detach(),
            "loss_aolp_final": loss_aolp_final.detach(),
            "loss_delta_dolp": loss_delta_dolp.detach(),
            "loss_delta_angle": loss_delta_angle.detach(),
            "loss_lowfreq": loss_lowfreq.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_residual_reg": loss_residual_reg.detach(),
            "loss_gate_reg": loss_gate_reg.detach(),
            "mean_reliability": reliability.mean().detach(),
            "mean_gate_dolp": pred["gate_dolp"].mean().detach(),
            "mean_gate_angle": pred["gate_angle"].mean().detach(),
            "mean_abs_delta_dolp": pred["delta_dolp"].abs().mean().detach(),
            "mean_abs_delta_angle_deg": (
                pred["delta_angle"].abs().mean() * (180.0 / math.pi)
            ).detach(),
        }

    def _aolp_reliability(self, dolp: torch.Tensor) -> torch.Tensor:
        denominator = max(self.dolp_high - self.dolp_low, self.eps)
        return torch.clamp((dolp - self.dolp_low) / denominator, 0.0, 1.0)

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.shape[1] == 1 and values.shape[1] != 1:
            weights = weights.expand(-1, values.shape[1], -1, -1)
        numerator = (values * weights).sum()
        denominator = weights.sum().clamp_min(self.eps)
        return numerator / denominator

    @staticmethod
    def _target_delta_angle(prior: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos_p = prior[:, 1:2]
        sin_p = prior[:, 2:3]
        cos_t = target[:, 1:2]
        sin_t = target[:, 2:3]
        return torch.atan2(
            sin_t * cos_p - cos_t * sin_p,
            cos_t * cos_p + sin_t * sin_p,
        )

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _lowfreq_loss(self, refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel_size = min(self.lowfreq_kernel_size, refined.shape[-2], refined.shape[-1])
        if kernel_size <= 1:
            return F.l1_loss(refined, target)
        refined_low = F.avg_pool2d(refined, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        if refined_low.shape[-2:] != refined.shape[-2:]:
            refined_low = refined_low[..., : refined.shape[-2], : refined.shape[-1]]
            target_low = target_low[..., : target.shape[-2], : target.shape[-1]]
        return F.l1_loss(refined_low, target_low)

    @staticmethod
    def _edge_loss(refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        refined_dx = refined[..., :, 1:] - refined[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        refined_dy = refined[..., 1:, :] - refined[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        return F.l1_loss(refined_dx, target_dx) + F.l1_loss(refined_dy, target_dy)
