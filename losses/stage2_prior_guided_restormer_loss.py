"""Loss for the prior-guided Restormer Stage 2 refiner."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class Stage2PriorGuidedRestormerLoss(nn.Module):
    def __init__(
        self,
        lambda_dolp: float = 1.0,
        lambda_vector: float = 1.0,
        lambda_aolp: float = 2.0,
        lambda_high_dolp_aolp: float = 1.0,
        lambda_edge: float = 0.2,
        lambda_residual_reg: float = 0.02,
        lambda_gate_reg: float = 0.01,
        lambda_no_harm: float = 0.2,
        residual_scale: float = 0.5,
        angle_residual_scale: float = math.pi / 2.0,
        dolp_low: float = 0.03,
        dolp_high: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_dolp = lambda_dolp
        self.lambda_vector = lambda_vector
        self.lambda_aolp = lambda_aolp
        self.lambda_high_dolp_aolp = lambda_high_dolp_aolp
        self.lambda_edge = lambda_edge
        self.lambda_residual_reg = lambda_residual_reg
        self.lambda_gate_reg = lambda_gate_reg
        self.lambda_no_harm = lambda_no_harm
        self.residual_scale = residual_scale
        self.angle_residual_scale = angle_residual_scale
        self.dolp_low = dolp_low
        self.dolp_high = dolp_high
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

        loss_dolp = F.smooth_l1_loss(refined[:, 0:1], target[:, 0:1])
        vector_error = F.smooth_l1_loss(
            refined[:, 1:3],
            target[:, 1:3],
            reduction="none",
        )
        loss_vector = self._weighted_mean(vector_error, reliability)

        aolp_error = self._aolp_error_rad(refined, target)
        loss_aolp = self._weighted_mean(aolp_error, reliability)
        high_mask = (target[:, 0:1] > self.dolp_high).to(target.dtype)
        loss_high_dolp_aolp = self._weighted_mean(aolp_error, high_mask)

        loss_edge_dolp, loss_edge_angle = self._edge_loss(refined, target, reliability)
        loss_edge = loss_edge_dolp + loss_edge_angle
        loss_residual_reg = (
            pred["delta_dolp"].abs().mean() / max(self.residual_scale, self.eps)
            + pred["delta_angle"].abs().mean() / max(self.angle_residual_scale, self.eps)
        )
        confidence_dolp = confidence[:, 0:1].clamp(0.0, 1.0)
        confidence_angle = confidence[:, 1:3].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        gate_dolp = pred["gate_dolp"] if "gate_dolp" in pred else pred["refinement_gate"]
        gate_angle = pred["gate_angle"] if "gate_angle" in pred else pred["refinement_gate"]
        normalized_delta_dolp = pred["delta_dolp"].abs() / max(self.residual_scale, self.eps)
        normalized_delta_angle = pred["delta_angle"].abs() / max(
            self.angle_residual_scale,
            self.eps,
        )
        loss_gate_reg = 0.5 * (
            (gate_dolp * confidence_dolp * normalized_delta_dolp).mean()
            + self._weighted_mean(
                gate_angle * confidence_angle * normalized_delta_angle,
                reliability,
            )
        )

        prior_dolp_error = (prior[:, 0:1] - target[:, 0:1]).abs()
        refined_dolp_error = (refined[:, 0:1] - target[:, 0:1]).abs()
        loss_no_harm_dolp = F.relu(refined_dolp_error - prior_dolp_error).mean()
        prior_aolp_error = self._aolp_error_rad(prior, target)
        loss_no_harm_angle = self._weighted_mean(
            F.relu(aolp_error - prior_aolp_error),
            reliability,
        )
        loss_no_harm = loss_no_harm_dolp + loss_no_harm_angle

        total = (
            self.lambda_dolp * loss_dolp
            + self.lambda_vector * loss_vector
            + self.lambda_aolp * loss_aolp
            + self.lambda_high_dolp_aolp * loss_high_dolp_aolp
            + self.lambda_edge * loss_edge
            + self.lambda_residual_reg * loss_residual_reg
            + self.lambda_gate_reg * loss_gate_reg
            + self.lambda_no_harm * loss_no_harm
        )

        return {
            "loss": total,
            "loss_dolp": loss_dolp.detach(),
            "loss_vector": loss_vector.detach(),
            "loss_aolp": loss_aolp.detach(),
            "loss_high_dolp_aolp": loss_high_dolp_aolp.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_edge_dolp": loss_edge_dolp.detach(),
            "loss_edge_angle": loss_edge_angle.detach(),
            "loss_residual_reg": loss_residual_reg.detach(),
            "loss_gate_reg": loss_gate_reg.detach(),
            "loss_no_harm": loss_no_harm.detach(),
            "mean_refinement_gate": pred["refinement_gate"].mean().detach(),
            "mean_gate_dolp": gate_dolp.mean().detach(),
            "mean_gate_angle": gate_angle.mean().detach(),
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
        weight_sum = weights.sum()
        if float(weight_sum.detach()) <= 0.0:
            return values.sum() * 0.0
        return (values * weights).sum() / weight_sum.clamp_min(self.eps)

    @staticmethod
    def _aolp_error_rad(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot = pred[:, 1:2] * target[:, 1:2] + pred[:, 2:3] * target[:, 2:3]
        cross = pred[:, 2:3] * target[:, 1:2] - pred[:, 1:2] * target[:, 2:3]
        return 0.5 * torch.atan2(cross.abs(), dot).abs()

    def _edge_loss(
        self,
        refined: torch.Tensor,
        target: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined_dx = refined[..., :, 1:] - refined[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        refined_dy = refined[..., 1:, :] - refined[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]

        dolp_edge = F.l1_loss(refined_dx[:, 0:1], target_dx[:, 0:1])
        dolp_edge = dolp_edge + F.l1_loss(refined_dy[:, 0:1], target_dy[:, 0:1])

        reliability_dx = torch.minimum(reliability[..., :, 1:], reliability[..., :, :-1])
        reliability_dy = torch.minimum(reliability[..., 1:, :], reliability[..., :-1, :])
        angle_edge = self._weighted_mean(
            (refined_dx[:, 1:3] - target_dx[:, 1:3]).abs(),
            reliability_dx,
        )
        angle_edge = angle_edge + self._weighted_mean(
            (refined_dy[:, 1:3] - target_dy[:, 1:3]).abs(),
            reliability_dy,
        )
        return dolp_edge, angle_edge
