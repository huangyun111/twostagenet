"""Losses for Stage 2 angular physical residual diffusion."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.stage2_angular_diffusion_net import apply_residual_to_prior


class Stage2AngularDiffusionLoss(nn.Module):
    def __init__(
        self,
        lambda_delta_dolp: float = 0.5,
        lambda_delta_angle: float = 1.0,
        lambda_dolp_final: float = 1.0,
        lambda_vector_final: float = 1.0,
        lambda_aolp_final: float = 2.0,
        lambda_lowfreq: float = 0.1,
        lambda_edge: float = 0.2,
        lambda_residual_reg: float = 0.05,
        lambda_conf_residual: float = 0.05,
        dolp_low: float = 0.03,
        dolp_high: float = 0.15,
        lowfreq_kernel_size: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_delta_dolp = lambda_delta_dolp
        self.lambda_delta_angle = lambda_delta_angle
        self.lambda_dolp_final = lambda_dolp_final
        self.lambda_vector_final = lambda_vector_final
        self.lambda_aolp_final = lambda_aolp_final
        self.lambda_lowfreq = lambda_lowfreq
        self.lambda_edge = lambda_edge
        self.lambda_residual_reg = lambda_residual_reg
        self.lambda_conf_residual = lambda_conf_residual
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
        residual_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_delta_dolp = pred["pred_delta_dolp"]
        pred_delta_angle = pred["pred_delta_angle"]
        refined = apply_residual_to_prior(prior, pred["pred_residual"], self.eps)
        reliability = self._aolp_reliability(target[:, 0:1])

        loss_delta_dolp = F.smooth_l1_loss(pred_delta_dolp, residual_gt[:, 0:1])
        angle_diff = self._wrap_angle(pred_delta_angle - residual_gt[:, 1:2])
        loss_delta_angle = self._weighted_smooth_l1(angle_diff, torch.zeros_like(angle_diff), reliability)

        loss_dolp_final = F.smooth_l1_loss(refined[:, 0:1], target[:, 0:1])
        vector_final = F.smooth_l1_loss(refined[:, 1:3], target[:, 1:3], reduction="none")
        loss_vector_final = self._weighted_mean(vector_final, reliability)

        dot = refined[:, 1:2] * target[:, 1:2] + refined[:, 2:3] * target[:, 2:3]
        cross = refined[:, 2:3] * target[:, 1:2] - refined[:, 1:2] * target[:, 2:3]
        two_aolp_error = torch.atan2(cross, dot.clamp(-1.0, 1.0)).abs()
        loss_aolp_final = self._weighted_mean(two_aolp_error, reliability)

        loss_edge = self._edge_loss(refined, target, reliability)
        loss_lowfreq = self._lowfreq_loss(refined, target, reliability)

        loss_residual_reg = pred_delta_dolp.abs().mean() + self._weighted_mean(
            pred_delta_angle.abs(),
            reliability,
        )
        conf_dolp = confidence[:, 0:1]
        conf_angle = confidence[:, 1:3].mean(dim=1, keepdim=True)
        loss_conf_residual = torch.mean(pred_delta_dolp.abs() * conf_dolp) + torch.mean(
            pred_delta_angle.abs() * conf_angle
        )

        total = (
            self.lambda_delta_dolp * loss_delta_dolp
            + self.lambda_delta_angle * loss_delta_angle
            + self.lambda_dolp_final * loss_dolp_final
            + self.lambda_vector_final * loss_vector_final
            + self.lambda_aolp_final * loss_aolp_final
            + self.lambda_lowfreq * loss_lowfreq
            + self.lambda_edge * loss_edge
            + self.lambda_residual_reg * loss_residual_reg
            + self.lambda_conf_residual * loss_conf_residual
        )

        return {
            "loss": total,
            "loss_delta_dolp": loss_delta_dolp.detach(),
            "loss_delta_angle": loss_delta_angle.detach(),
            "loss_dolp_final": loss_dolp_final.detach(),
            "loss_vector_final": loss_vector_final.detach(),
            "loss_aolp_final": loss_aolp_final.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_lowfreq": loss_lowfreq.detach(),
            "loss_residual_reg": loss_residual_reg.detach(),
            "loss_conf_residual": loss_conf_residual.detach(),
            "mean_abs_delta_dolp": pred_delta_dolp.abs().mean().detach(),
            "mean_abs_delta_angle": pred_delta_angle.abs().mean().detach(),
            "mean_reliability": reliability.mean().detach(),
        }

    def _aolp_reliability(self, dolp: torch.Tensor) -> torch.Tensor:
        denominator = max(self.dolp_high - self.dolp_low, self.eps)
        return torch.clamp((dolp - self.dolp_low) / denominator, 0.0, 1.0)

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.shape[1] == 1 and values.shape[1] != 1:
            weights = weights.expand(-1, values.shape[1], -1, -1)
        return (values * weights).sum() / weights.sum().clamp_min(self.eps)

    def _weighted_smooth_l1(
        self,
        values: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(values, target, reduction="none")
        return self._weighted_mean(loss, weights)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _lowfreq_loss(
        self,
        refined: torch.Tensor,
        target: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        kernel_size = min(self.lowfreq_kernel_size, refined.shape[-2], refined.shape[-1])
        if kernel_size <= 1:
            dolp_loss = F.l1_loss(refined[:, 0:1], target[:, 0:1])
            vector_loss = self._weighted_mean((refined[:, 1:3] - target[:, 1:3]).abs(), reliability)
            return dolp_loss + vector_loss
        refined_low = F.avg_pool2d(refined, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        if refined_low.shape[-2:] != refined.shape[-2:]:
            refined_low = refined_low[..., : refined.shape[-2], : refined.shape[-1]]
            target_low = target_low[..., : target.shape[-2], : target.shape[-1]]
        dolp_loss = F.l1_loss(refined_low[:, 0:1], target_low[:, 0:1])
        vector_loss = self._weighted_mean(
            (refined_low[:, 1:3] - target_low[:, 1:3]).abs(),
            reliability,
        )
        return dolp_loss + vector_loss

    def _edge_loss(
        self,
        refined: torch.Tensor,
        target: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        refined_dx = refined[..., :, 1:] - refined[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        refined_dy = refined[..., 1:, :] - refined[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        rel_dx = reliability[..., :, 1:]
        rel_dy = reliability[..., 1:, :]
        dolp_edge = F.l1_loss(refined_dx[:, 0:1], target_dx[:, 0:1]) + F.l1_loss(
            refined_dy[:, 0:1],
            target_dy[:, 0:1],
        )
        vector_edge = self._weighted_mean((refined_dx[:, 1:3] - target_dx[:, 1:3]).abs(), rel_dx)
        vector_edge = vector_edge + self._weighted_mean(
            (refined_dy[:, 1:3] - target_dy[:, 1:3]).abs(),
            rel_dy,
        )
        return dolp_edge + vector_edge
