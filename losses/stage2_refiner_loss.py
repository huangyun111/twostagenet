"""Losses for the Stage 2 confidence-guided residual refiner."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Stage2RefinerLoss(nn.Module):
    def __init__(
        self,
        lambda_dolp: float = 1.0,
        lambda_aolp: float = 1.0,
        lambda_vector: float = 0.5,
        lambda_lowfreq: float = 0.2,
        lambda_edge: float = 0.1,
        lambda_residual_reg: float = 0.05,
        lambda_gate_reg: float = 0.02,
        dolp_low: float = 0.03,
        dolp_high: float = 0.15,
        lowfreq_kernel_size: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_dolp = lambda_dolp
        self.lambda_aolp = lambda_aolp
        self.lambda_vector = lambda_vector
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
        pred_dict: dict[str, torch.Tensor],
        polar_gt: torch.Tensor,
        prior: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        refined = pred_dict["refined"]
        residual = pred_dict["residual"]
        gate = pred_dict["gate"]
        reliability = self._aolp_reliability(polar_gt[:, 0:1])

        loss_dolp = F.smooth_l1_loss(refined[:, 0:1], polar_gt[:, 0:1])

        aolp_error = F.smooth_l1_loss(
            refined[:, 1:3],
            polar_gt[:, 1:3],
            reduction="none",
        )
        loss_aolp = self._weighted_mean(aolp_error, reliability)

        dot = refined[:, 1:2] * polar_gt[:, 1:2] + refined[:, 2:3] * polar_gt[:, 2:3]
        vector_error = 1.0 - dot.clamp(-1.0, 1.0)
        loss_vector = self._weighted_mean(vector_error, reliability)

        loss_lowfreq = self._lowfreq_loss(refined, polar_gt)
        loss_edge = self._edge_loss(refined, polar_gt)
        loss_residual_reg = torch.mean(torch.abs(residual) * (0.2 + confidence))
        loss_gate_reg = torch.mean(gate * confidence)

        total_loss = (
            self.lambda_dolp * loss_dolp
            + self.lambda_aolp * loss_aolp
            + self.lambda_vector * loss_vector
            + self.lambda_lowfreq * loss_lowfreq
            + self.lambda_edge * loss_edge
            + self.lambda_residual_reg * loss_residual_reg
            + self.lambda_gate_reg * loss_gate_reg
        )

        return {
            "loss": total_loss,
            "loss_dolp": loss_dolp.detach(),
            "loss_aolp": loss_aolp.detach(),
            "loss_vector": loss_vector.detach(),
            "loss_lowfreq": loss_lowfreq.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_residual_reg": loss_residual_reg.detach(),
            "loss_gate_reg": loss_gate_reg.detach(),
            "mean_gate": gate.mean().detach(),
            "mean_abs_residual": residual.abs().mean().detach(),
            "mean_reliability": reliability.mean().detach(),
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

    def _lowfreq_loss(self, refined: torch.Tensor, polar_gt: torch.Tensor) -> torch.Tensor:
        kernel_size = min(self.lowfreq_kernel_size, refined.shape[-2], refined.shape[-1])
        if kernel_size <= 1:
            return F.l1_loss(refined, polar_gt)
        refined_low = F.avg_pool2d(refined, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        gt_low = F.avg_pool2d(polar_gt, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        if refined_low.shape[-2:] != refined.shape[-2:]:
            refined_low = refined_low[..., : refined.shape[-2], : refined.shape[-1]]
            gt_low = gt_low[..., : polar_gt.shape[-2], : polar_gt.shape[-1]]
        return F.l1_loss(refined_low, gt_low)

    @staticmethod
    def _edge_loss(refined: torch.Tensor, polar_gt: torch.Tensor) -> torch.Tensor:
        refined_dx = refined[..., :, 1:] - refined[..., :, :-1]
        gt_dx = polar_gt[..., :, 1:] - polar_gt[..., :, :-1]
        refined_dy = refined[..., 1:, :] - refined[..., :-1, :]
        gt_dy = polar_gt[..., 1:, :] - polar_gt[..., :-1, :]
        return F.l1_loss(refined_dx, gt_dx) + F.l1_loss(refined_dy, gt_dy)
