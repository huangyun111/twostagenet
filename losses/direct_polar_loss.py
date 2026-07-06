"""Loss for direct RGB/S0 -> [DoLP, cos2AoLP, sin2AoLP] baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectPolarLoss(nn.Module):
    """Deterministic polar loss without confidence or residual terms."""

    def __init__(
        self,
        lambda_dolp: float = 1.0,
        lambda_vector: float = 1.0,
        lambda_aolp: float = 0.5,
        lambda_lowfreq: float = 0.2,
        lambda_edge: float = 0.05,
        dolp_low: float = 0.03,
        dolp_high: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dolp_high <= dolp_low:
            raise ValueError("dolp_high must be greater than dolp_low.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.lambda_dolp = lambda_dolp
        self.lambda_vector = lambda_vector
        self.lambda_aolp = lambda_aolp
        self.lambda_lowfreq = lambda_lowfreq
        self.lambda_edge = lambda_edge
        self.dolp_low = dolp_low
        self.dolp_high = dolp_high
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        reliability = self._aolp_reliability(target[:, 0:1])
        loss_dolp = F.smooth_l1_loss(pred[:, 0:1], target[:, 0:1])
        vector_error = F.smooth_l1_loss(pred[:, 1:3], target[:, 1:3], reduction="none")
        loss_vector = self._weighted_mean(vector_error, reliability)
        loss_aolp = self._weighted_mean(self._two_aolp_error(pred, target), reliability)
        loss_lowfreq = self._low_frequency_loss(pred, target)
        loss_edge = self._edge_loss(pred, target)
        total = (
            self.lambda_dolp * loss_dolp
            + self.lambda_vector * loss_vector
            + self.lambda_aolp * loss_aolp
            + self.lambda_lowfreq * loss_lowfreq
            + self.lambda_edge * loss_edge
        )
        return {
            "loss": total,
            "loss_dolp": loss_dolp.detach(),
            "loss_vector": loss_vector.detach(),
            "loss_aolp": loss_aolp.detach(),
            "loss_lowfreq": loss_lowfreq.detach(),
            "loss_edge": loss_edge.detach(),
            "mean_aolp_reliability": reliability.mean().detach(),
        }

    def _aolp_reliability(self, dolp: torch.Tensor) -> torch.Tensor:
        reliability = (dolp - self.dolp_low) / (self.dolp_high - self.dolp_low)
        return torch.clamp(reliability, 0.0, 1.0)

    def _weighted_mean(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        expanded_weight = weight.expand_as(value)
        return (value * expanded_weight).sum() / expanded_weight.sum().clamp_min(self.eps)

    @staticmethod
    def _two_aolp_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross = pred[:, 2:3] * target[:, 1:2] - pred[:, 1:2] * target[:, 2:3]
        dot = pred[:, 1:2] * target[:, 1:2] + pred[:, 2:3] * target[:, 2:3]
        return torch.atan2(cross.abs(), dot).abs()

    @staticmethod
    def _low_frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        height, width = pred.shape[-2:]
        kernel_size = min(8, height, width)
        if kernel_size <= 1:
            return F.l1_loss(pred, target)
        pred_low = F.avg_pool2d(pred, kernel_size=kernel_size, stride=kernel_size)
        target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=kernel_size)
        return F.l1_loss(pred_low, target_low)

    @staticmethod
    def _edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pred.new_zeros(())
        if pred.shape[-1] > 1:
            loss = loss + F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1])
        if pred.shape[-2] > 1:
            loss = loss + F.l1_loss(pred[:, :, 1:, :] - pred[:, :, :-1, :], target[:, :, 1:, :] - target[:, :, :-1, :])
        return loss
