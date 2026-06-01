"""Losses for Stage 1 coarse polarization prior training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1PriorLoss(nn.Module):
    """Composite loss for RGB -> coarse polarization prior supervision."""

    def __init__(
        self,
        lambda_dolp: float = 1.0,
        lambda_aolp: float = 1.0,
        lambda_unc: float | None = None,
        lambda_lowfreq: float = 0.2,
        lambda_edge: float = 0.05,
        tau: float = 0.1,
        lowfreq_kernel_size: int = 8,
        dolp_low: float = 0.03,
        dolp_high: float = 0.15,
        conf_alpha_dolp: float = 10.0,
        conf_alpha_aolp: float = 5.0,
        lambda_conf: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if lowfreq_kernel_size <= 0:
            raise ValueError("lowfreq_kernel_size must be positive.")
        if dolp_high <= dolp_low:
            raise ValueError("dolp_high must be greater than dolp_low.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.lambda_dolp = lambda_dolp
        self.lambda_aolp = lambda_aolp
        self.lambda_conf = lambda_unc if lambda_conf is None else lambda_conf
        if self.lambda_conf is None:
            self.lambda_conf = 0.1
        self.lambda_unc = self.lambda_conf
        self.lambda_lowfreq = lambda_lowfreq
        self.lambda_edge = lambda_edge
        # Kept for compatibility with old configs; AoLP now uses DoLP reliability.
        self.tau = tau
        self.lowfreq_kernel_size = lowfreq_kernel_size
        self.dolp_low = dolp_low
        self.dolp_high = dolp_high
        self.conf_alpha_dolp = conf_alpha_dolp
        self.conf_alpha_aolp = conf_alpha_aolp
        self.eps = eps

    def forward(
        self,
        pred_dict: dict[str, torch.Tensor],
        target_polar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Stage 1 prior losses.

        Args:
            pred_dict: Output dict from PolarPriorNet.forward.
            target_polar: [B, 3, H, W] tensor ordered as
                [DoLP_gt, cos2AoLP_gt, sin2AoLP_gt].
        """
        polar_prior = pred_dict["polar_prior"]
        confidence = pred_dict["confidence"]

        dolp_gt = target_polar[:, 0:1]
        aolp_reliability = self._aolp_reliability(dolp_gt)

        loss_dolp = F.smooth_l1_loss(
            polar_prior[:, 0:1],
            dolp_gt,
        )

        # Suppress AoLP supervision where low DoLP makes the angle noise-dominated.
        aolp_error = F.smooth_l1_loss(
            polar_prior[:, 1:3],
            target_polar[:, 1:3],
            reduction="none",
        )
        loss_aolp = self._weighted_mean(aolp_error, aolp_reliability)

        loss_conf = self._confidence_loss(polar_prior, target_polar, confidence, aolp_reliability)

        loss_lowfreq = self._low_frequency_loss(polar_prior, target_polar)
        loss_edge = self._edge_loss(polar_prior, target_polar)

        total_loss = (
            self.lambda_dolp * loss_dolp
            + self.lambda_aolp * loss_aolp
            + self.lambda_conf * loss_conf
            + self.lambda_lowfreq * loss_lowfreq
            + self.lambda_edge * loss_edge
        )

        return {
            "loss": total_loss,
            "loss_dolp": loss_dolp.detach(),
            "loss_aolp": loss_aolp.detach(),
            "loss_conf": loss_conf.detach(),
            "loss_unc": loss_conf.detach(),
            "loss_lowfreq": loss_lowfreq.detach(),
            "loss_edge": loss_edge.detach(),
            "mean_aolp_reliability": aolp_reliability.mean().detach(),
            "mean_conf_dolp": confidence[:, 0:1].mean().detach(),
            "mean_conf_aolp": confidence[:, 1:3].mean().detach(),
        }

    def _aolp_reliability(self, dolp: torch.Tensor) -> torch.Tensor:
        reliability = (dolp - self.dolp_low) / (self.dolp_high - self.dolp_low)
        return torch.clamp(reliability, 0.0, 1.0)

    def _weighted_mean(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        expanded_weight = weight.expand_as(value)
        numerator = (value * expanded_weight).sum()
        denominator = expanded_weight.sum().clamp_min(self.eps)
        return numerator / denominator

    def _confidence_loss(
        self,
        polar_prior: torch.Tensor,
        target_polar: torch.Tensor,
        confidence: torch.Tensor,
        aolp_reliability: torch.Tensor,
    ) -> torch.Tensor:
        dolp_pred = polar_prior[:, 0:1].detach()
        dolp_gt = target_polar[:, 0:1]
        err_dolp = torch.abs(dolp_pred - dolp_gt)
        conf_dolp_target = torch.exp(-self.conf_alpha_dolp * err_dolp)
        conf_dolp_target = torch.clamp(conf_dolp_target, 0.0, 1.0)

        cos_sin_pred = polar_prior[:, 1:3].detach()
        cos_sin_gt = target_polar[:, 1:3]
        err_aolp = torch.sqrt(
            (cos_sin_pred[:, 0:1] - cos_sin_gt[:, 0:1]).square()
            + (cos_sin_pred[:, 1:2] - cos_sin_gt[:, 1:2]).square()
            + self.eps
        )
        conf_aolp_error = torch.exp(-self.conf_alpha_aolp * err_aolp)
        conf_aolp_target = aolp_reliability * conf_aolp_error
        conf_aolp_target = torch.clamp(conf_aolp_target, 0.0, 1.0)

        conf_target = torch.cat(
            [conf_dolp_target, conf_aolp_target, conf_aolp_target],
            dim=1,
        )
        confidence = torch.clamp(confidence, self.eps, 1.0 - self.eps)
        return F.binary_cross_entropy(confidence, conf_target)

    def _low_frequency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compare low-frequency structure with avg pooling."""
        height, width = pred.shape[-2:]
        kernel_size = min(self.lowfreq_kernel_size, height, width)
        if kernel_size <= 1:
            return F.l1_loss(pred, target)

        pred_low = F.avg_pool2d(pred, kernel_size=kernel_size, stride=kernel_size)
        target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=kernel_size)
        return F.l1_loss(pred_low, target_low)

    @staticmethod
    def _edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match finite-difference x/y gradients without extra dependencies."""
        loss = pred.new_zeros(())

        if pred.shape[-1] > 1:
            pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
            loss = loss + F.l1_loss(pred_grad_x, target_grad_x)

        if pred.shape[-2] > 1:
            pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
            loss = loss + F.l1_loss(pred_grad_y, target_grad_y)

        return loss
