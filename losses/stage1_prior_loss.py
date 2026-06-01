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
        lambda_unc: float = 0.1,
        lambda_lowfreq: float = 0.2,
        lambda_edge: float = 0.05,
        tau: float = 0.1,
        lowfreq_kernel_size: int = 8,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if lowfreq_kernel_size <= 0:
            raise ValueError("lowfreq_kernel_size must be positive.")

        self.lambda_dolp = lambda_dolp
        self.lambda_aolp = lambda_aolp
        self.lambda_unc = lambda_unc
        self.lambda_lowfreq = lambda_lowfreq
        self.lambda_edge = lambda_edge
        self.tau = tau
        self.lowfreq_kernel_size = lowfreq_kernel_size

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
        log_var = pred_dict["log_var"]

        loss_dolp = F.smooth_l1_loss(
            polar_prior[:, 0:1],
            target_polar[:, 0:1],
        )

        # AoLP is unreliable when DoLP is near zero, so weight cos/sin by DoLP_gt.
        aolp_error = F.smooth_l1_loss(
            polar_prior[:, 1:3],
            target_polar[:, 1:3],
            reduction="none",
        )
        aolp_weight = torch.clamp(target_polar[:, 0:1] / self.tau, 0.0, 1.0)
        loss_aolp = (aolp_error * aolp_weight).mean()

        # Clamp log variance before exponentiation to avoid numerical explosions.
        per_pixel_error = F.smooth_l1_loss(
            polar_prior,
            target_polar,
            reduction="none",
        )
        log_var_for_loss = torch.clamp(log_var, -10.0, 10.0)
        loss_unc = (
            torch.exp(-log_var_for_loss) * per_pixel_error + log_var_for_loss
        ).mean()

        loss_lowfreq = self._low_frequency_loss(polar_prior, target_polar)
        loss_edge = self._edge_loss(polar_prior, target_polar)

        total_loss = (
            self.lambda_dolp * loss_dolp
            + self.lambda_aolp * loss_aolp
            + self.lambda_unc * loss_unc
            + self.lambda_lowfreq * loss_lowfreq
            + self.lambda_edge * loss_edge
        )

        return {
            "loss": total_loss,
            "loss_dolp": loss_dolp.detach(),
            "loss_aolp": loss_aolp.detach(),
            "loss_unc": loss_unc.detach(),
            "loss_lowfreq": loss_lowfreq.detach(),
            "loss_edge": loss_edge.detach(),
        }

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
