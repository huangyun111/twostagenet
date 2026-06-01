"""Coarse polarization prior generator for Stage 1.

This module only defines the RGB -> coarse polarization prior network wrapper.
It intentionally contains no training logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class PolarPriorNet(nn.Module):
    """Generate coarse polarization priors and uncertainty-based confidence.

    Output channel layout from the decoder:
        0: raw_DoLP
        1: raw_cos2AoLP
        2: raw_sin2AoLP
        3: log_var_DoLP
        4: log_var_cos2AoLP
        5: log_var_sin2AoLP
    """

    def __init__(self, encoder_weights: str | None = None, eps: float = 1e-6) -> None:
        """Build the Unet++ backbone.

        Args:
            encoder_weights: Defaults to None to avoid downloading weights.
                Pass "imagenet" explicitly when pretrained encoder weights are desired.
            eps: Numerical stabilizer for unit-circle normalization.
        """
        super().__init__()
        self.eps = eps
        self.net = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=6,
        )

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run RGB input through the prior generator.

        Args:
            rgb: RGB tensor with shape [B, 3, H, W].

        Returns:
            A dict containing:
                polar_prior: [B, 3, H, W] = [DoLP_c, cos2AoLP_c, sin2AoLP_c]
                confidence: [B, 3, H, W], computed as sigmoid(-log_var)
                log_var: [B, 3, H, W]
                raw_output: [B, 6, H, W]
        """
        raw_output = self.net(rgb)

        raw_dolp = raw_output[:, 0:1]
        raw_cos = raw_output[:, 1:2]
        raw_sin = raw_output[:, 2:3]

        # Keep DoLP in [0, 1].
        dolp_c = torch.sigmoid(raw_dolp)

        # Project the raw angle representation onto the unit circle.
        norm = torch.sqrt(raw_cos.square() + raw_sin.square() + self.eps)
        cos2aolp_c = raw_cos / norm
        sin2aolp_c = raw_sin / norm

        polar_prior = torch.cat([dolp_c, cos2aolp_c, sin2aolp_c], dim=1)

        # Larger predicted log variance means lower confidence.
        log_var = raw_output[:, 3:6]
        confidence = torch.sigmoid(-log_var)

        return {
            "polar_prior": polar_prior,
            "confidence": confidence,
            "log_var": log_var,
            "raw_output": raw_output,
        }


def _print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Print compact smoke-test statistics for one output tensor."""
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}"
    )


if __name__ == "__main__":
    model = PolarPriorNet()
    model.eval()

    rgb = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(rgb)

    for key in ("polar_prior", "confidence", "log_var", "raw_output"):
        _print_tensor_stats(key, outputs[key])
