import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.stage2_prior_guided_restormer_loss import (  # noqa: E402
    Stage2PriorGuidedRestormerLoss,
)
from models.stage2_prior_guided_restormer_refiner import (  # noqa: E402
    Stage2PriorGuidedRestormerRefiner,
)


def make_prior(batch: int, height: int, width: int) -> torch.Tensor:
    dolp = torch.full((batch, 1, height, width), 0.35)
    cos2 = torch.ones(batch, 1, height, width)
    sin2 = torch.zeros(batch, 1, height, width)
    return torch.cat([dolp, cos2, sin2], dim=1)


class Stage2PriorGuidedRestormerRefinerTest(unittest.TestCase):
    def test_output_has_physical_ranges_and_version_c_terms(self) -> None:
        model = Stage2PriorGuidedRestormerRefiner(
            dim=8,
            num_blocks=(1, 1, 1),
            num_heads=(1, 2, 4),
            residual_scale=0.5,
            angle_residual_scale=math.pi,
        )
        rgb = torch.randn(2, 3, 31, 35)
        prior = make_prior(2, 31, 35)
        confidence = torch.full((2, 3, 31, 35), 0.6)

        pred = model(rgb, prior, confidence)

        self.assertEqual(tuple(pred["refined"].shape), (2, 3, 31, 35))
        self.assertEqual(tuple(pred["delta_dolp"].shape), (2, 1, 31, 35))
        self.assertEqual(tuple(pred["delta_angle"].shape), (2, 1, 31, 35))
        self.assertEqual(tuple(pred["refinement_gate"].shape), (2, 1, 31, 35))
        self.assertTrue(bool(torch.all(pred["refined"][:, 0] >= 0.0)))
        self.assertTrue(bool(torch.all(pred["refined"][:, 0] <= 1.0)))
        norms = torch.sqrt(pred["refined"][:, 1].square() + pred["refined"][:, 2].square())
        self.assertTrue(bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)))
        self.assertLessEqual(float(pred["delta_angle"].abs().max()), math.pi + 1e-5)
        self.assertTrue(bool(torch.all(pred["refinement_gate"] >= 0.0)))
        self.assertTrue(bool(torch.all(pred["refinement_gate"] <= 1.0)))

    def test_loss_backpropagates_and_reports_regularizers(self) -> None:
        model = Stage2PriorGuidedRestormerRefiner(
            dim=8,
            num_blocks=(1, 1, 1),
            num_heads=(1, 2, 4),
        )
        loss_fn = Stage2PriorGuidedRestormerLoss()
        rgb = torch.randn(1, 3, 32, 32)
        prior = make_prior(1, 32, 32)
        confidence = torch.full((1, 3, 32, 32), 0.5)
        target = torch.cat(
            [
                torch.full((1, 1, 32, 32), 0.42),
                torch.zeros(1, 1, 32, 32),
                torch.ones(1, 1, 32, 32),
            ],
            dim=1,
        )

        losses = loss_fn(model(rgb, prior, confidence), target, prior, confidence)
        losses["loss"].backward()

        self.assertEqual(
            set(losses),
            {
                "loss",
                "loss_dolp",
                "loss_vector",
                "loss_aolp",
                "loss_high_dolp_aolp",
                "loss_edge",
                "loss_residual_reg",
                "loss_gate_reg",
                "mean_refinement_gate",
                "mean_abs_delta_dolp",
                "mean_abs_delta_angle_deg",
            },
        )
        grad_norm = sum(
            float(param.grad.abs().sum())
            for param in model.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
