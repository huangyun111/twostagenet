import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

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
    def test_multiscale_guidance_respects_prior_reliability(self) -> None:
        image_features = torch.zeros(1, 4, 8, 8)
        prior_features = torch.ones(1, 4, 8, 8)
        confidence_features = torch.randn(1, 4, 8, 8)
        gate = nn.Conv2d(4, 4, kernel_size=1)
        nn.init.zeros_(gate.weight)
        nn.init.zeros_(gate.bias)

        reliable = Stage2PriorGuidedRestormerRefiner._guided_fusion(
            image_features,
            prior_features,
            confidence_features,
            torch.ones(1, 1, 8, 8),
            gate,
        )
        unreliable = Stage2PriorGuidedRestormerRefiner._guided_fusion(
            image_features,
            prior_features,
            confidence_features,
            torch.zeros(1, 1, 8, 8),
            gate,
        )

        self.assertTrue(bool(torch.allclose(reliable, prior_features)))
        self.assertTrue(bool(torch.allclose(unreliable, image_features)))

    def test_output_has_physical_ranges_and_version_d_terms(self) -> None:
        model = Stage2PriorGuidedRestormerRefiner(
            dim=8,
            num_blocks=(1, 1, 1),
            num_heads=(1, 2, 4),
            residual_scale=0.5,
            angle_residual_scale=math.pi / 2.0,
        )
        rgb = torch.randn(2, 3, 31, 35)
        prior = make_prior(2, 31, 35)
        confidence = torch.empty(2, 3, 31, 35)
        confidence[:, 0:1] = 0.9
        confidence[:, 1:3] = 0.1

        pred = model(rgb, prior, confidence)

        self.assertEqual(tuple(pred["refined"].shape), (2, 3, 31, 35))
        self.assertEqual(tuple(pred["delta_dolp"].shape), (2, 1, 31, 35))
        self.assertEqual(tuple(pred["delta_angle"].shape), (2, 1, 31, 35))
        self.assertEqual(tuple(pred["gate_dolp"].shape), (2, 1, 31, 35))
        self.assertEqual(tuple(pred["gate_angle"].shape), (2, 1, 31, 35))
        self.assertEqual(tuple(pred["refinement_gate"].shape), (2, 1, 31, 35))
        self.assertTrue(bool(torch.all(pred["refined"][:, 0] >= 0.0)))
        self.assertTrue(bool(torch.all(pred["refined"][:, 0] <= 1.0)))
        norms = torch.sqrt(pred["refined"][:, 1].square() + pred["refined"][:, 2].square())
        self.assertTrue(bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)))
        self.assertLessEqual(float(pred["delta_angle"].abs().max()), math.pi / 2.0 + 1e-5)
        self.assertTrue(bool(torch.allclose(pred["refined"], prior, atol=1e-5)))
        self.assertTrue(bool(torch.all(pred["gate_dolp"] < pred["gate_angle"])))
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
                "loss_edge_dolp",
                "loss_edge_angle",
                "loss_residual_reg",
                "loss_gate_reg",
                "loss_no_harm",
                "mean_refinement_gate",
                "mean_gate_dolp",
                "mean_gate_angle",
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

    def test_no_harm_loss_penalizes_a_worse_refinement(self) -> None:
        prior = make_prior(1, 8, 8)
        target = prior.clone()
        refined = torch.cat(
            [
                torch.full((1, 1, 8, 8), 0.8),
                -torch.ones(1, 1, 8, 8),
                torch.zeros(1, 1, 8, 8),
            ],
            dim=1,
        )
        pred = {
            "refined": refined,
            "delta_dolp": torch.full((1, 1, 8, 8), 0.45),
            "delta_angle": torch.full((1, 1, 8, 8), math.pi / 2.0),
            "gate_dolp": torch.ones(1, 1, 8, 8),
            "gate_angle": torch.ones(1, 1, 8, 8),
            "refinement_gate": torch.ones(1, 1, 8, 8),
        }
        confidence = torch.ones(1, 3, 8, 8)

        losses = Stage2PriorGuidedRestormerLoss()(pred, target, prior, confidence)

        self.assertGreater(float(losses["loss_no_harm"]), 0.0)


if __name__ == "__main__":
    unittest.main()
