import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.stage2_asymmetric_restormer_loss import Stage2AsymmetricRestormerLoss
from models.stage2_asymmetric_restormer_refiner import Stage2AsymmetricRestormerRefiner


def make_prior(batch: int, height: int, width: int) -> torch.Tensor:
    dolp = torch.rand(batch, 1, height, width)
    angle = torch.rand(batch, 1, height, width) * torch.pi
    return torch.cat([dolp, torch.cos(2.0 * angle), torch.sin(2.0 * angle)], dim=1)


class Stage2AsymmetricRestormerRefinerTest(unittest.TestCase):
    def build_model(self) -> Stage2AsymmetricRestormerRefiner:
        return Stage2AsymmetricRestormerRefiner(
            dim=8,
            num_blocks=(1, 1, 1),
            num_heads=(1, 1, 1),
            expansion=2.0,
        )

    def test_zero_initialization_is_identity_without_dolp_gate(self) -> None:
        model = self.build_model().eval()
        rgb = torch.randn(2, 3, 32, 36)
        prior = make_prior(2, 32, 36)
        confidence = torch.rand(2, 3, 32, 36)
        with torch.no_grad():
            output = model(rgb, prior, confidence)
        self.assertNotIn("gate_dolp", output)
        self.assertTrue(torch.allclose(output["refined"], prior, atol=2e-6, rtol=0.0))
        self.assertTrue(torch.all(output["gate_angle"] >= 0.0))
        self.assertTrue(torch.all(output["gate_angle"] <= 1.0))

    def test_dolp_head_receives_gradient_and_loss_is_finite(self) -> None:
        model = self.build_model().train()
        loss_fn = Stage2AsymmetricRestormerLoss()
        rgb = torch.randn(2, 3, 32, 36)
        prior = make_prior(2, 32, 36)
        target = make_prior(2, 32, 36)
        confidence = torch.rand(2, 3, 32, 36)
        output = model(rgb, prior, confidence)
        losses = loss_fn(output, target, prior, confidence)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        gradient = model.delta_dolp_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertEqual(set(loss_fn.METRIC_KEYS), set(losses))


if __name__ == "__main__":
    unittest.main()
