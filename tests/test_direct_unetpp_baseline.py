import unittest
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.direct_polar_loss import DirectPolarLoss
from models.direct_unetpp_baseline import DirectUnetPlusPlusBaseline


class DirectUnetPlusPlusBaselineTest(unittest.TestCase):
    def test_output_has_physical_ranges(self) -> None:
        model = DirectUnetPlusPlusBaseline(encoder_name="resnet18")
        rgb = torch.randn(2, 3, 32, 32)

        pred = model(rgb)

        self.assertEqual(tuple(pred.shape), (2, 3, 32, 32))
        self.assertTrue(bool(torch.all(pred[:, 0] >= 0.0)))
        self.assertTrue(bool(torch.all(pred[:, 0] <= 1.0)))
        norms = torch.sqrt(pred[:, 1].square() + pred[:, 2].square())
        self.assertTrue(bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)))

    def test_loss_returns_trainable_terms(self) -> None:
        loss_fn = DirectPolarLoss()
        raw = torch.zeros(2, 3, 16, 16, requires_grad=True)
        pred = torch.cat(
            [
                raw[:, 0:1].sigmoid(),
                torch.ones_like(raw[:, 1:2]),
                torch.zeros_like(raw[:, 2:3]),
            ],
            dim=1,
        )
        target = torch.cat(
            [
                torch.full((2, 1, 16, 16), 0.5),
                torch.zeros(2, 1, 16, 16),
                torch.ones(2, 1, 16, 16),
            ],
            dim=1,
        )

        losses = loss_fn(pred, target)

        self.assertEqual(
            set(losses),
            {
                "loss",
                "loss_dolp",
                "loss_vector",
                "loss_aolp",
                "loss_lowfreq",
                "loss_edge",
                "mean_aolp_reliability",
            },
        )
        self.assertTrue(bool(losses["loss"].requires_grad))
        self.assertGreater(float(losses["loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
