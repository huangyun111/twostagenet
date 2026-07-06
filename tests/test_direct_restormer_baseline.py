import unittest
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.direct_polar_loss import DirectPolarLoss
from models.direct_restormer_baseline import DirectRestormerBaseline


class DirectRestormerBaselineTest(unittest.TestCase):
    def test_output_has_physical_ranges(self) -> None:
        model = DirectRestormerBaseline(
            dim=8,
            num_blocks=(1, 1, 1, 1),
            num_heads=(1, 1, 2, 4),
        )
        rgb = torch.randn(2, 3, 33, 35)

        pred = model(rgb)

        self.assertEqual(tuple(pred.shape), (2, 3, 33, 35))
        self.assertTrue(bool(torch.all(pred[:, 0] >= 0.0)))
        self.assertTrue(bool(torch.all(pred[:, 0] <= 1.0)))
        norms = torch.sqrt(pred[:, 1].square() + pred[:, 2].square())
        self.assertTrue(bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)))

    def test_loss_backpropagates(self) -> None:
        model = DirectRestormerBaseline(
            dim=8,
            num_blocks=(1, 1, 1, 1),
            num_heads=(1, 1, 2, 4),
        )
        loss_fn = DirectPolarLoss()
        rgb = torch.randn(1, 3, 32, 32)
        target = torch.cat(
            [
                torch.full((1, 1, 32, 32), 0.25),
                torch.zeros(1, 1, 32, 32),
                torch.ones(1, 1, 32, 32),
            ],
            dim=1,
        )

        losses = loss_fn(model(rgb), target)
        losses["loss"].backward()

        grad_norm = sum(
            float(param.grad.abs().sum())
            for param in model.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
