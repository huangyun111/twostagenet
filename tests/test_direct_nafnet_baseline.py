import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.direct_polar_loss import DirectPolarLoss
from models.direct_nafnet_baseline import DirectNAFNetBaseline
from train_direct_nafnet_hammer_finetune import load_weights_only


class DirectNAFNetBaselineTest(unittest.TestCase):
    def make_small_model(self) -> DirectNAFNetBaseline:
        return DirectNAFNetBaseline(
            width=8,
            enc_blk_nums=(1, 1),
            middle_blk_num=1,
            dec_blk_nums=(1, 1),
        )

    def test_output_has_physical_ranges_and_original_size(self) -> None:
        model = self.make_small_model()
        prediction = model(torch.randn(2, 3, 33, 35))

        self.assertEqual(tuple(prediction.shape), (2, 3, 33, 35))
        self.assertTrue(bool(torch.all(prediction[:, 0] >= 0.0)))
        self.assertTrue(bool(torch.all(prediction[:, 0] <= 1.0)))
        vector_norm = prediction[:, 1].square().add(prediction[:, 2].square()).sqrt()
        self.assertTrue(
            bool(torch.allclose(vector_norm, torch.ones_like(vector_norm), atol=1e-4))
        )

    def test_direct_loss_backpropagates(self) -> None:
        model = self.make_small_model()
        target = torch.cat(
            (
                torch.full((1, 1, 32, 32), 0.25),
                torch.zeros(1, 1, 32, 32),
                torch.ones(1, 1, 32, 32),
            ),
            dim=1,
        )
        losses = DirectPolarLoss()(model(torch.randn(1, 3, 32, 32)), target)
        losses["loss"].backward()

        total_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(total_gradient, 0.0)

    def test_hammer_init_loads_weights_only_and_reports_source(self) -> None:
        source = self.make_small_model()
        target = self.make_small_model()
        with torch.no_grad():
            next(source.parameters()).fill_(0.125)
            next(target.parameters()).zero_()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "best_val.pth"
            torch.save(
                {
                    "epoch": 19,
                    "model": source.state_dict(),
                    "optimizer": {"must_not_be_loaded": True},
                    "best_val_loss": 0.375,
                },
                checkpoint_path,
            )
            metadata = load_weights_only(str(checkpoint_path), target, torch.device("cpu"))

        self.assertTrue(torch.equal(next(source.parameters()), next(target.parameters())))
        self.assertEqual(metadata["source_epoch"], 19)
        self.assertEqual(metadata["source_best_val_loss"], 0.375)

    def test_encoder_decoder_depths_must_match(self) -> None:
        with self.assertRaises(ValueError):
            DirectNAFNetBaseline(enc_blk_nums=(1, 1), dec_blk_nums=(1,))


if __name__ == "__main__":
    unittest.main()
