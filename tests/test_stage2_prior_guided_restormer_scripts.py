import argparse
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import infer_stage2_prior_guided_restormer as infer_script  # noqa: E402
import train_stage2_prior_guided_restormer as train_script  # noqa: E402


class Stage2PriorGuidedRestormerScriptsTest(unittest.TestCase):
    def test_train_build_model_uses_version_d_model(self) -> None:
        args = argparse.Namespace(
            dim=8,
            num_blocks=(1, 1, 1),
            num_heads=(1, 2, 4),
            ffn_expansion=2.0,
            residual_scale=0.5,
            angle_residual_scale=1.5707963267948966,
            min_gate=0.05,
        )

        model = train_script.build_model(args, torch.device("cpu"))

        self.assertEqual(model.__class__.__name__, "Stage2PriorGuidedRestormerRefiner")

    def test_infer_summarizes_stage1_and_stage2_metrics(self) -> None:
        rows = [
            {
                "name": "a",
                "stage1_dolp_mae": 0.2,
                "stage2_dolp_mae": 0.1,
                "stage1_weighted_aolp_error_deg": 20.0,
                "stage2_weighted_aolp_error_deg": 10.0,
            },
            {
                "name": "b",
                "stage1_dolp_mae": 0.4,
                "stage2_dolp_mae": 0.3,
                "stage1_weighted_aolp_error_deg": 40.0,
                "stage2_weighted_aolp_error_deg": 30.0,
            },
        ]

        summary = infer_script.summarize_rows(rows)

        self.assertAlmostEqual(summary["stage1_dolp_mae"], 0.3)
        self.assertAlmostEqual(summary["stage2_dolp_mae"], 0.2)
        self.assertAlmostEqual(summary["stage1_weighted_aolp_error_deg"], 30.0)
        self.assertAlmostEqual(summary["stage2_weighted_aolp_error_deg"], 20.0)


if __name__ == "__main__":
    unittest.main()
