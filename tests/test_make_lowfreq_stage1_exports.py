import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_lowfreq_stage1_exports import process_export_pair


class MakeLowfreqStage1ExportsTest(unittest.TestCase):
    def test_process_export_pair_preserves_polar_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            src_prior = root / "src" / "prior_npy"
            src_conf = root / "src" / "confidence_npy"
            dst_prior = root / "dst" / "prior_npy"
            dst_conf = root / "dst" / "confidence_npy"
            src_prior.mkdir(parents=True)
            src_conf.mkdir(parents=True)

            prior = np.zeros((3, 17, 19), dtype=np.float32)
            prior[0] = 0.4
            prior[1] = 1.0
            confidence = np.full((3, 17, 19), 0.8, dtype=np.float32)
            np.save(src_prior / "sample.npy", prior)
            np.save(src_conf / "sample.npy", confidence)

            process_export_pair(
                prior_path=src_prior / "sample.npy",
                confidence_path=src_conf / "sample.npy",
                output_prior_dir=dst_prior,
                output_confidence_dir=dst_conf,
                factor=4,
                confidence_scale=0.5,
            )

            coarse_prior = np.load(dst_prior / "sample.npy")
            coarse_conf = np.load(dst_conf / "sample.npy")
            self.assertEqual(coarse_prior.shape, (3, 17, 19))
            self.assertEqual(coarse_conf.shape, (3, 17, 19))
            self.assertGreaterEqual(float(coarse_prior[0].min()), 0.0)
            self.assertLessEqual(float(coarse_prior[0].max()), 1.0)
            norms = np.sqrt(coarse_prior[1] ** 2 + coarse_prior[2] ** 2)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-4))
            self.assertTrue(np.allclose(coarse_conf, 0.4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
