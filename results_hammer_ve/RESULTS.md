# HAMMER Version E Formal Test

- Test set: HAMMER, frozen 1414 frames
- Training protocol: HAMMER train/val, supervised
- Checkpoint selection: validation only
- Selected checkpoint: `best_val.pth`, epoch 72, validation loss 2.275343
- Formal inference: three shards on the then-free GPUs 5, 6, and 7; all 1414 unique rows merged successfully
- Output resizing: none

| Method | DoLP MAE ↓ | RMSE ↓ | vector ↓ | weighted AoLP ↓ | high-DoLP AoLP ↓ |
|---|---:|---:|---:|---:|---:|
| Frozen low-frequency Stage1 | 0.0773 | 0.1243 | 1.2678 | 40.01° | 36.67° |
| Version D, epoch 40 | 0.0773 | 0.1243 | 0.8678 | 24.57° | 23.17° |
| **Version E, epoch 72** | **0.0526** | **0.0861** | **0.8643** | **24.31°** | **23.10°** |
| Direct U-Net++ | 0.0585 | 0.0943 | 0.8737 | 24.95° | 23.61° |

Under the same formal HAMMER protocol, Version E records lower values than both Version D and Direct U-Net++ on all five reported metrics. Relative to Version D, the main change is the intended DoLP repair: DoLP MAE decreases by 31.89% and RMSE by 30.70%, while the three angle/vector metrics also improve slightly. Relative to Direct U-Net++, Version E decreases DoLP MAE by 9.99%, RMSE by 8.62%, vector error by 1.07%, weighted AoLP by 2.57%, and high-DoLP AoLP by 2.16%.

This was the one-time formal test after validation-only checkpoint selection. Do not tune the architecture or choose another checkpoint using these test results.

Artifacts:

- `summary.json`: merged Version E summary
- `metrics.csv`: 1414 per-frame metric rows
- `direct_unetpp_hammer_summary.json`: same-protocol Direct U-Net++ comparison summary
- `version_d_summary.json`: same-protocol Version D comparison summary
- `hammer_ve_formal_results.png`: rendered formal result table
- `logs/`: completed inference logs for the three GPU shards

Remote formal output:

`/home/hy/twostage/stage2_asymmetric_restormer_ve_hammer_test_outputs`
