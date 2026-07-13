# HAMMER Version D Formal Test

- Test set: HAMMER, frozen 1414 frames
- Training protocol: HAMMER train/val, supervised
- Checkpoint selection: validation only
- Selected checkpoint: `best_val.pth`, epoch 40, validation loss 2.071533
- Formal inference: five shards on GPUs 1-5; all 1414 unique rows merged successfully
- Output resizing: none

| Method | DoLP MAE ↓ | RMSE ↓ | vector ↓ | weighted AoLP ↓ | high-DoLP AoLP ↓ |
|---|---:|---:|---:|---:|---:|
| Frozen low-frequency Stage1 | 0.0773 | 0.1243 | 1.2678 | 40.01° | 36.67° |
| Version D, epoch 40 | 0.0773 | 0.1243 | 0.8678 | 24.57° | 23.17° |
| Direct U-Net++ | 0.0585 | 0.0943 | 0.8737 | 24.95° | 23.61° |

Version D substantially improves vector and AoLP metrics over its frozen Stage1 prior while leaving DoLP essentially unchanged. Against the same-protocol Direct U-Net++ baseline, Version D is slightly better on vector and AoLP but clearly worse on DoLP MAE/RMSE. The paper-safe conclusion is angle-competitive with a DoLP tradeoff, not a blanket improvement.

Artifacts:

- `summary.json`: merged Version D summary
- `metrics.csv`: 1414 per-frame metric rows
- `direct_unetpp_hammer_summary.json`: same-protocol comparison summary
- `hammer_vd_formal_results.png`: rendered formal result table
- `test_multigpu.log`: five-shard launcher and merge log

Remote formal output:

`/home/hy/twostage/stage2_reliability_gated_restormer_vd_hammer_test_outputs`
