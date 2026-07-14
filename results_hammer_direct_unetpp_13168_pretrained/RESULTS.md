# HAMMER Direct U-Net++ 13168-Pretrained Formal Test

- Pretraining: final_new512 13168 train/val, selected by its validation set
- Fine-tuning: HAMMER train 5253 / val 540
- Fine-tuning stopped after epoch 75 at the user's request
- Selected checkpoint: HAMMER validation-only `best_val.pth`, epoch 70, validation loss 1.006718
- Formal test: frozen HAMMER 1414 frames, one-time evaluation
- Inference GPUs: the then-free GPUs 0, 5, and 6; three shards of 472/471/471
- Merge validation: exactly 1414 rows and 1414 unique sample names
- Output resizing: requested for protocol parity, but resized count was zero

| Method | DoLP MAE ↓ | RMSE ↓ | vector ↓ | weighted AoLP ↓ | high-DoLP AoLP ↓ |
|---|---:|---:|---:|---:|---:|
| Version E, epoch 72 | **0.0526** | **0.0861** | 0.8643 | **24.31°** | 23.10° |
| Direct U-Net++ scratch, epoch 76 | 0.0585 | 0.0943 | 0.8737 | 24.95° | 23.61° |
| Direct U-Net++ 13168 pretrained, epoch 70 | 0.0577 | 0.0924 | **0.8618** | 24.33° | **22.86°** |

The 13168 pretraining improves the Direct U-Net++ scratch baseline on all five metrics. With total pretraining information better aligned, neither model wins every metric: Version E remains lower on DoLP MAE, RMSE, and weighted AoLP, while the pretrained Direct U-Net++ is lower on vector error and high-DoLP AoLP.

This comparison aligns total data exposure more closely but is still a system comparison: Version E freezes a 13168-trained Stage1 and trains only Stage2 on HAMMER, whereas Direct U-Net++ fine-tunes its entire network on HAMMER.

Artifacts:

- `summary.json`: merged formal-test summary
- `metrics.csv`: 1414 per-frame metric rows
- `config.json` and `train.log`: fine-tuning configuration and log through epoch 75
- `logs/`: completed three-shard inference logs
- `hammer_pretrained_direct_comparison.png`: rendered comparison table
- copied Version D, Version E, and scratch Direct U-Net++ summaries for provenance

Remote output:

`/home/hy/twostage/direct_unetpp_13168_pretrained_hammer_test_outputs`
