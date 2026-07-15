# HAMMER Direct NAFNet End-to-End Result

- Protocol: random initialization, HAMMER train/val only, validation-selected
  checkpoint, one frozen HAMMER test evaluation.
- Validation-selected checkpoint: epoch 29, validation loss `1.0085768876`.
- Test set: 1414 frames; the merged CSV contains 1414 unique frame names.
- Model: task-adapted NAFNet-width32 using the official SIDD block layout.
- No final_new512/13168 data or checkpoint was used.

| Metric | Direct NAFNet | Direct U-Net++ scratch |
|---|---:|---:|
| DoLP MAE | 0.05661604 | 0.05849019 |
| DoLP RMSE | 0.09058309 | 0.09426134 |
| vector | 0.88256796 | 0.87368246 |
| weighted AoLP | 25.36656211° | 24.95109514° |
| high-DoLP AoLP | 24.19445091° | 23.61072311° |

Direct NAFNet is lower on the two DoLP metrics, while Direct U-Net++ scratch
is lower on vector and both AoLP metrics. These are final frozen-test results;
do not tune another checkpoint from them.
