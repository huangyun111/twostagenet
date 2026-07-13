# HANDOFF - Version D Training

## Objective

Continue with training and evaluation of the Version D reliability-gated Stage2 network. Do not redesign the network again before the required 20-sample overfit check.

## Repository State

- Local root: `G:\fuxiandaima\twostagenet`
- GitHub: `https://github.com/huangyun111/twostagenet`
- Active local branch: `main`
- Version D commit: `26290364a1f320f731a40e47b7f3513af1f0942b`
- GitHub backup branch: `codex/version-d-reliability-gated`
- Version C rollback tag: `version-c-before-vd` -> `f7d832c8593422508801a8d0adcc3e6888f12470`
- Local pre-sync dirty backup: `backup/pre-main-sync-20260713` -> `e84de4a`

The five older root-level `*_handoff.md` files are pre-existing untracked user artifacts. Preserve them. They are not part of Version D.

## Version D Implementation

Version D keeps the Version C Restormer backbone and changes the guidance/refinement logic:

1. Separate RGB, prior, and confidence stems.
2. Confidence-guided prior feature modulation at all three Restormer scales.
3. Separate `gate_dolp` and `gate_angle` using channel-specific confidence.
4. Circular AoLP update: `theta_final = theta_prior + gate_angle * delta_angle`.
5. Default AoLP residual range changed from `pi` to `pi/2`.
6. Zero-initialized residual/gate heads, so initial output is the normalized Stage1 prior.
7. Normalized residual regularization, reliability-weighted angular edge loss, and no-harm loss.
8. Version D checkpoint/output directories are isolated from Version C.

Primary files:

- `models/stage2_prior_guided_restormer_refiner.py`
- `losses/stage2_prior_guided_restormer_loss.py`
- `train_stage2_prior_guided_restormer.py`
- `train_stage2_prior_guided_restormer_online_stage1.py`
- `infer_stage2_prior_guided_restormer.py`
- `scripts/run_stage2_prior_guided_restormer_13168_train.sh`
- `scripts/run_stage2_prior_guided_restormer_hammer_infer.sh`

Version C Stage2 checkpoints are not architecture-compatible with Version D. Start Stage2 training from scratch. The frozen Stage1 checkpoint and exported priors remain reusable.

## Verification Already Done

- Full local test suite: 11 tests passed.
- Python compile checks passed.
- Bash syntax checks passed.
- Three-step forward/backward smoke passed.
- Non-zero finite gradients confirmed for RGB/prior/confidence stems, multi-scale guidance, both residual heads, and both gate heads.
- No full dataset training has been started.

Local test environment: `G:\ana\envs\ips_cuda12\python.exe`.

## Experimental Discipline

- Use train/val only for model and checkpoint decisions.
- Do not repeatedly inspect the frozen final_new512 or HAMMER test sets while developing Version D.
- Keep `official512` and `final_new512` results separate.
- Run the 20-sample train overfit check before formal training.
- Compare original Stage1, low-frequency prior, Version D, Direct Restormer, and Direct U-Net++ in the final ablation.
- If GPU utilization is low and memory has real headroom, probe batch size upward after a safe smoke run.

## Server Defaults

Preferred formal-training server: dl114.

- Python: `/home/hy/miniconda3/envs/PA/bin/python`
- Repo: `/home/hy/twostagenet`
- Dataset manifest: `/home/hy/Documents/hy_FNdataset/dataset/assembly_manifest.json`
- Dataset root: `/home/hy/Documents/hy_FNdataset`
- Expected low-frequency prior directory: `/home/hy/twostagenet/stage1_exports_13168_lowfreq_x4`
- Formal Version D checkpoint directory: `/home/hy/twostagenet/checkpoints_stage2_reliability_gated_restormer_vd_13168`
- Formal log: `/home/hy/twostagenet/stage2_reliability_gated_restormer_vd_13168_train.log`

Before pulling on dl114, inspect `git status`. Preserve remote edits if the repo is dirty. Then update to commit `2629036` and verify the low-frequency prior directory exists.

## Exact Next Step

1. Connect to dl114 and inspect repo/data/GPU state.
2. Update the server worktree to GitHub `main@2629036` without discarding server-only changes.
3. Run a single-GPU 20-sample overfit experiment using train samples only.
4. Confirm loss decreases strongly, outputs depart from the identity prior, both gates remain finite/non-collapsed, and refined train metrics beat the coarse prior.
5. Only after that passes, launch formal train/val training with:

```bash
cd /home/hy/twostagenet
GPU_IDS=0,1,2,3,4,5,6,7 \
BATCH_SIZE=7 \
NUM_WORKERS=8 \
NUM_EPOCHS=80 \
bash scripts/run_stage2_prior_guided_restormer_13168_train.sh
```

Start with available GPUs only. Run a safe batch probe and raise `BATCH_SIZE` when utilization/memory permit. Never overwrite Version C directories.

## Success Gates Before Test

- Validation metrics must beat the low-frequency prior.
- Version D should beat or clearly challenge Direct U-Net++ on vector/AoLP, not merely repair an artificially weakened prior.
- Stage2 must not show the previous cross-domain pattern where it worsens Stage1.
- Select `best_val.pth` using validation only.
- Test inference happens once after architecture and checkpoint selection are frozen.

## Running HAMMER Retrain (2026-07-13)

- Deployment root: `/home/hy/twostage` on `dl_server_114` (`hy@100.76.211.27`). The dirty historical `/home/hy/twostagenet` worktree was preserved.
- Online Stage1 Version D training now supports `nn.DataParallel` for both the frozen Stage1 and trainable Stage2 models.
- HAMMER split sizes verified: train `5253`, val `540`, test `1414`.
- Direct U-Net++-aligned settings: crop `512`, `image_max`, `80` epochs, AdamW, LR `1e-4`, weight decay `1e-4`, workers `4`, seed `42`.
- GPU policy: GPUs `0` and `6` belong to other jobs and must not be touched; training uses `1,2,3,4,5`; GPU `7` stays free.
- Stable formal batch: total batch `10` (2 samples/GPU), about `21.4-21.7 GiB` per training GPU.
- Required 20-sample overfit gate passed. Train loss fell `1.825054 -> 1.200195`; all diagnostics were finite. On the same 20 train samples, coarse-to-refined metrics changed: DoLP MAE `0.07277660 -> 0.07124497`, vector L1 `0.73827407 -> 0.51965080`, weighted AoLP `30.28504620 -> 19.29228144` degrees. Mean gates were DoLP `0.37254880`, angle `0.56047592`.
- Formal tmux session: `vd_hammer_formal`.
- Formal checkpoint dir: `/home/hy/twostage/checkpoints_stage2_reliability_gated_restormer_vd_hammer`.
- Combined train/test log: `/home/hy/twostage/stage2_reliability_gated_restormer_vd_hammer_train_test.log`.
- Launcher: `scripts/run_stage2_prior_guided_restormer_hammer_train_test.sh`. It trains on HAMMER, selects `best_val.pth` using validation only, builds the frozen Stage1 low-frequency test priors, then runs the one-time HAMMER test automatically.

The running-training instructions below were completed and are superseded by the formal-test section that follows.

## HAMMER Formal Test Complete (2026-07-13)

- Training was early-stopped after epoch 57 because epoch 40 remained best for 17 consecutive completed epochs.
- Formal checkpoint: epoch 40 `best_val.pth`, validation loss `2.071533`.
- Formal HAMMER test completed once on all 1414 frozen test frames, sharded over the then-free GPUs `1,2,3,4,5`. GPUs `0,6,7` belonged to other jobs and were not touched.
- Merged Version D metrics: DoLP MAE `0.07729487`, RMSE `0.12429723`, vector `0.86777019`, weighted AoLP `24.565683°`, high-DoLP AoLP `23.165395°`.
- Same-protocol Direct U-Net++ comparison: `0.05849019`, `0.09426134`, `0.87368246`, `24.951095°`, `23.610723°`.
- Interpretation: Version D is slightly better on vector/AoLP but clearly worse on DoLP. Do not claim blanket superiority.
- Local result package: `results_hammer_vd/`.
- Remote formal output: `/home/hy/twostage/stage2_reliability_gated_restormer_vd_hammer_test_outputs`.
- Result figure: `results_hammer_vd/hammer_vd_formal_results.png`.
