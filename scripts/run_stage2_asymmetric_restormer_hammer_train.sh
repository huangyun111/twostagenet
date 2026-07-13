#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/hy/Documents/HAMMER_twostage/train_split}"
VAL_ROOT="${VAL_ROOT:-/home/hy/Documents/HAMMER_twostage/val_split}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/home/hy/twostagenet/checkpoints_stage1_13168/best_val.pth}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_stage2_asymmetric_restormer_ve_hammer}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/stage2_asymmetric_restormer_ve_hammer.launch.log}"
GPU_IDS="${GPU_IDS:-1,2,3,4,5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOLP_RESIDUAL_SCALE="${DOLP_RESIDUAL_SCALE:-0.25}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
RESUME="${RESUME:-}"

cd "${ROOT_DIR}"
mkdir -p "${SAVE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"

TRAIN_CMD=(
  "${PYTHON_BIN}" train_stage2_asymmetric_restormer_online_stage1.py
  --root_dir "${TRAIN_ROOT}"
  --val_root_dir "${VAL_ROOT}"
  --stage1_checkpoint "${STAGE1_CHECKPOINT}"
  --save_dir "${SAVE_DIR}"
  --preprocess_mode official_train
  --crop_size 512
  --normalize_mode image_max
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --lr 1e-4
  --weight_decay 1e-4
  --num_workers "${NUM_WORKERS}"
  --dim 32
  --num_blocks 2,2,4
  --num_heads 1,2,4
  --ffn_expansion 2.66
  --dolp_residual_scale "${DOLP_RESIDUAL_SCALE}"
  --angle_residual_scale 1.5707963267948966
  --min_angle_gate 0.05
  --weak_factor 4
  --confidence_scale 0.5
  --save_freq 20
  --seed 42
  --device cuda
  --data_parallel
)

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_val_samples "${MAX_VAL_SAMPLES}")
fi
if [[ -n "${RESUME}" ]]; then
  TRAIN_CMD+=(--resume "${RESUME}")
fi

echo "[train] architecture=E GPU_IDS=${GPU_IDS} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}" | tee "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
