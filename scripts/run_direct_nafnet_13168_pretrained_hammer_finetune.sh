#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/hy/Documents/HAMMER_twostage/train_split}"
VAL_ROOT="${VAL_ROOT:-/home/hy/Documents/HAMMER_twostage/val_split}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${ROOT_DIR}/checkpoints_direct_nafnet_13168/best_val.pth}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_direct_nafnet_13168_pretrained_hammer}"
GPU_IDS="${GPU_IDS:-0,5,6}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
RESUME="${RESUME:-}"

cd "${ROOT_DIR}"
test -f "${INIT_CHECKPOINT}"
mkdir -p "${SAVE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"

TRAIN_CMD=(
  "${PYTHON_BIN}" train_direct_nafnet_hammer_finetune.py
  --root_dir "${TRAIN_ROOT}"
  --val_root_dir "${VAL_ROOT}"
  --save_dir "${SAVE_DIR}"
  --preprocess_mode official_train
  --crop_size 512
  --normalize_mode image_max
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --lr 1e-4
  --weight_decay 1e-4
  --num_workers "${NUM_WORKERS}"
  --width 32
  --enc_blk_nums 2 2 4 8
  --middle_blk_num 12
  --dec_blk_nums 2 2 2 2
  --save_freq 20
  --seed 42
  --device cuda
  --data_parallel
)

if [[ -n "${RESUME}" ]]; then
  TRAIN_CMD+=(--resume "${RESUME}")
else
  TRAIN_CMD+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_val_samples "${MAX_VAL_SAMPLES}")
fi

echo "[train] Direct NAFNet: 13168 best_val -> HAMMER train/val fine-tuning"
echo "[train] GPU_IDS=${GPU_IDS} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}"
echo "[train] init=${INIT_CHECKPOINT} save_dir=${SAVE_DIR} resume=${RESUME}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_CMD[@]}"
