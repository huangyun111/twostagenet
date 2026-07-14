#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
MANIFEST="${MANIFEST:-/home/hy/Documents/hy_FNdataset/dataset/assembly_manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-/home/hy/Documents/hy_FNdataset}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_direct_nafnet_13168}"
GPU_IDS="${GPU_IDS:-0,5,6}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
RESUME="${RESUME:-}"

cd "${ROOT_DIR}"
mkdir -p "${SAVE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"

TRAIN_CMD=(
  "${PYTHON_BIN}" train_direct_nafnet_baseline.py
  --manifest "${MANIFEST}"
  --dataset_root "${DATASET_ROOT}"
  --save_dir "${SAVE_DIR}"
  --crop_size 480
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --lr 1e-4
  --weight_decay 1e-4
  --num_workers "${NUM_WORKERS}"
  --width 32
  --enc_blk_nums 2 2 4 8
  --middle_blk_num 12
  --dec_blk_nums 2 2 2 2
  --save_freq 10
  --seed 42
  --device cuda
  --data_parallel
)

if [[ -n "${RESUME}" ]]; then
  TRAIN_CMD+=(--resume "${RESUME}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_val_samples "${MAX_VAL_SAMPLES}")
fi

echo "[train] Direct NAFNet: final_new512 13168 pretraining"
echo "[train] GPU_IDS=${GPU_IDS} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}"
echo "[train] save_dir=${SAVE_DIR} resume=${RESUME}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_CMD[@]}"
