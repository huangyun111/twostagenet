#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostagenet}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
MANIFEST="${MANIFEST:-/home/hy/Documents/hy_FNdataset/dataset/assembly_manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-/home/hy/Documents/hy_FNdataset}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_direct_restormer_13168}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/direct_restormer_13168_train.log}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,7}"
BATCH_SIZE="${BATCH_SIZE:-14}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
LR="${LR:-1e-4}"
DIM="${DIM:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-2,2,2,4}"
NUM_HEADS="${NUM_HEADS:-1,2,4,8}"
FFN_EXPANSION="${FFN_EXPANSION:-2.66}"

cd "${ROOT_DIR}"
mkdir -p "${SAVE_DIR}"
echo "[launch] Direct Restormer train GPU_IDS=${GPU_IDS} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}" | tee "${LOG_FILE}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" train_direct_restormer_baseline.py \
  --manifest "${MANIFEST}" \
  --dataset_root "${DATASET_ROOT}" \
  --split train \
  --val_split val \
  --save_dir "${SAVE_DIR}" \
  --crop_size 480 \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --dim "${DIM}" \
  --num_blocks "${NUM_BLOCKS}" \
  --num_heads "${NUM_HEADS}" \
  --ffn_expansion "${FFN_EXPANSION}" \
  --save_freq 10 \
  --device cuda \
  --data_parallel 2>&1 | tee -a "${LOG_FILE}"
