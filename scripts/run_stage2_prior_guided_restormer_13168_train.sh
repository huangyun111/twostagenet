#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostagenet}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
MANIFEST="${MANIFEST:-/home/hy/Documents/hy_FNdataset/dataset/assembly_manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-/home/hy/Documents/hy_FNdataset}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/stage1_exports_13168_lowfreq_x4}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_stage2_prior_guided_restormer_13168}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/stage2_prior_guided_restormer_13168_train.log}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,7}"
BATCH_SIZE="${BATCH_SIZE:-7}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
LR="${LR:-1e-4}"
DIM="${DIM:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-2,2,4}"
NUM_HEADS="${NUM_HEADS:-1,2,4}"
FFN_EXPANSION="${FFN_EXPANSION:-2.66}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.5}"
ANGLE_RESIDUAL_SCALE="${ANGLE_RESIDUAL_SCALE:-3.141592653589793}"
MIN_GATE="${MIN_GATE:-0.05}"

cd "${ROOT_DIR}"
mkdir -p "${SAVE_DIR}"
echo "[launch] Stage2 Prior-Guided Restormer train GPU_IDS=${GPU_IDS} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}" | tee "${LOG_FILE}"
echo "[launch] stage1_dir=${STAGE1_DIR}" | tee -a "${LOG_FILE}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" train_stage2_prior_guided_restormer.py \
  --manifest "${MANIFEST}" \
  --dataset_root "${DATASET_ROOT}" \
  --stage1_dir "${STAGE1_DIR}" \
  --split train \
  --val_split val \
  --save_dir "${SAVE_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --dim "${DIM}" \
  --num_blocks "${NUM_BLOCKS}" \
  --num_heads "${NUM_HEADS}" \
  --ffn_expansion "${FFN_EXPANSION}" \
  --residual_scale "${RESIDUAL_SCALE}" \
  --angle_residual_scale "${ANGLE_RESIDUAL_SCALE}" \
  --min_gate "${MIN_GATE}" \
  --save_freq 10 \
  --seed 42 \
  --device cuda \
  --data_parallel 2>&1 | tee -a "${LOG_FILE}"
