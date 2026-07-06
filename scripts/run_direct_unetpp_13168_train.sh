#!/usr/bin/env bash
set -euo pipefail

cd /home/hy/twostagenet

PY=/home/hy/miniconda3/envs/PA/bin/python
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-60}"
SAVE_DIR="${SAVE_DIR:-/home/hy/twostagenet/checkpoints_direct_unetpp_13168}"

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" train_direct_unetpp_baseline.py \
  --manifest /home/hy/Documents/hy_FNdataset/dataset/assembly_manifest.json \
  --dataset_root /home/hy/Documents/hy_FNdataset \
  --save_dir "$SAVE_DIR" \
  --preprocess_mode official_train \
  --crop_size 480 \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$EPOCHS" \
  --num_workers 8 \
  --lr 1e-4 \
  --encoder_lr 1e-5 \
  --encoder_name resnet34 \
  --encoder_weights none \
  --seed 42 \
  --device cuda \
  --save_freq 10
