#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostagenet}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
MANIFEST="${MANIFEST:-/home/hy/Documents/hy_FNdataset/dataset/assembly_manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-/home/hy/Documents/hy_FNdataset}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/checkpoints_direct_unetpp_13168/best_val.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/direct_unetpp_13168_test_outputs}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ENCODER_NAME="${ENCODER_NAME:-resnet34}"

cd "${ROOT_DIR}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

GPU_IDS_CLEAN="${GPU_IDS//,/ }"
read -r -a GPUS <<< "${GPU_IDS_CLEAN}"
NUM_SHARDS="${#GPUS[@]}"
echo "[launch] Direct U-Net++ test inference with ${NUM_SHARDS} shards: ${GPU_IDS}"
echo "[launch] output: ${OUTPUT_DIR}"

for SHARD_INDEX in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$SHARD_INDEX]}"
  SHARD_DIR="${OUTPUT_DIR}/shard_${SHARD_INDEX}"
  LOG_FILE="${OUTPUT_DIR}/logs/shard_${SHARD_INDEX}.log"
  (
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" infer_direct_unetpp_baseline.py \
      --manifest "${MANIFEST}" \
      --dataset_root "${DATASET_ROOT}" \
      --split test \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${SHARD_DIR}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --device cuda \
      --encoder_name "${ENCODER_NAME}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${SHARD_INDEX}"
  ) > "${LOG_FILE}" 2>&1 &
done

wait
echo "[merge] shard processes finished"

"${PYTHON_BIN}" scripts/merge_direct_unetpp_shards.py \
  --output_dir "${OUTPUT_DIR}" \
  --expected_count 1474

echo "[done] Direct U-Net++ test inference complete"
