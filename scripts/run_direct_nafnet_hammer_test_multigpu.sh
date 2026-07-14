#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
HAMMER_ROOT="${HAMMER_ROOT:-/home/hy/Documents/HAMMER_twostage/test}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/checkpoints_direct_nafnet_hammer_e2e/best_val.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/direct_nafnet_hammer_e2e_test_outputs}"
GPU_IDS="${GPU_IDS:-0 5 6}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1414}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "${ROOT_DIR}"
test -f "${CHECKPOINT}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[error] output directory already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}/logs"

GPU_IDS_CLEAN="${GPU_IDS//,/ }"
read -r -a GPUS <<< "${GPU_IDS_CLEAN}"
NUM_SHARDS="${#GPUS[@]}"
echo "[launch] Direct NAFNet HAMMER test: ${NUM_SHARDS} shards on ${GPU_IDS}"
echo "[launch] validation-selected checkpoint=${CHECKPOINT}"

PIDS=()
for SHARD_INDEX in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$SHARD_INDEX]}"
  SHARD_DIR="${OUTPUT_DIR}/shard_${SHARD_INDEX}"
  LOG_FILE="${OUTPUT_DIR}/logs/shard_${SHARD_INDEX}.log"
  EXTRA_ARGS=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
  fi
  (
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" infer_direct_nafnet_hammer.py \
      --root_dir "${HAMMER_ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${SHARD_DIR}" \
      --preprocess_mode official_infer \
      --normalize_mode image_max \
      --batch_size 1 \
      --num_workers "${NUM_WORKERS}" \
      --device cuda \
      --resize_output_to_gt \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${SHARD_INDEX}" \
      "${EXTRA_ARGS[@]}"
  ) > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  echo "[error] at least one inference shard failed" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/merge_direct_unetpp_hammer_shards.py \
  --output_dir "${OUTPUT_DIR}" \
  --expected_count "${EXPECTED_COUNT}"
echo "[done] Direct NAFNet HAMMER end-to-end test complete"
