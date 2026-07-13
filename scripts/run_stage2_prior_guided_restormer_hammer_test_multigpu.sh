#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
HAMMER_ROOT="${HAMMER_ROOT:-/home/hy/Documents/HAMMER_twostage/test}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/stage1_exports_13168_hammer_lowfreq_x4}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/checkpoints_stage2_reliability_gated_restormer_vd_hammer/best_val.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/stage2_reliability_gated_restormer_vd_hammer_test_outputs}"
GPU_IDS="${GPU_IDS:-1 2 3 4 5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1414}"
PAPER_VIS_EVERY="${PAPER_VIS_EVERY:-50}"

cd "${ROOT_DIR}"
test -f "${CHECKPOINT}"
test -d "${STAGE1_DIR}/prior_npy"
test -d "${STAGE1_DIR}/confidence_npy"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

GPU_IDS_CLEAN="${GPU_IDS//,/ }"
read -r -a GPUS <<< "${GPU_IDS_CLEAN}"
NUM_SHARDS="${#GPUS[@]}"
echo "[launch] Version D HAMMER test with ${NUM_SHARDS} shards: ${GPU_IDS}"
echo "[launch] checkpoint=${CHECKPOINT}"
echo "[launch] output=${OUTPUT_DIR}"

PIDS=()
for SHARD_INDEX in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$SHARD_INDEX]}"
  SHARD_DIR="${OUTPUT_DIR}/shard_${SHARD_INDEX}"
  LOG_FILE="${OUTPUT_DIR}/logs/shard_${SHARD_INDEX}.log"
  (
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" infer_stage2_prior_guided_restormer.py \
      --root_dir "${HAMMER_ROOT}" \
      --stage1_dir "${STAGE1_DIR}" \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${SHARD_DIR}" \
      --preprocess_mode official_infer \
      --normalize_mode image_max \
      --batch_size 1 \
      --num_workers "${NUM_WORKERS}" \
      --device cuda \
      --dim 32 \
      --num_blocks 2,2,4 \
      --num_heads 1,2,4 \
      --ffn_expansion 2.66 \
      --residual_scale 0.5 \
      --angle_residual_scale 1.5707963267948966 \
      --min_gate 0.05 \
      --paper_vis_every "${PAPER_VIS_EVERY}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${SHARD_INDEX}"
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
  echo "[error] one or more inference shards failed" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/merge_stage2_prior_guided_restormer_shards.py \
  --output_dir "${OUTPUT_DIR}" \
  --expected_count "${EXPECTED_COUNT}"

echo "[done] Version D HAMMER multi-GPU test complete"
