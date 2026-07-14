#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
HAMMER_ROOT="${HAMMER_ROOT:-/home/hy/Documents/HAMMER_twostage/test}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/stage1_exports_13168_hammer_lowfreq_x4}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/checkpoints_stage2_asymmetric_restormer_ve_hammer/best_val.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/stage2_asymmetric_restormer_ve_hammer_test_outputs}"
GPU_IDS="${GPU_IDS:-5 6 7}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1414}"
PAPER_VIS_EVERY="${PAPER_VIS_EVERY:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "${ROOT_DIR}"
test -f "${CHECKPOINT}"
test -d "${STAGE1_DIR}/prior_npy"
test -d "${STAGE1_DIR}/confidence_npy"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[error] output directory already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}/logs"

GPU_IDS_CLEAN="${GPU_IDS//,/ }"
read -r -a GPUS <<< "${GPU_IDS_CLEAN}"
NUM_SHARDS="${#GPUS[@]}"
echo "[launch] Version E HAMMER test with ${NUM_SHARDS} shards: ${GPU_IDS}"
echo "[launch] checkpoint=${CHECKPOINT}"
echo "[launch] output=${OUTPUT_DIR}"

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
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" infer_stage2_prior_guided_restormer.py \
      --root_dir "${HAMMER_ROOT}" \
      --stage1_dir "${STAGE1_DIR}" \
      --checkpoint "${CHECKPOINT}" \
      --architecture_version E \
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
      --dolp_residual_scale 0.25 \
      --angle_residual_scale 1.5707963267948966 \
      --min_angle_gate 0.05 \
      --paper_vis_every "${PAPER_VIS_EVERY}" \
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
  echo "[error] one or more inference shards failed" >&2
  exit 1
fi

if [[ -z "${MAX_SAMPLES}" ]]; then
  "${PYTHON_BIN}" scripts/merge_stage2_prior_guided_restormer_shards.py \
    --output_dir "${OUTPUT_DIR}" \
    --expected_count "${EXPECTED_COUNT}"
fi

echo "[done] Version E HAMMER multi-GPU test complete"
