#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
GPU_IDS="${GPU_IDS:-0,5,6}"
BATCH_SIZE="${BATCH_SIZE:-3}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_direct_nafnet_hammer_e2e}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/direct_nafnet_hammer_e2e_test_outputs}"

cd "${ROOT_DIR}"

if [[ -f "${SAVE_DIR}/train.log" ]] && grep -q "Finished training" "${SAVE_DIR}/train.log"; then
  echo "[pipeline] HAMMER training already complete; keeping validation-selected best."
else
  RESUME=""
  if [[ -f "${SAVE_DIR}/last.pth" ]]; then
    RESUME="${SAVE_DIR}/last.pth"
  fi
  ROOT_DIR="${ROOT_DIR}" \
  GPU_IDS="${GPU_IDS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SAVE_DIR="${SAVE_DIR}" \
  RESUME="${RESUME}" \
    bash scripts/run_direct_nafnet_hammer_train.sh
fi

test -f "${SAVE_DIR}/best_val.pth"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  echo "[pipeline] HAMMER test already complete: ${OUTPUT_DIR}/summary.json"
else
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "[error] incomplete test output already exists: ${OUTPUT_DIR}" >&2
    exit 2
  fi
  ROOT_DIR="${ROOT_DIR}" \
  GPU_IDS="${GPU_IDS}" \
  CHECKPOINT="${SAVE_DIR}/best_val.pth" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
    bash scripts/run_direct_nafnet_hammer_test_multigpu.sh
fi
