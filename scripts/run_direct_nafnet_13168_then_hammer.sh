#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
GPU_IDS="${GPU_IDS:-0,5,6}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-12}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-12}"
PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_DIR:-${ROOT_DIR}/checkpoints_direct_nafnet_13168}"
FINETUNE_SAVE_DIR="${FINETUNE_SAVE_DIR:-${ROOT_DIR}/checkpoints_direct_nafnet_13168_pretrained_hammer}"

cd "${ROOT_DIR}"

if [[ -f "${PRETRAIN_SAVE_DIR}/train.log" ]] && grep -q "Finished training" "${PRETRAIN_SAVE_DIR}/train.log"; then
  echo "[pipeline] 13168 pretraining already complete; keeping its best_val checkpoint."
else
  PRETRAIN_RESUME=""
  if [[ -f "${PRETRAIN_SAVE_DIR}/last.pth" ]]; then
    PRETRAIN_RESUME="${PRETRAIN_SAVE_DIR}/last.pth"
  fi
  ROOT_DIR="${ROOT_DIR}" \
  GPU_IDS="${GPU_IDS}" \
  BATCH_SIZE="${PRETRAIN_BATCH_SIZE}" \
  SAVE_DIR="${PRETRAIN_SAVE_DIR}" \
  RESUME="${PRETRAIN_RESUME}" \
    bash scripts/run_direct_nafnet_13168_train.sh
fi

test -f "${PRETRAIN_SAVE_DIR}/best_val.pth"

if [[ -f "${FINETUNE_SAVE_DIR}/train.log" ]] && grep -q "Finished training" "${FINETUNE_SAVE_DIR}/train.log"; then
  echo "[pipeline] HAMMER fine-tuning already complete; keeping its best_val checkpoint."
else
  FINETUNE_RESUME=""
  if [[ -f "${FINETUNE_SAVE_DIR}/last.pth" ]]; then
    FINETUNE_RESUME="${FINETUNE_SAVE_DIR}/last.pth"
  fi
  ROOT_DIR="${ROOT_DIR}" \
  GPU_IDS="${GPU_IDS}" \
  BATCH_SIZE="${FINETUNE_BATCH_SIZE}" \
  INIT_CHECKPOINT="${PRETRAIN_SAVE_DIR}/best_val.pth" \
  SAVE_DIR="${FINETUNE_SAVE_DIR}" \
  RESUME="${FINETUNE_RESUME}" \
    bash scripts/run_direct_nafnet_13168_pretrained_hammer_finetune.sh
fi

echo "[pipeline] Training complete. HAMMER model selection is validation-only."
echo "[pipeline] No HAMMER test inference was launched."
