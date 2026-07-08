#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostagenet}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
HAMMER_ROOT="${HAMMER_ROOT:-/home/hy/Documents/HAMMER_twostage/test}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/stage1_exports_13168_hammer_lowfreq_x4}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/checkpoints_stage2_prior_guided_restormer_13168/best_val.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/stage2_prior_guided_restormer_hammer_outputs}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DIM="${DIM:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-2,2,4}"
NUM_HEADS="${NUM_HEADS:-1,2,4}"
FFN_EXPANSION="${FFN_EXPANSION:-2.66}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.5}"
ANGLE_RESIDUAL_SCALE="${ANGLE_RESIDUAL_SCALE:-3.141592653589793}"
MIN_GATE="${MIN_GATE:-0.05}"
PAPER_VIS_EVERY="${PAPER_VIS_EVERY:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "${ROOT_DIR}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
echo "[launch] Stage2 Prior-Guided Restormer HAMMER infer GPU=${GPU_ID}"
echo "[launch] checkpoint=${CHECKPOINT}"
echo "[launch] stage1_dir=${STAGE1_DIR}"
echo "[launch] output=${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" infer_stage2_prior_guided_restormer.py
  --root_dir "${HAMMER_ROOT}"
  --stage1_dir "${STAGE1_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --preprocess_mode official_infer
  --normalize_mode image_max
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --device cuda
  --dim "${DIM}"
  --num_blocks "${NUM_BLOCKS}"
  --num_heads "${NUM_HEADS}"
  --ffn_expansion "${FFN_EXPANSION}"
  --residual_scale "${RESIDUAL_SCALE}"
  --angle_residual_scale "${ANGLE_RESIDUAL_SCALE}"
  --min_gate "${MIN_GATE}"
  --paper_vis_every "${PAPER_VIS_EVERY}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD+=(--max_samples "${MAX_SAMPLES}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
echo "[done] Stage2 Prior-Guided Restormer HAMMER inference complete"
