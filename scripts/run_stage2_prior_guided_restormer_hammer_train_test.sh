#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/hy/Documents/HAMMER_twostage/train_split}"
VAL_ROOT="${VAL_ROOT:-/home/hy/Documents/HAMMER_twostage/val_split}"
TEST_ROOT="${TEST_ROOT:-/home/hy/Documents/HAMMER_twostage/test}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/home/hy/twostagenet/checkpoints_stage1_13168/best_val.pth}"
STAGE1_EXPORT_DIR="${STAGE1_EXPORT_DIR:-/home/hy/twostagenet/stage1_exports_13168_hammer}"
STAGE1_LOWFREQ_DIR="${STAGE1_LOWFREQ_DIR:-${ROOT_DIR}/stage1_exports_13168_hammer_lowfreq_x4}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_stage2_reliability_gated_restormer_vd_hammer}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/stage2_reliability_gated_restormer_vd_hammer_test_outputs}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/stage2_reliability_gated_restormer_vd_hammer_train_test.log}"
GPU_IDS="${GPU_IDS:-1,2,3,4,5}"
TEST_GPU_ID="${TEST_GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"

cd "${ROOT_DIR}"
mkdir -p "${SAVE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"

TRAIN_CMD=(
  "${PYTHON_BIN}" train_stage2_prior_guided_restormer_online_stage1.py
  --root_dir "${TRAIN_ROOT}"
  --val_root_dir "${VAL_ROOT}"
  --stage1_checkpoint "${STAGE1_CHECKPOINT}"
  --save_dir "${SAVE_DIR}"
  --preprocess_mode official_train
  --crop_size 512
  --normalize_mode image_max
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --lr 1e-4
  --weight_decay 1e-4
  --num_workers "${NUM_WORKERS}"
  --dim 32
  --num_blocks 2,2,4
  --num_heads 1,2,4
  --ffn_expansion 2.66
  --residual_scale 0.5
  --angle_residual_scale 1.5707963267948966
  --min_gate 0.05
  --weak_factor 4
  --confidence_scale 0.5
  --save_freq 20
  --seed 42
  --device cuda
  --data_parallel
)

if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  TRAIN_CMD+=(--max_val_samples "${MAX_VAL_SAMPLES}")
fi

echo "[train] GPU_IDS=${GPU_IDS} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}" | tee "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

if [[ ! -d "${STAGE1_LOWFREQ_DIR}/prior_npy" ]]; then
  echo "[prepare-test] building low-frequency Stage1 exports" | tee -a "${LOG_FILE}"
  "${PYTHON_BIN}" scripts/make_lowfreq_stage1_exports.py \
    --input_dir "${STAGE1_EXPORT_DIR}" \
    --output_dir "${STAGE1_LOWFREQ_DIR}" \
    --factor 4 \
    --confidence_scale 0.5 2>&1 | tee -a "${LOG_FILE}"
fi

rm -rf "${OUTPUT_DIR}"
echo "[test] GPU=${TEST_GPU_ID} checkpoint=${SAVE_DIR}/best_val.pth" | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${TEST_GPU_ID}" "${PYTHON_BIN}" infer_stage2_prior_guided_restormer.py \
  --root_dir "${TEST_ROOT}" \
  --stage1_dir "${STAGE1_LOWFREQ_DIR}" \
  --checkpoint "${SAVE_DIR}/best_val.pth" \
  --output_dir "${OUTPUT_DIR}" \
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
  --paper_vis_every 50 2>&1 | tee -a "${LOG_FILE}"

echo "[done] HAMMER Version D train and test complete" | tee -a "${LOG_FILE}"
