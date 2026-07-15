#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostage}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/hy/miniconda3/envs/PA/bin/torchrun}"
MANIFEST="${MANIFEST:-/home/hy/Documents/hy_FNdataset/dataset/assembly_manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-/home/hy/Documents/hy_FNdataset}"
HAMMER_ROOT="${HAMMER_ROOT:-/home/hy/Documents/HAMMER_twostage/test}"
GPU_IDS="${GPU_IDS:-0,5,6}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"

STAGE1_SAVE_DIR="${STAGE1_SAVE_DIR:-${ROOT_DIR}/checkpoints_stage1_ve_final_new512_retrain}"
STAGE2_SAVE_DIR="${STAGE2_SAVE_DIR:-${ROOT_DIR}/checkpoints_stage2_asymmetric_restormer_ve_final_new512_retrain}"
FINAL_TEST_DIR="${FINAL_TEST_DIR:-${ROOT_DIR}/version_e_final_new512_retrain_test}"
HAMMER_TEST_DIR="${HAMMER_TEST_DIR:-${ROOT_DIR}/version_e_final_new512_to_hammer_zeroshot}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-60}"
STAGE1_BATCH_PER_GPU="${STAGE1_BATCH_PER_GPU:-6}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-80}"
STAGE2_BATCH="${STAGE2_BATCH:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "${ROOT_DIR}"
mkdir -p "${STAGE1_SAVE_DIR}" "${STAGE2_SAVE_DIR}" "${FINAL_TEST_DIR}" "${HAMMER_TEST_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"

echo "[protocol] final_new512 train/val Stage1 scratch -> frozen Stage1 -> Version E scratch"
echo "[protocol] val-selected checkpoints -> final_new512 test=1474 -> HAMMER zero-shot=1414"
echo "[resources] GPU_IDS=${GPU_IDS} num_gpus=${NUM_GPUS}"

if [[ ! -f "${STAGE1_SAVE_DIR}/COMPLETE" ]]; then
  STAGE1_RESUME=()
  if [[ -f "${STAGE1_SAVE_DIR}/last.pth" ]]; then
    STAGE1_RESUME=(--resume "${STAGE1_SAVE_DIR}/last.pth")
    echo "[stage1] resuming ${STAGE1_SAVE_DIR}/last.pth"
  else
    echo "[stage1] starting from scratch"
  fi
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NUM_GPUS}" \
    train_stage1_prior.py \
    --manifest "${MANIFEST}" \
    --dataset_root "${DATASET_ROOT}" \
    --split train \
    --val_split val \
    --preprocess_mode official_train \
    --crop_size 480 \
    --batch_size "${STAGE1_BATCH_PER_GPU}" \
    --num_epochs "${STAGE1_EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --lr 1e-4 \
    --encoder_lr 1e-5 \
    --seed 42 \
    --save_dir "${STAGE1_SAVE_DIR}" \
    --save_freq 10 \
    --vis_freq 10 \
    "${STAGE1_RESUME[@]}"
  test -f "${STAGE1_SAVE_DIR}/best_val.pth"
  touch "${STAGE1_SAVE_DIR}/COMPLETE"
else
  echo "[stage1] COMPLETE marker found; keeping the completed run"
fi

if [[ ! -f "${STAGE2_SAVE_DIR}/COMPLETE" ]]; then
  STAGE2_RESUME=()
  if [[ -f "${STAGE2_SAVE_DIR}/last.pth" ]]; then
    STAGE2_RESUME=(--resume "${STAGE2_SAVE_DIR}/last.pth")
    echo "[stage2] resuming ${STAGE2_SAVE_DIR}/last.pth"
  else
    echo "[stage2] starting Version E from scratch"
  fi
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" \
    train_stage2_asymmetric_restormer_manifest.py \
    --manifest "${MANIFEST}" \
    --dataset_root "${DATASET_ROOT}" \
    --split train \
    --val_split val \
    --stage1_checkpoint "${STAGE1_SAVE_DIR}/best_val.pth" \
    --save_dir "${STAGE2_SAVE_DIR}" \
    --batch_size "${STAGE2_BATCH}" \
    --num_epochs "${STAGE2_EPOCHS}" \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers "${NUM_WORKERS}" \
    --dim 32 \
    --num_blocks 2,2,4 \
    --num_heads 1,2,4 \
    --ffn_expansion 2.66 \
    --dolp_residual_scale 0.25 \
    --angle_residual_scale 1.5707963267948966 \
    --min_angle_gate 0.05 \
    --weak_factor 4 \
    --confidence_scale 0.5 \
    --save_freq 20 \
    --seed 42 \
    --device cuda \
    --data_parallel \
    "${STAGE2_RESUME[@]}"
  test -f "${STAGE2_SAVE_DIR}/best_val.pth"
  touch "${STAGE2_SAVE_DIR}/COMPLETE"
else
  echo "[stage2] COMPLETE marker found; keeping the completed run"
fi

run_sharded_eval() {
  local label="$1"
  local output_dir="$2"
  local expected_count="$3"
  shift 3
  local -a dataset_args=("$@")
  local -a pids=()
  local shard_index gpu shard_dir
  echo "[test:${label}] starting ${NUM_GPUS} shards, expected=${expected_count}"
  for shard_index in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$shard_index]}"
    shard_dir="${output_dir}/shard_$(printf '%02d' "${shard_index}")"
    mkdir -p "${shard_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" infer_version_e_online_stage1.py \
      "${dataset_args[@]}" \
      --stage1_checkpoint "${STAGE1_SAVE_DIR}/best_val.pth" \
      --stage2_checkpoint "${STAGE2_SAVE_DIR}/best_val.pth" \
      --output_dir "${shard_dir}" \
      --batch_size 2 \
      --num_workers 4 \
      --device cuda \
      --weak_factor 4 \
      --confidence_scale 0.5 \
      --paper_vis_every 50 \
      --num_shards "${NUM_GPUS}" \
      --shard_index "${shard_index}" \
      > "${shard_dir}/run.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[test:${label}] one or more shards failed" >&2
    return 1
  fi
  "${PYTHON_BIN}" scripts/merge_version_e_online_stage1_shards.py \
    --output_dir "${output_dir}" \
    --expected_count "${expected_count}" \
    | tee "${output_dir}/merge.log"
  echo "[test:${label}] complete"
}

if [[ ! -f "${FINAL_TEST_DIR}/COMPLETE" ]]; then
  run_sharded_eval \
    final_new512 \
    "${FINAL_TEST_DIR}" \
    1474 \
    --manifest "${MANIFEST}" \
    --dataset_root "${DATASET_ROOT}" \
    --split test
  touch "${FINAL_TEST_DIR}/COMPLETE"
else
  echo "[test:final_new512] COMPLETE marker found"
fi

if [[ ! -f "${HAMMER_TEST_DIR}/COMPLETE" ]]; then
  run_sharded_eval \
    hammer_zeroshot \
    "${HAMMER_TEST_DIR}" \
    1414 \
    --root_dir "${HAMMER_ROOT}" \
    --preprocess_mode official_infer \
    --normalize_mode image_max \
    --divisible_by 32
  touch "${HAMMER_TEST_DIR}/COMPLETE"
else
  echo "[test:hammer_zeroshot] COMPLETE marker found"
fi

echo "[pipeline] all stages complete"
