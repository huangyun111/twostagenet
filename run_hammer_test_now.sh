#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/twostagenet_stage2_pr
PY=/root/miniconda3/envs/PA/bin/python
DATA=/autodl-fs/data/HAMMER_PAstyle/test
WORK=/autodl-fs/data/twostagenet_runs/hammer_test_work
FLAT="${WORK}/flat"
STAGE1_FULL="${WORK}/stage1_exports"
STAGE1_LOW="${WORK}/stage1_exports_lowfreq_x4"
CKPT_STAGE1=/autodl-fs/data/twostagenet_runs/checkpoints_stage1_prior_official_512/best.pth
CKPT_STAGE2=/autodl-fs/data/twostagenet_runs/checkpoints_stage2_prior_guided_restormer_hammer_vc_online_lowfreq_x4/best_val.pth
OUT=/autodl-fs/data/twostagenet_runs/hammer_test_stage2_outputs

# free root disk from failed /tmp run
rm -rf /tmp/hammer_test_flat /tmp/hammer_test_stage1_exports /tmp/hammer_test_stage1_exports_lowfreq_x4

echo "[1/4] flattening test scenes into ${FLAT}"
rm -rf "${FLAT}"
mkdir -p "${FLAT}/S0" "${FLAT}/Polarization_Encoding"
for scene_dir in "${DATA}"/*/; do
  scene=$(basename "${scene_dir}")
  for f in "${scene_dir}/S0"/*.png; do
    [ -e "$f" ] || continue
    ln -s "$f" "${FLAT}/S0/${scene}_$(basename "$f")"
  done
  for f in "${scene_dir}/Polarization_Encoding"/*.png; do
    [ -e "$f" ] || continue
    ln -s "$f" "${FLAT}/Polarization_Encoding/${scene}_$(basename "$f")"
  done
done
echo "flattened samples: $(ls "${FLAT}/S0" | wc -l)"

cd "${PROJECT}"

echo "[2/4] exporting Stage1 priors (full res)"
"${PY}" scripts/export_stage1_prior.py \
  --root_dir "${FLAT}" \
  --checkpoint "${CKPT_STAGE1}" \
  --output_dir "${STAGE1_FULL}" \
  --preprocess_mode official_infer \
  --normalize_mode image_max \
  --output_size_mode native \
  --batch_size 1 \
  --num_workers 4 \
  --device cuda

echo "[3/4] making low-frequency Stage1 exports (factor 4)"
"${PY}" scripts/make_lowfreq_stage1_exports.py \
  --input_dir "${STAGE1_FULL}" \
  --output_dir "${STAGE1_LOW}" \
  --factor 4 \
  --confidence_scale 0.5

echo "[4/4] Stage2 inference"
rm -rf "${OUT}"
"${PY}" infer_stage2_prior_guided_restormer.py \
  --root_dir "${FLAT}" \
  --stage1_dir "${STAGE1_LOW}" \
  --checkpoint "${CKPT_STAGE2}" \
  --output_dir "${OUT}" \
  --preprocess_mode official_infer \
  --normalize_mode image_max \
  --resize_output_to_gt \
  --batch_size 1 \
  --num_workers 4 \
  --device cuda \
  --dim 32 \
  --num_blocks 2,2,4 \
  --num_heads 1,2,4 \
  --ffn_expansion 2.66 \
  --residual_scale 0.5 \
  --angle_residual_scale 3.141592653589793 \
  --min_gate 0.05 \
  --paper_vis_every 0

echo "[done] outputs in ${OUT}"
cat "${OUT}/summary.json"

