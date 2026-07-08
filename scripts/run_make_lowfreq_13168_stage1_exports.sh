#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hy/twostagenet}"
PYTHON_BIN="${PYTHON_BIN:-/home/hy/miniconda3/envs/PA/bin/python}"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/stage1_exports_13168}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/stage1_exports_13168_lowfreq_x4}"
FACTOR="${FACTOR:-4}"
CONFIDENCE_SCALE="${CONFIDENCE_SCALE:-0.5}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/make_lowfreq_stage1_exports.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --factor "${FACTOR}" \
  --confidence_scale "${CONFIDENCE_SCALE}"
