#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

MODEL_PATH="${MODEL_PATH:-/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848}"
ADAPTER_PATH="${ADAPTER_PATH:-/home/ubuntu/Experiment_Output/final_adapter}"
EVAL_DATASET="${EVAL_DATASET:-/dataset/ortho_lora/bit_rest.jsonl}"
EVAL_TOKEN_SAMPLE="${EVAL_TOKEN_SAMPLE:-${SCRIPT_DIR}/eval_20_per_category.jsonl}"
EVAL_TEXT_SAMPLE="${EVAL_TEXT_SAMPLE:-${SCRIPT_DIR}/eval_20_per_category_text.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${SCRIPT_DIR}/results}"
PER_CATEGORY="${PER_CATEGORY:-20}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
BACKEND="${BACKEND:-auto}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/auto_evaluator.py" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --dataset-jsonl "${EVAL_DATASET}" \
  --token-sample-jsonl "${EVAL_TOKEN_SAMPLE}" \
  --text-sample-jsonl "${EVAL_TEXT_SAMPLE}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --per-category "${PER_CATEGORY}" \
  --backend "${BACKEND}" \
  --load-in-4bit "${LOAD_IN_4BIT}" \
  "$@"
