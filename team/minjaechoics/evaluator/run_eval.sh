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
EVAL_DATASET="${EVAL_DATASET:-/home/ubuntu/dataset/merged_sft_dataset/tokens}"
EVAL_TOKEN_SAMPLE="${EVAL_TOKEN_SAMPLE:-${SCRIPT_DIR}/eval_60_per_category.jsonl}"
EVAL_TEXT_SAMPLE="${EVAL_TEXT_SAMPLE:-${SCRIPT_DIR}/eval_60_per_category_text.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${SCRIPT_DIR}/results}"
PER_CATEGORY="${PER_CATEGORY:-60}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
BACKEND="${BACKEND:-vllm}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
MAX_TOKENS="${MAX_TOKENS:-7680}"
TOP_P="${TOP_P:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/auto_evaluator.py" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --dataset-jsonl "${EVAL_DATASET}" \
  --token-sample-jsonl "${EVAL_TOKEN_SAMPLE}" \
  --text-sample-jsonl "${EVAL_TEXT_SAMPLE}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --per-category "${PER_CATEGORY}" \
  --backend "${BACKEND}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --max-tokens "${MAX_TOKENS}" \
  --top-p "${TOP_P}" \
  --temperature "${TEMPERATURE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --load-in-4bit "${LOAD_IN_4BIT}" \
  "$@"
