#!/usr/bin/env bash
set -euo pipefail

# Local Fast Ortho-LoRA run script.
#
# Uses the pre-tokenized bit/rest dataset generated at:
#   /dataset/ortho_lora/bit_rest.jsonl
#
# Each JSONL record contains:
#   problem_id, task ("bit" or "rest"), category, tokens, mask

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

MODEL_PATH="${MODEL_PATH:-/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848}"
TOKENIZED_JSONL="${TOKENIZED_JSONL:-/dataset/ortho_lora/bit_rest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/Experiment_Output}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
SAVE_STEPS="${SAVE_STEPS:-100}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found under MODEL_PATH=${MODEL_PATH}" >&2
  echo "Set MODEL_PATH to the local Hugging Face snapshot directory." >&2
  exit 1
fi

if [[ ! -f "${TOKENIZED_JSONL}" ]]; then
  echo "Tokenized dataset not found: ${TOKENIZED_JSONL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/ortholora_ver0.py" \
  --model_path "${MODEL_PATH}" \
  --tokenized_jsonl "${TOKENIZED_JSONL}" \
  --task_field task \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_len 8192 \
  --num_steps 1000 \
  --task_batch_size 16 \
  --micro_batch_size 4 \
  --active_tasks_per_step 2 \
  --task_sampling round_robin \
  --shuffle_dataset false \
  --lora_r 8 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --add_lm_head_lora false \
  --learning_rate 2e-4 \
  --dtype bf16 \
  --attn_implementation eager \
  --gradient_checkpointing true \
  --use_cce true \
  --moe_tie_weights true \
  --zip_submission true \
  --logging_steps 1 \
  --save_steps 50

echo "Done."
echo "Final adapter: ${OUTPUT_DIR}/final_adapter"
echo "Submission zip: ${OUTPUT_DIR}/final_adapter/submission.zip"
