#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="/home/ubuntu/evaluator/results/checkpoint-600"
mkdir -p "${OUTPUT_DIR}"
: > "${OUTPUT_DIR}/eval.log"

cd /home/ubuntu
PYTHONUNBUFFERED=1 \
PRINT_FULL_GENERATIONS=true \
WRITE_LIVE_RESULTS=true \
ADAPTER_PATH=/home/ubuntu/Experiment_Output/checkpoint-600 \
EVAL_OUTPUT_DIR="${OUTPUT_DIR}" \
/home/ubuntu/evaluator/run_eval.sh 2>&1 | tee "${OUTPUT_DIR}/eval.log"
