#!/usr/bin/env bash
set -euo pipefail

ROOT="${RSP_ROOT:-/workspace/rsp_vast}"
PAYLOAD="$ROOT/payload"
DATASET="$PAYLOAD/rsp_dataset"
OUTPUT="$ROOT/output"
CACHE="$ROOT/cache"
VENV="$ROOT/venv"
LOG="$ROOT/rsp_train_pro6000.log"
PATHS="$ROOT/resolved_rsp_inputs.json"

mkdir -p "$OUTPUT" "$CACHE"
export KAGGLEHUB_CACHE="$CACHE/kagglehub"
export HF_HOME="$CACHE/huggingface"
export PIP_CACHE_DIR="$CACHE/pip"
exec > >(tee -a "$LOG") 2>&1

if [[ ! -x "$VENV/bin/python3" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
export PATH="$VENV/bin:$PATH"

python3 --version
nvidia-smi

python3 -m pip install --disable-pip-version-check -q kagglehub

python3 - <<'PY' "$PATHS"
import json
from pathlib import Path
import kagglehub
import sys

model_dir = Path(kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")).resolve()
packages = Path(kagglehub.dataset_download("mayukh18/nemotron-packages")).resolve()
package_dirs = sorted(packages.rglob("packages"))
causal = sorted(packages.rglob("causal_conv1d-*.whl"))
mamba = sorted(packages.rglob("mamba_ssm-*.whl"))
if len(package_dirs) != 1 or len(causal) != 1 or len(mamba) != 1:
    raise SystemExit({
        "package_dirs": [str(p) for p in package_dirs],
        "causal": [str(p) for p in causal],
        "mamba": [str(p) for p in mamba],
    })
paths = {
    "model_dir": str(model_dir),
    "package_dir": str(package_dirs[0]),
    "causal_wheel": str(causal[0]),
    "mamba_wheel": str(mamba[0]),
}
Path(sys.argv[1]).write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
print(json.dumps(paths, indent=2))
PY

read_path() {
  python3 - "$PATHS" "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
PY
}

MODEL_DIR="$(read_path model_dir)"
PACKAGE_DIR="$(read_path package_dir)"
CAUSAL_WHEEL="$(read_path causal_wheel)"
MAMBA_WHEEL="$(read_path mamba_wheel)"

python3 -m pip install --disable-pip-version-check --no-index --find-links "$PACKAGE_DIR" "$CAUSAL_WHEEL" "$MAMBA_WHEEL"

cd "$PAYLOAD"
python3 verify_rsp_dataset.py --dataset-dir "$DATASET" --json-output "$OUTPUT/rsp_dataset_verification.json"
python3 verify_rsp_train_shell.py \
  --train-script "$PAYLOAD/rsp_train_huikang_compatible.py" \
  --dataset-verification "$OUTPUT/rsp_dataset_verification.json" \
  --json-output "$OUTPUT/rsp_train_shell_verification.json"

python3 rsp_train_huikang_compatible.py \
  --dataset-dir "$DATASET" \
  --model "$MODEL_DIR" \
  --output-dir "$OUTPUT/rsp_adapter" \
  --submission-zip "$OUTPUT/submission.zip" \
  --audit-json "$OUTPUT/rsp_training_audit.json" \
  --max-seq-length 8192 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --sft-learning-rate 1.6e-4 \
  --preference-learning-rate 3.5e-5 \
  --sft-epochs "${RSP_SFT_EPOCHS:-1.0}" \
  --preference-epochs "${RSP_PREF_EPOCHS:-0.35}" \
  --per-device-train-batch-size "${RSP_BATCH_SIZE:-2}" \
  --gradient-accumulation-steps "${RSP_GRAD_ACCUM:-8}" \
  --preference-batch-size "${RSP_PREF_BATCH_SIZE:-2}" \
  --preference-gradient-accumulation-steps "${RSP_PREF_GRAD_ACCUM:-8}" \
  --warmup-steps 20 \
  --enable-4bit

python3 verify_rsp_train_shell.py \
  --train-script "$PAYLOAD/rsp_train_huikang_compatible.py" \
  --dataset-verification "$OUTPUT/rsp_dataset_verification.json" \
  --adapter-zip "$OUTPUT/submission.zip" \
  --json-output "$OUTPUT/rsp_post_training_adapter_gate.json"

echo "RSP PRO 6000 train complete: $OUTPUT/submission.zip"
