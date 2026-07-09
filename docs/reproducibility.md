# Reproducibility

## Scope

This project is reproducible at the package and verification level. A reader should be able to inspect the dataset package, run static gates, and understand how the adapter training run is launched.

Full model training and evaluation require external assets that are intentionally not stored in the repository:

- NVIDIA Nemotron base model weights.
- CUDA-compatible GPU environment.
- Challenge data and local evaluation setup.
- Dependency wheels required by the model runtime.
- Sufficient disk space for model caches and adapter outputs.

## Required Inputs

| Input | Purpose |
| --- | --- |
| `data/rsp_dataset/rsp_anchor_sft.jsonl` | Anchor completion-only SFT data |
| `data/rsp_dataset/rsp_decision_sft.jsonl` | Rule-trace completion-only SFT data |
| `data/rsp_dataset/rsp_decision_preferences.jsonl` | Chosen/rejected preference data |
| `data/rsp_dataset/rsp_manifest.json` | Dataset manifest and counts |
| Nemotron model path or model ID | Tokenizer and model loading |
| Compatible CUDA/PyTorch stack | GPU training |

If the public repository does not include full JSONL data, publish a small sample plus instructions for regenerating or obtaining the private/full data package. The recommended public convention is to stage the full dataset under `data/rsp_dataset` while keeping that directory out of Git.

## Static Verification

Run dataset verification:

```bash
DATASET_DIR=data/rsp_dataset

python verify_rsp_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --json-output "$DATASET_DIR/rsp_verification.json"
```

Run train-shell verification:

```bash
python verify_rsp_train_shell.py \
  --train-script rsp_train_huikang_compatible.py \
  --dataset-verification "$DATASET_DIR/rsp_verification.json" \
  --json-output "$DATASET_DIR/rsp_train_shell_verification.json"
```

Expected current dataset result:

```json
{
  "rsp_dataset_valid": true,
  "current_minimum_goal_achieved": "no",
  "submission_allowed": false,
  "gpu_execution_allowed": false,
  "errors": 0
}
```

## Training

Dry-run the trainer when the tokenizer/model path is available:

```bash
python rsp_train_huikang_compatible.py \
  --dataset-dir "$DATASET_DIR" \
  --model /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --dry-run
```

Build the external GPU payload:

```bash
python build_rsp_vast_payload.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir outputs/rsp_vast_payload/payload \
  --archive outputs/rsp_vast_payload/rsp_pro6000_payload.tar.gz
```

Launch the prepared GPU wrapper after staging the payload at the wrapper's expected workspace path in a compatible external environment:

```bash
bash /workspace/rsp_vast/payload/rsp_run_train_pro6000.sh
```

The wrapper prepares a virtual environment, resolves the Nemotron model and package wheels through KaggleHub-compatible assets, runs static gates, trains the adapter, and validates the produced adapter zip. It assumes the RSP payload has already been staged under `/workspace/rsp_vast/payload`.

## Adapter Validation

After training:

```bash
python verify_rsp_train_shell.py \
  --train-script rsp_train_huikang_compatible.py \
  --dataset-verification "$DATASET_DIR/rsp_verification.json" \
  --adapter-zip /path/to/submission.zip \
  --json-output /path/to/rsp_post_training_adapter_gate.json
```

Adapter validation should confirm:

- `adapter_config.json` exists.
- `adapter_model.safetensors` exists.
- LoRA rank is 32.
- LoRA alpha is 32.
- LoRA dropout is 0.0.
- Target modules match the locked Nemotron adapter contract.
- LoRA A/B tensors are present and structurally consistent.

## Evaluation

Evaluation is intentionally separate from training. Use the local evaluator only after an adapter artifact passes structure gates.

The evaluation protocol should record:

- Base model path.
- Adapter artifact hash.
- Evaluation script version.
- Dataset or split identifier.
- Per-domain accuracy.
- Boxed-answer extraction behavior.
- Any decoding parameters.

Do not treat a public submission as the validation loop. A submission should be made only after local evidence is complete and promoted.

## Reproducibility Boundaries

Current reproducibility status:

- Dataset package: statically verified.
- Train shell: statically verified.
- GPU training: prepared but not final-score-proven in this folder.
- Adapter selection: not finalized for RSP.
- Leaderboard score: not claimed.

The project should remain transparent about these boundaries in public documentation.
