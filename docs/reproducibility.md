# Reproducibility

이 문서는 clean public repo에서 어디까지 재현 가능한지와, 어떤 외부 artifact가 필요한지 설명합니다.

## Scope

현재 repo가 재현 가능하게 제공하는 범위:

- RSP dataset schema
- small example dataset
- dataset static verification
- train shell static verification
- rank-32 LoRA trainer dry-run path
- external GPU payload build path
- adapter zip structure validation
- selected teammate source/code inspection under `team/minjaechoics/`

현재 repo만으로 재현할 수 없는 범위:

- full token/mask replay corpus preprocessing
- full private/generated datasets
- final selected adapter
- full NVIDIA 30B model training
- final leaderboard score
- teammate historical runs that depend on excluded local datasets/adapters/results

## Required External Inputs

| Input | Purpose | Public repo status |
| --- | --- | --- |
| Nemotron base model | tokenizer/model loading | not included |
| Full RSP dataset | actual training | not included |
| token/mask replay corpus | source analysis/replay reference | external artifact, not included |
| Competition `train.csv` | task/domain analysis | external Kaggle competition file |
| Hidden test set | actual scoring | not available locally |
| CUDA/PyTorch runtime | GPU training | environment-dependent |
| Dependency wheels | Kaggle/offline runtime support | not included |
| Eval split/setup | local evidence collection | not fully included |
| Teammate local artifacts | historical weak-domain experiments | selected code included, data/checkpoints/results excluded |

## Competition Files

Kaggle official files observed through the competition API:

| File | Public rows | Fields | Reproducibility note |
| --- | ---: | --- | --- |
| `train.csv` | 9,500 | `id`, `prompt`, `answer` | can be downloaded from Kaggle by participants |
| `test.csv` | 3 | `id`, `prompt` | sample only; scoring replaces it with several hundred hidden problems |

The public `test.csv` should not be used as final evidence. It is useful for checking submission mechanics, prompt formatting, and boxed-answer extraction behavior.

## Dataset Staging

Public repo convention:

```text
data/rsp_dataset/
├── rsp_anchor_sft.jsonl
├── rsp_decision_sft.jsonl
├── rsp_decision_preferences.jsonl
└── rsp_manifest.json
```

`data/`는 GitHub에 full data를 올리지 않는 전제로 `.gitignore` 처리합니다. Public sample은 `examples/rsp_dataset_sample/`에 있습니다. 이 sample은 schema preview용이며, full row-count gate를 통과하도록 설계된 dataset은 아닙니다.

Build command:

```bash
DATASET_DIR=data/rsp_dataset

python build_rsp_dataset.py \
  --anchor /path/to/anchor.jsonl \
  --equation /path/to/equation.jsonl \
  --target-repair /path/to/target_repair_rows.jsonl \
  --output-dir "$DATASET_DIR"
```

## Static Verification

Dataset verification:

```bash
python verify_rsp_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --json-output "$DATASET_DIR/rsp_verification.json"
```

Train shell verification:

```bash
python verify_rsp_train_shell.py \
  --train-script rsp_train_tokenmask_compatible.py \
  --dataset-verification "$DATASET_DIR/rsp_verification.json" \
  --json-output "$DATASET_DIR/rsp_train_shell_verification.json"
```

Expected verification style:

```json
{
  "rsp_dataset_valid": true,
  "current_minimum_goal_achieved": "no",
  "submission_allowed": false,
  "gpu_execution_allowed": false,
  "errors": 0
}
```

`submission_allowed=false`와 `gpu_execution_allowed=false`는 실패가 아니라 fail-closed design입니다. Dataset verification 단계에서 바로 submission 가능하다고 판단하지 않도록 설계한 것입니다.

## Trainer Dry Run

Model/tokenizer path가 준비되면 dry-run을 실행할 수 있습니다.

```bash
python rsp_train_tokenmask_compatible.py \
  --dataset-dir "$DATASET_DIR" \
  --model /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --dry-run
```

Dry-run은 full training validation이 아닙니다. 목적은 argument, dataset loading, normalization path를 빠르게 확인하는 것입니다.

## GPU Payload

External GPU training payload build:

```bash
python build_rsp_vast_payload.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir outputs/rsp_vast_payload/payload \
  --archive outputs/rsp_vast_payload/rsp_pro6000_payload.tar.gz
```

Training wrapper:

```bash
bash /workspace/rsp_vast/payload/rsp_run_train_pro6000.sh
```

이 경로는 PRO 6000-class GPU 환경을 상정합니다. T4/4-bit path는 feasibility probe로만 해석하고, final full training route로 과장하지 않습니다.

## Adapter Validation

Training 후 adapter zip이 있으면 다음을 실행합니다.

```bash
python verify_rsp_train_shell.py \
  --train-script rsp_train_tokenmask_compatible.py \
  --dataset-verification "$DATASET_DIR/rsp_verification.json" \
  --adapter-zip /path/to/submission.zip \
  --json-output /path/to/rsp_post_training_adapter_gate.json
```

검증 항목:

- `adapter_config.json`
- `adapter_model.safetensors`
- LoRA rank 32
- LoRA alpha 32
- LoRA dropout 0.0
- bias `none`
- target modules match
- LoRA A/B tensor headers structurally consistent

## Evaluation

Evaluation은 training과 분리합니다.

```bash
bash eval/run_eval.sh
```

또는 `eval/auto_evaluator.py`를 직접 실행할 수 있습니다. 실제 실행에는 base model, adapter path, eval data path, GPU runtime이 필요합니다.

Evaluation record에는 최소한 다음을 남겨야 합니다.

- base model path/hash
- adapter zip hash
- eval script version
- eval data/split identifier
- decoding parameters
- per-domain accuracy
- boxed-answer extraction behavior

Kaggle official evaluation uses vLLM with the submitted LoRA adapter, extracts the final answer primarily from `\boxed{}`, and scores exact string match or numeric relative tolerance `1e-2`. Local evaluation should mirror these constraints as closely as possible, but local results remain separate from leaderboard results.

## Reproducibility Boundaries

현재 상태:

| Area | Status |
| --- | --- |
| Public code inspection | supported |
| Example dataset verification | supported |
| Full dataset verification | requires private/full data |
| Full GPU training | requires external model/runtime |
| Adapter selection | not included |
| Final leaderboard score | not claimed |

이 repo는 “최종 점수를 즉시 재현하는 package”가 아니라, **어떤 데이터 설계와 학습 shell이 final adapter training으로 이어지는지 검증 가능한 형태로 보여주는 package**입니다.

`team/minjaechoics/`는 selected source artifact만 포함합니다. 해당 scripts에는 원래 runtime의 절대 경로가 남아 있을 수 있으며, 현재 repo에서 바로 실행되는 primary path로 보지 않습니다.
