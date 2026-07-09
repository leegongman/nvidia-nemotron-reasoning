# Training Methodology

## Objective

The current method trains a rank-32 LoRA adapter for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. The goal is to improve rule selection and final-answer stability on hidden reasoning tasks without claiming a final verified score.

The method is designed around the observed failure pattern: the model often produces plausible reasoning but selects the wrong transformation rule. RSP turns that into a supervised and preference-learning problem.

## Model and Adapter Configuration

| Setting | Value |
| --- | --- |
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Adapter type | LoRA/QLoRA-compatible PEFT adapter |
| Rank | 32 |
| Alpha | 32 |
| Dropout | 0.0 |
| Max sequence length | 8192 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `out_proj`, `up_proj`, `down_proj`, `lm_head` |
| Primary script | `rsp_train_huikang_compatible.py` |

The adapter shape is checked after training by verifying `adapter_config.json` and the safetensors LoRA tensor headers.

## Dataset Families

The RSP dataset has three row families.

### Anchor SFT

Anchor rows preserve broad behavior. They cover multiple reasoning domains and are not meant to overfit only the known failure domains.

Current verified count: 7,646 rows.

### Decision SFT

Decision rows include explicit rule traces for target domains. These examples train the model to emit a correct rule-selection path before the final boxed answer.

Current verified count: 2,666 rows.

### Decision Preferences

Preference rows contrast a chosen completion against a rejected completion. The rejected branch is designed to represent an incorrect rule choice or failure mode.

Current verified count: 2,500 rows.

## Training Phases

### Phase 1: Weighted Completion-Only SFT

The SFT phase masks prompt tokens and applies loss only to completion tokens. This keeps the objective focused on the reasoning trace and boxed answer rather than memorizing prompt formatting.

Inputs:

- `rsp_anchor_sft.jsonl`
- `rsp_decision_sft.jsonl`

Important behavior:

- Prompts are normalized with a boxed-answer instruction suffix.
- Completions are normalized to end with a terminal `\boxed{answer}`.
- Row weights are preserved as `sample_weight`.

### Phase 2: Pairwise Rule-Selection Preference Learning

The preference phase compares chosen and rejected completions for the same prompt. The implementation uses average log probabilities over completion tokens and a SimPO-style pairwise loss.

Inputs:

- `rsp_decision_preferences.jsonl`

Important behavior:

- The chosen and rejected branches are both completion-only scored.
- The loss rewards the chosen branch relative to the rejected branch.
- The phase is intentionally shorter than SFT and uses a lower learning rate.

## Verification Before Training

Before GPU training, `verify_rsp_dataset.py` checks:

- Required files exist.
- Required fields exist per row family.
- IDs are unique.
- Boxed answers match declared final answers.
- Bit-manipulation decision traces reproduce examples and target answers where applicable.
- Minimum row-count gates are satisfied.

`verify_rsp_train_shell.py` then checks:

- The training script keeps evaluation and submission disabled.
- The locked LoRA target modules match the expected module set.
- The preference objective and required RSP files are referenced.
- Forbidden submission/evaluation patterns are absent.

## What This Method Proves

The current artifacts support the following verified claims:

- The RSP dataset package passes static validation.
- The training script is configured as a train-only adapter builder.
- The pipeline is configured to produce an adapter artifact in a compatible GPU environment.
- The adapter artifact can be structurally validated after training.

## What This Method Does Not Yet Prove

The current artifacts do not verify that:

- A final RSP adapter has been selected.
- A final leaderboard score has been achieved.
- The adapter improves every reasoning domain.
- The historical best score is reproducible from the current RSP package.

Those claims require a completed training run, local evaluation evidence, adapter validation, and final evidence promotion.
