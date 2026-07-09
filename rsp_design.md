# RSP Design

현재 최소 목표 달성: 아니요

## Candidate

`RSP: Rule Selection Post-Training`

RSP is the single active design direction. It replaces broad fresh SFT and narrow W1-only repair with solver-verified decision repair for the NVIDIA Nemotron Model Reasoning Challenge.

## Competition-Specific Objective

The competition evaluates a rank-limited LoRA adapter on `Nemotron-3-Nano-30B-A3B-BF16`. The scored tasks are not broad natural language knowledge tasks. They are hidden transformation-rule puzzles, evaluated by greedy generation and final answer extraction from `\boxed{}`.

Therefore the adapter must improve:

- hidden rule selection;
- deterministic trace execution;
- boxed final-answer stability;
- protected-domain preservation.

It must not broadly overwrite the base or huikang/W1 reasoning behavior.

## Model And Adapter Constraints

RSP assumes the huikang-compatible adapter shell:

- BF16 LoRA, not 4-bit training.
- rank 32, alpha 32, dropout 0.0.
- max sequence length 8192.
- completion-only loss.
- target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `in_proj`
  - `out_proj`
  - `up_proj`
  - `down_proj`
  - `lm_head`
- adapter output must contain non-empty safetensors LoRA A/B tensors for declared target modules.
- saved adapter must smoke-load into the base model before any zip is considered usable.

## Domain Formalization

Every row is treated as:

```text
PARSE -> CANDIDATES -> SELECT -> VERIFY -> ANSWER
```

The central training signal is not generic CoT imitation. It is verified selection of the correct rule branch over plausible wrong branches.

## Decision Points

| Domain | Decision Point |
| --- | --- |
| bit_manipulation | output-bit `B0..B7` gate selection, e.g. `XOR(i2,i7)` vs `OR(i2,i7)` |
| equation_numeric | DSL/program/branch/position/operator selection |
| cipher | bijective character or word-candidate mapping |
| gravity | scalar interval/median selection |
| unit_conversion | factor interval/median selection |
| numeral | deterministic conversion rule preservation |
| cryptarithm | symbol transformation and arithmetic-consistency selection |

## Row Types

RSP uses three row families:

1. `anchor_sft`
   - Huikang-style clean deterministic traces.
   - Purpose: preserve solved-domain behavior and trace style.

2. `decision_sft`
   - Compact rows that emphasize selected rule decisions.
   - Purpose: increase probability of correct decision tokens.

3. `decision_preference`
   - Pairwise correct branch vs plausible wrong branch.
   - Purpose: suppress the specific rule-selection mistakes that cause wrong final answers.

## Initial Data Sources

- Clean anchor:
  - `data/source/anchor_sft.jsonl`
- Equation symbolic coverage:
  - `data/source/equation_numeric.jsonl`
- Existing verified target repair rows:
  - `data/source/target_repair_rows.jsonl`

These are inputs to be verified, not accepted by trust.

## Target Shape

RSP targets above-minimum local performance:

| Domain | Target |
| --- | ---: |
| cryptarithm | 20/20 |
| equation_numeric | 17/20+ |
| bit_manipulation | 19/20+ |
| cipher | 20/20 |
| gravity | 20/20 |
| numeral | 20/20 |
| unit_conversion | 20/20 |

## Rejection Conditions

RSP is not trainable if any of the following holds:

- exact eval prompt overlap is found;
- a row's terminal boxed answer disagrees with `final_answer`;
- a bit selected gate does not reproduce all examples and target;
- an equation DSL does not reproduce all examples and target;
- a decision preference has a rejected branch with the same final answer as the chosen branch;
- malformed boxed answers are present;
- anchor rows are outnumbered by repair/preference rows;
- adapter continuation or trajectory reproduction is not available;
- adapter save/load structure gates are absent.

## Current Status

RSP is not a trained adapter. It is a candidate design being converted into data builders and verifiers.
