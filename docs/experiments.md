# Experiments

## Summary

The project went through several candidate directions before settling on RSP as the public portfolio story. This document records the experiment history without treating any unverified result as the final outcome.

The strongest current claim is that the RSP package is train-ready and safety-gated. It is not a final scored adapter.

## Experiment Timeline

| Candidate | Goal | Outcome | Public interpretation |
| --- | --- | --- | --- |
| Historical internal note: 0.86 control/adapter | Preserve or reproduce a strong prior adapter path | Exists in project notes, but the full current recipe is not proven end-to-end from RSP | Historical context only, not a current claim |
| Candidate v1 | Basic adapter/data iteration | Reported weaker result than target | Superseded |
| Tong-logprob v8 | Explore alternate scoring or selection heuristics | Did not become the final path | Superseded |
| MG2 FGR-TR | Guarded trust-region method around known failure domains | Historical internal note: user-reported public score around 0.85, but rejected as current final route | Useful design context, not current claim |
| MG3 | Transfer/curriculum candidate | Historical internal note: user-reported public score around 0.09 and rejected | Negative result |
| RSP | Rule-selection post-training with anchor, decision, and preference rows | Dataset and train shell verified; no final adapter score yet | Current portfolio-ready pipeline |

## Why Earlier Directions Were Rejected

Earlier candidates tended to optimize around candidate outputs or local assumptions without enough separation between training, evaluation, and final evidence. Some paths had promising or historically useful notes, but they were not suitable as the main public claim because they lacked a clean reproducible chain from source data to final adapter to verified score.

MG3 is especially important as a negative result. It showed that a plausible curriculum or transfer story can fail badly when the learned behavior does not align with the challenge's hidden rule structure.

## Why RSP Became the Main Path

RSP was selected because it matches the observed failure mode more directly:

- Hidden tasks often fail at the rule-selection step.
- Bit-manipulation and equation tasks can be represented as explicit decision traces.
- Incorrect branches can be turned into rejected preference examples.
- Broad anchor rows can reduce the risk of damaging protected domains.
- Static verifiers can check many mistakes before GPU training.

## Current RSP Evidence

The RSP manifest reports:

| Row family | Count |
| --- | ---: |
| `anchor_sft` | 7,646 |
| `decision_sft` | 2,666 |
| `decision_preferences` | 2,500 |

The dataset verifier reports:

| Field | Value |
| --- | --- |
| `rsp_dataset_valid` | `true` |
| `errors` | `0` |
| `gpu_execution_allowed` | `false` |
| `submission_allowed` | `false` |

The `false` values for GPU execution and submission are intentional fail-closed states at the dataset-verification stage. Later workflow stages must explicitly promote artifacts after training and evaluation evidence exists.

## Claims to Avoid

Do not describe the current repository as:

- A final leaderboard-winning solution.
- A proven 0.86+ RSP adapter.
- A guaranteed score improvement.
- A completed final submission package.
- A fully cleaned public repository.

Use these descriptions instead:

- Train-ready Nemotron adapter pipeline.
- Reproducible RSP training package.
- Rule-selection learning formulation for reasoning failures.
- Evaluation and submission safety gates.
- Transparent experiment log with negative results.
