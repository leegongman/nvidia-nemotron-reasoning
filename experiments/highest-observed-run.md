# Highest Observed Run

내 기록에서 가장 강한 기준점으로 사용한 것은 A086 계열 rank-32 LoRA adapter입니다. 이 문서는 최종 leaderboard 성과를 재주장하는 문서가 아니라, 후속 실험이 어떤 adapter 구조와 recipe를 기준으로 시작했는지 기록합니다.

## Structure

- Nemotron 3 Nano 30B base model
- PEFT LoRA adapter
- rank 32 / alpha 32 / dropout 0.0
- target modules: attention, MLP, `lm_head`
- vLLM LoRARequest로 로드 가능한 adapter 구조

## Recipe File

가장 구체적인 학습 recipe는 Kaggle full-recipe notebook에 남아 있습니다. 공개용 정리에서는 원본 notebook과 외부 runtime 경로를 그대로 복사하지 않고, 실행 순서와 코드 근거를 [recipe-evidence.md](recipe-evidence.md)에 옮겼습니다.

핵심 설정은 다음과 같습니다.

| 항목 | recipe에서 확인된 값 |
| --- | --- |
| LoRA | `r=32`, `alpha=32`, `dropout=0.0` |
| Sequence | `max_seq_len=8192` |
| Batch | `batch_size=32`, `micro_batch_size=4` |
| Optimizer | AdamW, `weight_decay=0.0`, linear LR decay |
| Precision | base `bf16`, LoRA parameter `fp32` |
| Memory | gradient checkpointing, Cut Cross Entropy |
| MoE | expert LoRA weight tying, manual `lm_head` LoRA 보완 |

## Interpretation

A086은 이후 실험의 기준점이었습니다.

- local training pipeline이 강한 adapter와 어떤 점에서 달랐는지 비교
- target module set, dropout, max sequence length, token/mask loss 차이 확인
- adapter merge/SVD 실험에서 rank-32 constraint 기준으로 사용
- multi-adapter eval에서 baseline/control로 사용

## Public Scope

이 repo에는 A086 checkpoint나 `submission.zip`을 포함하지 않습니다. 이 문서는 최고 기록 계열 adapter의 구조와 후속 실험 방향을 설명하기 위한 기록입니다.
