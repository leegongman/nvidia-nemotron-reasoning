# 07. RSP Rule Selection

RSP는 Rule Selection Post-Training의 약자입니다.

이 실험은 reasoning failure를 단순 오답이 아니라 **잘못된 rule branch 선택**으로 본 것입니다.

## Dataset Families

| Family | 역할 |
| --- | --- |
| `anchor_sft` | 기존 behavior 보존 |
| `decision_sft` | 올바른 rule trace 학습 |
| `decision_preferences` | correct branch vs plausible wrong branch 비교 |

## Training

1. weighted completion-only SFT
2. SimPO-style pairwise preference learning

## Code

- `scripts/data/build_rsp_dataset.py`
- `scripts/data/verify_rsp_dataset.py`
- `scripts/train/rsp_train_tokenmask_compatible.py`
- `schemas/rsp_schema.json`
