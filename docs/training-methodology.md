# Training Methodology

이 문서는 학습 방법의 entrypoint입니다. 전체 recipe evidence와 개별 실험은 연결된 문서에 분리했습니다.

## Method Overview

- 기존 token/mask corpus를 prompt와 assistant completion의 loss mask로 replay
- 보조 math/reasoning 데이터를 같은 token/mask 형식으로 변환해 interleave
- weak domain에 target-token loss weight를 적용
- rank-32 LoRA adapter를 학습하고 merge/SVD 변환 가능성을 검토
- STaR-style filtering과 rule-selection prediction을 후속 데이터 설계로 비교

## Primary References

- [`experiments/recipe-evidence.md`](../experiments/recipe-evidence.md): 확인된 Kaggle recipe
- [`experiments/01-token-mask-replay-sft.md`](../experiments/01-token-mask-replay-sft.md): token/mask replay
- [`experiments/03-auxiliary-data-mixing.md`](../experiments/03-auxiliary-data-mixing.md): auxiliary mixing
- [`experiments/04-domain-weighting.md`](../experiments/04-domain-weighting.md): domain weighting
- [`techniques/peft-methods.md`](../techniques/peft-methods.md): PEFT method boundary
- [`scripts/train/rsp_train_tokenmask_compatible.py`](../scripts/train/rsp_train_tokenmask_compatible.py): training entrypoint

## Claim Boundary

이 문서는 학습 방법과 구현 경로를 설명합니다. 최종 adapter와 leaderboard score가 공개 검증되었다고 주장하지 않습니다.
