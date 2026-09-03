# Best Observed Adapter

내 기록상 가장 강한 기준점은 A086 계열 adapter입니다.

이 문서는 checkpoint를 공개하는 것이 아니라, 이후 실험의 기준이 된 구조를 정리합니다.

## Structure

| 항목 | 값 |
| --- | --- |
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Adapter type | PEFT LoRA |
| Rank | 32 |
| Alpha | 32 |
| Dropout | 0.0 |
| Bias | none |
| Tensor dtype | F32 LoRA tensors observed |
| Layers | 52 |
| Experts | 128 |

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
in_proj
out_proj
up_proj
down_proj
lm_head
```

## Why It Mattered

이 adapter는 이후 실험의 기준점이었습니다.

- local training이 왜 강한 adapter와 다르게 나오는지 비교
- target module set, sequence length, dropout, mask 방식 차이 확인
- adapter merge/SVD 실험에서 rank-32 constraint 기준으로 사용
- multi-adapter eval에서 baseline/control로 사용

## Claim Boundary

이 repo에는 A086 checkpoint나 submission artifact가 포함되어 있지 않습니다.

따라서 이 repo의 claim은 다음 수준입니다.

- highest observed adapter structure를 분석했다.
- 그 구조를 참고해 train-ready pipeline을 정리했다.
- 동일 점수 재현이나 최종 leaderboard 성능은 주장하지 않는다.
