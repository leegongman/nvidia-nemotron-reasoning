# Loss Masking

이 프로젝트에서 가장 중요한 학습 신호 설계는 completion-only loss입니다.

## Token/Mask SFT

```text
prompt tokens      -> mask 0
assistant tokens   -> mask 1
final boxed answer -> mask 1
```

## Compared Concepts

| Method | 설명 |
| --- | --- |
| Response-only loss | response token에만 loss |
| Assistant-only loss | multi-turn chat에서 assistant 발화만 loss |
| Completion-only token/mask | prompt를 제외하고 reasoning/completion만 loss |
| NEFTune | loss masking은 아니지만 embedding noise로 generalization 보조 |

이 repo의 RSP trainer는 prompt label을 `-100`으로 masking합니다.
