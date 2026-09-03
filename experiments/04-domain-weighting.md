# 04. Domain Weighting

bit/equation처럼 약한 domain에 더 강한 학습 신호를 주기 위해 sample/loss weight를 조정한 실험입니다. 확인된 full recipe는 원본 row와 순서를 바꾸지 않고, 해당 row의 assistant target token weight만 배수 조정합니다.

## Example

```json
{
  "equation_numeric": 1.20,
  "bit_manipulation": 1.10
}
```

이 값은 LoRA alpha가 아니라 data/loss weighting입니다. 노트북은 `train.csv`의 id/prompt/answer를 이용해 domain을 추정하고, equation·bit domain에만 multiplier를 적용합니다. cipher, cryptarithm, gravity, numeral, unit conversion은 이 단계의 multiplier 대상이 아닙니다.

## Objective

- weak domain update 강화
- easy/protected domain regression 최소화
- 보조 데이터 mixing의 영향 조절

## Cautions

weight를 키우면 특정 domain은 좋아질 수 있지만, 기존에 잘 맞히던 domain을 깨뜨릴 수 있습니다. 그래서 anchor rows와 evaluation gate가 필요했습니다.
