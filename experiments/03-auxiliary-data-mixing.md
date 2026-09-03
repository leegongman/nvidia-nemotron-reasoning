# 03. Auxiliary Data Mixing

기존 token/mask 학습 stream에 `replay_math`와 소규모 rule replay를 섞어 adapter를 학습한 실험입니다. 핵심은 보조 데이터를 별도 format으로 학습하지 않고, 원본 corpus와 같은 `tokens`/`mask` 계약으로 변환한 뒤 interleave한 점입니다.

## Data Sources

| 데이터 | 목적 |
| --- | --- |
| Math replay | broad reasoning coverage 추가 |
| Equation branch-map | symbolic/equation weak domain 보강 |
| Target repair rows | 실제 failure와 비슷한 corrected trace 제공 |

## Recipe Evidence

확인된 recipe에서는 다음 순서가 사용되었습니다.

1. math message의 prompt와 assistant reasoning/final content를 chat template로 렌더링합니다.
2. prompt token은 `0`, assistant token은 `1`인 mask를 만듭니다.
3. 전체 길이를 8,192 token으로 제한하고, assistant target token 약 2M에 도달할 때까지 row를 보존합니다.
4. 원본 examples 사이에 replay row를 일정 간격으로 삽입합니다.
5. 별도로 생성한 24개 subtraction replay row도 학습 stream에 균등하게 삽입합니다.

노트북 출력에는 323개의 math replay row와 2,004,570개의 trainable replay answer token이 기록되어 있습니다. 이 수치는 해당 실행의 staging 결과이며, 모든 변형 실험에 공통인 최종 데이터셋 크기로 해석하면 안 됩니다.

## Mixing Strategy

- 보조 데이터가 target corpus를 압도하지 않게 interleave
- token budget cap 적용
- completion-only mask 유지
- weak domain 보강과 protected domain 보존 사이의 trade-off 확인

## Observations

보조 데이터는 많을수록 좋은 것이 아니라, 어떤 domain에 얼마나 섞는지가 중요했습니다.
