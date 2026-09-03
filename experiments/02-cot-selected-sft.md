# 02. CoT-Selected SFT

generated CoT를 전부 사용하는 대신, rule-based checker를 통과한 reasoning만 SFT 데이터로 쓰는 실험입니다.

## Objective

- noisy CoT 제거
- final answer correctness 확보
- domain별 prompt 방식 비교
- answer-only가 아니라 reasoning trace 학습

## Setup

1. domain-specific prompt로 reasoning 생성
2. `\boxed{}`에서 final answer 추출
3. rule checker로 정답 여부 확인
4. 통과한 row만 SFT 후보로 사용

## Interpretation

데이터 양보다 correctness filtering이 중요하다는 방향을 확인했습니다. 다만 이 실험은 token/mask replay와 완전히 같은 학습 format은 아니기 때문에, 최종 score claim으로 쓰지는 않습니다.
