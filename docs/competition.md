# Competition

이 프로젝트의 출발점은 Kaggle의 NVIDIA Nemotron Model Reasoning Challenge입니다.

## Problem Format

대회 문제는 few-shot examples를 보고 숨은 규칙을 추론한 뒤 target input에 적용하는 puzzle입니다. 주요 family는 다음과 같습니다.

| Family | 문제 성격 |
| --- | --- |
| Bit manipulation | 8-bit input/output transformation rule 추론 |
| Gravity | examples에서 숨은 gravitational constant 추론 |
| Unit conversion | Wonderland unit conversion factor 추론 |
| Text encryption | substitution/encryption rule 복원 |
| Numeral conversion | 숫자를 다른 numeral system으로 변환 |
| Equation transformation | symbolic equation/string transformation rule 추론 |

## Submission Format

대회는 답안 CSV를 직접 제출하는 방식이 아니라, Nemotron base model에 붙일 LoRA adapter를 제출하는 방식입니다.

- base model: `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- submission: `submission.zip`
- adapter constraint: LoRA rank <= 32
- inference: vLLM + LoRA adapter
- answer extraction: final `\boxed{}` 중심

따라서 단순 prompt engineering보다 중요한 것은 **제출 가능한 adapter 구조**와 **hidden puzzle generalization**입니다.

## Historical Placement

기록된 competition placement는 **275 / 4,183 (상위 약 6.6%)**입니다. 이는 당시 제출 기록을 나타내는 historical placement이며, 현재 공개 저장소에서 최종 adapter와 leaderboard 결과를 재현했다는 의미는 아닙니다.
