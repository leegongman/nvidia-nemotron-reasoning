# Architecture

이 문서는 공개 저장소의 시스템 구조를 빠르게 확인하기 위한 overview입니다.

## Pipeline

```text
External dataset artifacts
        -> scripts/data/build_rsp_dataset.py
        -> data/rsp_dataset/
        -> scripts/data/verify_rsp_dataset.py
        -> scripts/train/rsp_train_tokenmask_compatible.py
        -> adapter structure gate
        -> local evaluation or runtime payload
```

## Components

| Component | Location | Role |
| --- | --- | --- |
| Dataset builder | `scripts/data/` | token/mask 및 RSP row 생성 |
| Dataset contract | `schemas/` | row schema와 rejection 조건 정의 |
| Training entrypoint | `scripts/train/` | Nemotron LoRA 학습 및 adapter 저장 |
| Evaluation | `scripts/eval/` | boxed answer extraction과 local scoring |
| Packaging and gates | `scripts/package/` | Kaggle/Vast bundle과 static validation |
| Experiment records | `experiments/`, `techniques/` | 실제 시도와 방법별 분석 보존 |

자세한 데이터 구조는 [`dataset.md`](dataset.md), 학습 흐름은 [`training-methodology.md`](training-methodology.md), 실행 조건은 [`reproducibility.md`](reproducibility.md)를 참고합니다.
