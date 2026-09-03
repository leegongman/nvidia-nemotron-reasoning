# Project Status

현재 상태는 **portfolio draft**입니다. Documentation is ready; repository contents still need cleanup.

## What This Repo Shows

- NVIDIA Nemotron adapter challenge 이해
- 기록된 competition placement: 275 / 4,183 (상위 약 6.6%)
- 외부 Nemotron corpus provenance 및 attribution 문서화
- token/mask SFT 구조 분석
- 데이터셋 재구성 및 보조 데이터 mixing 실험
- PEFT/LoRA 기반 adapter 학습 pipeline
- 공개 가능한 작은 recipe example: `configs/rsp_default.example.json`
- adapter merge/SVD, STaR-style, preference learning 등 실험 정리
- train/eval/submission safety 분리

## What It Does Not Include

- full dataset
- final adapter checkpoint
- `submission.zip`
- Kaggle private score evidence
- 대형 notebook outputs
- credential/token

## Current Direction

문서와 파일 구조는 내가 실제로 실험한 기술과 데이터 구성이 보이도록 재정리되어 있습니다. 공개 전에는 파일 provenance, 팀 artifact 범위, 실행 환경 의존성을 다시 확인해야 합니다.
