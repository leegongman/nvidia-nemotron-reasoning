# Project Status

이 문서는 GitHub 공개 시점에서 이 저장소가 어디까지 검증되었고, 어디부터는 아직 주장하면 안 되는지를 명확히 구분하기 위한 상태 문서입니다.

## 현재 공개 가능한 포지션

현재 가장 안전한 설명은 다음입니다.

> NVIDIA Nemotron 기반 reasoning adapter를 학습하기 위해, token/mask SFT corpus 구조를 분석하고 token/mask SFT, 보조 데이터 혼합, adapter 변환/SVD 분석, RSP(rule-selection prediction) 데이터 설계를 train-ready pipeline으로 정리한 프로젝트입니다.

이 프로젝트는 최종 점수나 최종 adapter를 공개 검증한 결과물이 아니라, reasoning failure를 데이터/학습 문제로 재정의하고 재현 가능한 학습 패키지로 정리한 engineering portfolio입니다.

## Evidence Status

| 영역 | 현재 증거 | 공개 주장 수준 |
| --- | --- | --- |
| Competition context | Kaggle Description/Evaluation/Data pages, `train.csv`/`test.csv` 구조 확인 | rank-32 LoRA adapter submission challenge로 설명 가능 |
| Competition train set | `train.csv` 9,500 rows, 6개 prompt family 확인 | domain-aware data analysis의 출발점으로 설명 가능 |
| token/mask corpus structure 분석 | `tokens/<problem_id>/synthetic.json`, `logprobs/index.jsonl`, token/mask format 분석 | 공개 데이터 구조를 분석하고 호환 포맷을 재구성 |
| 기본 SFT/LoRA 학습 | Kaggle notebook 기록, `rsp_train_tokenmask_compatible.py` | Nemotron LoRA adapter 학습 pipeline 구성 |
| CoT-selected SFT | rule-based correctness filtering, generated CoT 활용 기록 | 정답/추론 품질 기반 SFT data selection 실험 |
| 보조 데이터 혼합 | math replay, equation branch-map, auxiliary rehearsal rows 기록 | 기존 Token/mask 학습 데이터에 보조 데이터 소량 혼합 실험 |
| Domain weighting | `equation_numeric`, `bit_manipulation` 등 domain별 loss/sample weight 기록 | domain별 중요도 조정 실험 |
| Adapter merge/conversion | adapter 변환 notebook, SVD rank compression 코드 기록 | adapter transport, merge, SVD 압축 실험 |
| Team weak-domain artifacts | `team/minjaechoics/` selected files | teammate equation failure analysis와 residual/patch LoRA experiments 보존 |
| STaR-inspired 설계 | self-correction/STaR 관련 연구 및 생성기 기록 | STaR-inspired self-correction data design 검토 및 일부 실험 |
| RSP dataset package | `build_rsp_dataset.py`, `verify_rsp_dataset.py`, `rsp_schema.json` | rule-selection learning formulation을 재현 가능한 dataset package로 구성 |
| Evaluation safety | `eval/auto_evaluator.py`, `eval/run_eval.sh` | local evaluation/submission safety gate 구성 |
| 최종 adapter | 공개 clean repo에 checkpoint/adapter 미포함 | 최종 adapter 공개 검증은 주장하지 않음 |
| 최종 점수 | 공개 재현 가능한 final metric 미포함 | leaderboard score 달성 주장은 하지 않음 |

## 구현 상태

현재 clean public directory에는 다음 성격의 파일만 포함하는 것이 적절합니다.

- 공개 문서: `README.md`, `docs/*.md`
- 재현용 코드: RSP dataset builder, trainer wrapper, verifier, eval script
- 작은 예제 데이터 또는 schema
- 환경 문서: `requirements.txt`
- 공개 설정: `.gitignore`, `LICENSE`

원본 폴더의 전체 dataset, checkpoint, output, cache, submission artifact는 Git history에 넣지 않는 것이 맞습니다.

## 공개 전 유지해야 할 Claim Boundary

사용해도 되는 표현:

- train-ready pipeline
- reproducible training package
- competition-compatible rank-32 LoRA adapter workflow
- token/mask SFT format
- reasoning failure analysis
- rule-selection learning formulation
- NVIDIA Nemotron LoRA adapter workflow
- evaluation/submission safety gate
- adapter merge and SVD compression experiments
- domain-weighted auxiliary-data mixing experiments
- teammate weak-domain analysis and selected experiment artifacts

피해야 할 표현:

- final score를 달성했다는 표현
- 최종 순위/수상 성과처럼 보이는 표현
- 최고 성능 adapter처럼 보이는 표현
- leaderboard improvement가 확정 검증되었다는 표현
- fully validated final adapter
- 외부 원천 corpus 자체를 직접 제작했다는 표현
- STaR full training loop를 완성했다는 표현

## 아직 부족한 정보

공개 포트폴리오로는 충분히 설명 가능하지만, 연구/재현성 문서로 더 강해지려면 아래 정보가 추가로 필요합니다.

- 최종 선택 adapter의 hash, config, validation JSON
- 공개 가능한 local evaluation 결과와 per-domain breakdown
- 실제 최종 학습 hardware, runtime, seed, dependency lock
- dataset redistribution 가능 범위와 license 정리
- teammate imported files에 대한 최종 public permission/license 확인
- Kaggle notebook별 정확한 실험 순서와 결과 로그
- adapter merge 방식별 비교표
- SVD rank compression 전후 adapter 품질 비교 로그
- STaR-inspired data가 실제 학습에 들어간 범위와 sample count

## 공개 제외 대상

아래 항목은 삭제하지 말고 GitHub 제외 대상으로만 관리합니다.

- 원본/가공 대형 dataset
- `outputs/`
- `checkpoints/`, `checkpoint-*`
- `*.safetensors`, `*.bin`, `*.pt`, `*.pth`
- 대형 `jsonl`, `parquet`, `arrow`
- `submission.zip`
- Hugging Face/Kaggle/NVIDIA/W&B token 또는 credential 파일
- notebook output이 많은 임시 notebook
- third-party notebook dump
- `.hf_cache/`, `.dataset_deps/`, `__pycache__/`

## Public Release Checklist

- [x] README를 과장 없는 portfolio 설명으로 정리
- [x] dataset design과 claim boundary를 문서화
- [x] training methodology와 experiments를 기술 중심으로 구체화
- [x] eval 경로를 `eval/auto_evaluator.py` 기준으로 정리
- [x] `.gitignore`, `LICENSE`, `requirements.txt` 포함
- [ ] full dataset/checkpoint/final adapter 공개 여부 결정
- [ ] 최종 adapter가 있다면 별도 release artifact로 분리
- [ ] push 직전 credential/large-file scan 재실행
- [ ] README의 claim boundary 재검토

## Recommended Repository Description

추천 GitHub description:

> Train-ready NVIDIA Nemotron reasoning adapter pipeline with Token/mask SFT data analysis, auxiliary-data mixing, adapter conversion/SVD experiments, and evaluation safety gates.

추천 topics:

- `nvidia`
- `nemotron`
- `llm-finetuning`
- `lora`
- `peft`
- `reasoning`
- `synthetic-data`
- `sft`
- `adapter-training`
- `kaggle`

## 최종 상태 요약

문서와 공개용 clean directory는 portfolio 관점에서 사용할 수 있는 수준으로 정리되었습니다. 다만 이 저장소는 최종 검증 점수나 최종 adapter를 증명하는 저장소가 아니며, 공개 전에도 그 경계를 유지하는 것이 중요합니다.
