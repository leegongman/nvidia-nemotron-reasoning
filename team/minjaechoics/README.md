# Minjae Team Artifacts

이 폴더는 함께 참여한 팀원 Minjae의 공개 GitHub repository에서 가져온 실험 코드와 분석 문서입니다.

원본:

- <https://github.com/minjaechoics/nvidia-nemotron3-reasoning-challenge>
- Imported commit: `718a744e`

이 파일들은 내 RSP public package와 섞어 “내가 단독 구현한 코드”로 주장하지 않습니다. 역할은 팀 단위 실험 증거, weak-domain 분석, residual/patch LoRA 실험 맥락을 보존하는 것입니다.

## Included Files

| Path | 역할 |
| --- | --- |
| `analysis/equation_numeric_error_analysis.md` | `equation_numeric` symbolic branch-map failure 분석 |
| `analysis/residual_lora_svd_principle.md` | frozen base adapter + residual LoRA + SVD rank-32 recompression 원리 |
| `Decoding_DS/token_to_decoded.py` | token/mask dataset을 decoded prompt/response 형태로 변환 |
| `Ortho_LoRA/ortholora_ver0.py` | task-wise Ortho-LoRA gradient projection 실험 |
| `Ortho_LoRA/run_ortholora_ver0.sh` | Ortho-LoRA 실행 wrapper |
| `additonal_tuning/*.py` | equation symbolic failure 보정을 위한 narrow SFT/GRPO/residual LoRA/DPO/EWC/guarded replay 실험 |
| `evaluator/*.py`, `evaluator/*.sh` | local metric-style evaluator와 equation-specific evaluator |
| `lambdalora/*.py` | patch LoRA training, lambda sweep, rank-32 SVD merge/export pipeline |
| `setup_env/*.py`, `setup_env/pipinstall.txt` | Nemotron module inspection과 environment setup helper |

## Excluded From Import

원본 repo에는 public clean repo에 넣으면 안 되는 파일도 포함되어 있어 제외했습니다.

- `.ssh/authorized_keys`
- `autosubmission.py`
- `dataset/`
- `archive.zip`
- `dataset_052001.zip`
- `*.safetensors`
- `*.jsonl`, `*.csv`
- `evaluator/results/`
- `unsloth_compiled_cache/`
- notebook outputs and temporary logs

## Technical Context

이 팀원 artifact가 보강하는 내용은 다음입니다.

- `equation_numeric`을 단일 카테고리로 보지 않고 `source_replay_*`, `symbolic_branch_*`, numeric answer, symbolic answer로 나눠 분석
- symbolic branch-map failure가 단순 산술 fallback, unknown operator fallback, punctuation parsing failure로 나타나는 점 확인
- 기존 rank-32 adapter를 직접 크게 업데이트하지 않고 residual LoRA를 작게 학습한 뒤 SVD로 다시 rank 32에 맞추는 실험
- lambda/scale sweep으로 weak-domain 보정과 protected-domain regression 사이의 trade-off 탐색
- Ortho-LoRA, DPO, EWC, guarded replay, GRPO-style update 같은 후속 실험 후보 검토

## Claim Boundary

이 폴더의 코드는 historical/team experiment artifacts입니다. 현재 clean repo의 핵심 재현 경로는 root의 RSP scripts입니다.

따라서 README에서는 다음처럼 표현합니다.

- 팀 실험에서 equation symbolic failure와 residual LoRA/SVD merge를 분석했다.
- 팀원 artifact를 참고해 RSP의 rule-selection framing을 강화했다.
- 이 파일들이 최종 verified adapter나 최종 leaderboard score를 증명하는 것은 아니다.
