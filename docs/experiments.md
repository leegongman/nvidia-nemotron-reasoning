# Experiments

이 문서는 실제로 진행한 실험과 기술적 판단을 정리합니다. 목적은 “최종 점수 달성”을 주장하는 것이 아니라, 어떤 가설을 세우고 어떤 구현/데이터 설계를 시도했는지 투명하게 보여주는 것입니다.

## Summary

프로젝트는 크게 네 단계로 진행되었습니다.

1. Kaggle competition dataset과 Huikang 공개 데이터 구조 분석
2. CoT-selected SFT와 Huikang-style replay 학습 실험
3. auxiliary data mixing, domain weighting, adapter merge/conversion/SVD 실험
4. teammate weak-domain experiments와 residual/patch LoRA 분석
5. STaR-inspired self-correction과 RSP rule-selection dataset 정리

현재 가장 강한 public claim은 final score가 아니라 다음입니다.

> NVIDIA Nemotron 기반 reasoning adapter 학습을 위해 Huikang-style SFT format을 분석하고, 보조 데이터 mixing과 rule-selection preference learning으로 확장한 train-ready pipeline을 구현했다.

## Experiment Matrix

| Experiment | Goal | Method | Evidence | Claim boundary |
| --- | --- | --- | --- | --- |
| Competition dataset analysis | task/domain 구조 이해 | `train.csv`, `test.csv`, official evaluation page 확인 | Kaggle competition files/pages | hidden test distribution claim 아님 |
| Huikang dataset analysis | 원천 데이터 구조 이해 | `tokens`, `mask`, `logprobs/index.jsonl` 분석 | Huikang snapshot, decode notebook | 데이터 제작자 claim 아님 |
| CoT-selected SFT | noisy generated CoT 제거 | rule-based checker로 correct CoT만 SFT | `nemotron-sft-lora-with-cot-my-dataset` | final score evidence 아님 |
| Huikang replay SFT | token/mask replay 학습 | epoch-0 order로 `synthetic.json` 로드 | Huikang notebook, local replay notebooks | 원본 recipe 완전 재현 claim 아님 |
| Merged balanced SFT | 개인/보조 데이터를 Huikang-style corpus로 섞기 | `merged_sft_dataset/tokens` 학습 | `sft-my-data-balace` | dataset ownership claim 아님 |
| Math replay mixing | broad reasoning coverage 추가 | math replay를 token/mask로 변환 후 interleave | double-update notebook | isolated improvement evidence 아님 |
| E3 branch-map mixing | equation rule-selection 보강 | branch-map completion rows를 tokenized 후 interleave | double-update notebook | final equation gain evidence 아님 |
| Domain weighting | weak domain 보강 | equation/bit sample/loss weight 조정 | `w1_provenance_contract.json` | LoRA alpha tuning과 구분 |
| Adapter merge | 여러 adapter의 signal 결합 | adapter delta/key mapping, merge/transport 실험 | conversion notebooks, local artifacts | final merged adapter claim 아님 |
| Adapter conversion | Tinker adapter를 PEFT submission 형태로 변환 | key mapping, fused projection handling | `tinker-adapter...` notebooks | conversion은 lossy 가능 |
| SVD rank compression | rank/fused projection mismatch 해결 | LoRA delta SVD 후 rank 32 factorization | conversion notebooks, `glyphmatics` | 성능 보장 claim 아님 |
| Teammate weak-domain analysis | equation symbolic failure 원인 파악 | `hk_*`/`my_*`, numeric/symbolic split 분석 | `team/minjaechoics/analysis/*` | team artifact로 분리 |
| Residual/patch LoRA | 기존 adapter 보존하며 weak domain 보정 | frozen adapter + residual/patch LoRA + SVD rank-32 merge | `team/minjaechoics/additonal_tuning`, `team/minjaechoics/lambdalora` | final adapter evidence 아님 |
| Ortho-LoRA / guard methods | protected domain regression 완화 | task-wise gradient projection, guarded replay, DPO/EWC, GRPO-style update | `team/minjaechoics/Ortho_LoRA`, `team/minjaechoics/additonal_tuning` | experimental |
| STaR-inspired data | self-correction reasoning 검토 | answer/reasoning 개선 loop와 repair rows 설계 | STaR notes/generators | full STaR loop 완료 claim 아님 |
| Multi-adapter eval | adapter별 차이 비교 | same eval sample, separate output dirs | `multi_adapter_eval.ipynb` | final public eval 아님 |
| RSP | rule-selection 문제로 재정의 | anchor SFT + decision SFT + preference rows | public scripts in repo | train-ready, not final adapter |

## 1. Competition Dataset Analysis

공식 대회는 NVIDIA Research가 만든 reasoning benchmark에서 Nemotron-3-Nano-30B용 LoRA adapter를 제출하는 형태입니다.

공개 파일 기준:

| File | Rows | Fields |
| --- | ---: | --- |
| `train.csv` | 9,500 | `id`, `prompt`, `answer` |
| `test.csv` | 3 sample rows | `id`, `prompt` |

공식 설명상 `test.csv`는 submission authoring용 sample이며, scoring 시에는 수백 개 hidden problems로 교체됩니다. 따라서 local `test.csv` 3개를 잘 푸는 것은 실제 성능 검증이 아닙니다.

확인한 train domain 분포:

| Domain | Rows |
| --- | ---: |
| Bit manipulation | 1,602 |
| Gravity | 1,597 |
| Unit conversion | 1,594 |
| Text encryption | 1,576 |
| Numeral conversion | 1,576 |
| Equation transformation | 1,555 |

평가 방식:

- 제출물은 rank 32 이하 LoRA adapter가 들어 있는 `submission.zip`
- evaluator는 vLLM으로 Nemotron base model + LoRA adapter를 로드
- final answer는 `\boxed{}`에서 우선 추출
- exact string match 또는 numeric relative tolerance `1e-2`
- score는 correctly answered proportion

이 조건 때문에 실험은 “답을 맞히는 notebook”보다 “제출 가능한 rank-32 adapter를 안정적으로 만드는 pipeline”에 집중했습니다.

## 2. Huikang Dataset Analysis

Huikang 공개 자료에서 확인한 핵심은 데이터가 일반 text JSONL이 아니라 **pre-tokenized SFT corpus**라는 점입니다.

주요 구조:

```text
training/sft/04-08-16-14/
├── tokens/<problem_id>/synthetic.json
└── logprobs/index.jsonl
```

분석 결과:

- `index.jsonl`은 epoch, step, problem_id를 포함해 training order를 복원할 수 있습니다.
- `synthetic.json`은 `tokens`와 `mask`를 포함합니다.
- prompt 영역은 mask 0, assistant reasoning/final answer 영역은 mask 1입니다.
- 학습은 `tokens[:-1] -> tokens[1:]` next-token prediction 형태로 구성됩니다.

이 분석은 이후 모든 Huikang-compatible training shell의 기준이 되었습니다.

## 3. CoT-Selected SFT

초기 notebook `nemotron-sft-lora-with-cot-my-dataset`에서는 generated CoT를 모두 쓰지 않고, rule-based checker를 통과한 row만 선택했습니다.

실험 아이디어:

- LLM으로 CoT reasoning 생성
- final answer를 `\boxed{}`에서 추출
- domain별 checker로 correct answer 여부 확인
- correct row만 SFT dataset으로 사용

Domain-specific prompt engineering도 적용했습니다.

| Domain | 설계 방향 |
| --- | --- |
| Text Encryption | Wonderland 77-word dictionary mapping 정보 활용 |
| Bit Manipulation | output bit별 boolean candidate 분석 |
| Unit/Gravity/Numeral | step-by-step deterministic reasoning |
| Equation | 초기에는 solver 지식이 부족해 개선 여지가 남음 |

해석:

- 데이터 품질과 correctness filtering이 중요하다는 방향을 확인했습니다.
- 다만 이 notebook의 setting은 Huikang-style token/mask replay와 완전히 같지는 않습니다.

## 4. Huikang-Style Replay and Balanced SFT

`sft-my-data-balace` notebook은 Huikang 원본과 같은 구조의 `tokens` directory와 `logprobs/index.jsonl`을 읽는 학습 shell을 사용했습니다.

주요 설정:

- rank 32
- alpha 32
- dropout 0.0
- max sequence length 8192
- target modules: attention, MLP, `in_proj`, `out_proj`, `lm_head`
- BF16 model load
- completion mask 기반 weighted loss
- adapter output을 `submission.zip` 형태로 package

이 실험은 “text formatting SFT”보다 Huikang-compatible token/mask format이 중요하다는 판단으로 이어졌습니다.

## 5. Auxiliary Data Mixing

### Math Replay

Double-update notebook에서는 외부 math replay JSONL을 Nemotron chat template으로 렌더링하고 token/mask format으로 변환했습니다.

핵심 구현:

- prompt-only ids: `apply_chat_template(messages[:-1], add_generation_prompt=True)`
- full ids: `apply_chat_template(messages, add_generation_prompt=False)`
- prompt length까지 mask 0
- assistant reasoning/final content mask 1
- target replay answer tokens cap 적용
- Huikang examples 사이에 replay rows 삽입

이 방식은 broad reasoning behavior를 보강하려는 시도였습니다.

### E3 Branch-Map

Equation symbolic branch-map rows도 같은 방식으로 tokenized했습니다.

핵심 구현:

- prompt/completion pair를 chat template으로 변환
- max length 8192 초과 row 제외
- completion 영역만 loss mask 1
- math replay가 섞인 stream에 다시 E3 rows interleave

이 실험은 equation/rule-selection failure를 보강하려는 시도였습니다.

## 6. Domain-Specific Weighting

W1 provenance evidence에서 다음 reweighting이 확인됩니다.

```json
{
  "equation_numeric": 1.25,
  "bit_manipulation": 1.1
}
```

해석:

- weak domain을 조금 더 자주/강하게 학습시키려는 시도입니다.
- 문서에서는 이를 `domain-specific loss/sample weighting`으로 표현합니다.
- LoRA config의 `lora_alpha`와 혼동하지 않기 위해 “domain alpha”라는 표현은 피합니다.

## 7. Adapter Merge, Conversion, and SVD

실험 과정에서는 단일 adapter 학습뿐 아니라, 여러 adapter에서 얻은 signal을 결합하거나 submission-compatible PEFT adapter로 변환하는 시도도 있었습니다.

Adapter merge/transport에서 다룬 문제:

- 같은 base model을 대상으로 한 여러 LoRA delta의 결합
- domain별로 강한 adapter signal을 어떻게 섞을지에 대한 실험
- Tinker 계열 adapter와 PEFT submission adapter의 key/shape 차이
- fused projection과 per-module projection 차이
- rank 64 이상 또는 fused delta를 rank 32 제한에 맞추는 문제

Huikang/Tinker 계열 adapter를 Kaggle submission-compatible PEFT adapter로 맞추는 과정에서 SVD가 사용되었습니다.

대표 문제:

- Tinker key prefix와 PEFT key prefix 차이
- fused expert projection과 per-expert projection 차이
- `gate_proj` + `x_proj`를 `in_proj` 형태로 맞추는 문제
- rank 64를 rank 32로 줄여야 하는 문제

SVD 실험의 역할:

```text
LoRA delta = B @ A
delta를 SVD로 factorize
top-k singular directions만 사용
rank 32 LoRA A/B로 재구성
```

중요한 한계:

- SVD compression은 lossy입니다.
- singular mass 손실이 training-serving mismatch를 만들 수 있습니다.
- 따라서 이 문서에서는 SVD를 final performance technique이 아니라 adapter compatibility/transport experiment로 설명합니다.

## 8. GlyphMatics Adapter Transport

`glyphmatics` notebook은 SVD compression을 더 공격적으로 조정한 adapter transport 실험입니다.

확인된 아이디어:

- `SVD_ENERGY_GAIN_CAP`
- `ROW_NORM_GAIN_CAP`
- `PAIRFOLD_MIN_SIM`
- pairfold direction matching
- dual-phase rebuild fallback
- forced rank 32 유지

이 실험은 adapter 변환 과정에서 손실된 singular direction/energy를 어느 정도 보정할 수 있는지 탐색한 것입니다.

Claim boundary:

- experimental adapter transport입니다.
- clean public package의 core claim은 아닙니다.
- final adapter 성능을 보장하지 않습니다.

## 9. Teammate Weak-Domain Experiments

함께 참여한 teammate Minjae의 공개 repository에서 weak-domain 보정 실험 파일을 선별해 `team/minjaechoics/`에 포함했습니다. 이 파일들은 내 RSP package와 별도로 보관하며, 팀 단위 실험 evidence로만 사용합니다.

### Equation Numeric Error Analysis

`team/minjaechoics/analysis/equation_numeric_error_analysis.md`는 `equation_numeric`을 단일 aggregate로 보지 않고 다음처럼 쪼개어 분석합니다.

| Split | 의미 |
| --- | --- |
| `hk_*` | Huikang-style arithmetic/numeric examples |
| `my_*` | 추가 구성된 symbolic branch-map examples |
| numeric answer | 숫자형 final answer |
| symbolic answer | punctuation/symbol 기반 final answer |

핵심 관찰은 `equation_numeric` 약점이 단순 계산 능력 부족이라기보다, `input[2]` 기준 branch 선택, position-specific symbol mapping, punctuation parsing failure에서 발생했다는 점입니다.

주요 failure mode:

- symbolic branch-map 문제를 arithmetic/operator 문제로 오해
- prompt에 없는 operator가 나오면 absolute difference나 concatenation으로 fallback
- punctuation symbol을 내부 letter code로 바꾼 뒤 최종 symbol alphabet으로 복원하지 못함
- malformed `\boxed{}` output이 answer extraction risk를 키움

이 분석은 RSP의 “rule-selection failure” framing과 직접 연결됩니다.

### Residual LoRA + SVD Merge

`team/minjaechoics/analysis/residual_lora_svd_principle.md`와 `team/minjaechoics/additonal_tuning/sft_residual_lora_svd.py`는 기존 rank-32 adapter를 직접 크게 바꾸지 않고, 작은 residual LoRA를 얹은 뒤 다시 rank 32로 압축하는 방식을 설명합니다.

개념:

```text
W = W_base + DeltaW_existing + DeltaW_residual
DeltaW_final ~= DeltaW_existing + scale * DeltaW_residual
rank(DeltaW_final) <= 32
```

이 방식의 목적은 weak domain을 보정하면서 protected domain regression을 줄이는 것입니다. 다만 scale이 너무 작으면 개선이 없고, 너무 크면 기존 정답을 깨뜨릴 수 있습니다. 따라서 이 실험은 “성능 보장 기법”이 아니라 **adapter stability와 rank-32 submission constraint 사이의 trade-off 탐색**으로 해석합니다.

### Patch/Lambda LoRA

`team/minjaechoics/lambdalora/train_patch_lora_and_merge.py`는 patch LoRA를 학습한 뒤 여러 lambda scale로 export하고, SVD로 rank 32에 맞추는 pipeline입니다.

핵심 아이디어:

- existing adapter는 baseline으로 유지
- patch adapter만 narrow data로 학습
- lambda sweep으로 영향력 조정
- competition `max_lora_rank=32`에 맞게 SVD merge
- adapter rank check와 eval config manifest를 남김

### Guarded Replay, DPO, EWC, GRPO-Style Update

`team/minjaechoics/additonal_tuning/`에는 equation symbolic failure를 보정하기 위한 여러 좁은 실험이 있습니다.

| Script family | 목적 |
| --- | --- |
| `sft_symbolic_fix.py` | symbolic/equation wrong rows를 oversample하고 replay rows로 drift 완화 |
| `sft_debug_teacher_trace_micro.py` | evaluator `reference_response` 기반 teacher trace micro SFT |
| `sft_residual_lora_dpo.py` | chosen/rejected hard negative pair로 residual DPO |
| `sft_residual_lora_ewc.py` | old behavior를 Fisher/EWC guard로 보호 |
| `sft_residual_lora_guarded_replay.py` | weak-domain update와 real guard replay 결합 |
| `grpo_symbolic_fix.py` | competition metric reward를 이용한 direct grouped policy-gradient style update |

이 실험들은 최종 public claim을 강화하기보다, 왜 RSP에서 `decision_preferences`와 guard/anchor rows가 필요한지 설명하는 배경입니다.

### Ortho-LoRA

`team/minjaechoics/Ortho_LoRA/ortholora_ver0.py`는 task-wise gradient projection을 통해 weak task와 rest task의 gradient interference를 줄이려는 실험입니다.

이 코드는 다음을 포함합니다.

- pre-tokenized token/mask training
- Unsloth FastLanguageModel
- Cut Cross Entropy style loss path
- task-wise Ortho-LoRA gradient projection
- LoRA adapter checkpoint saving

현재 clean repo의 primary path는 RSP trainer이므로, Ortho-LoRA는 후속 실험/팀 실험 artifact로만 둡니다.

## 10. STaR-Inspired Self-Correction

STaR(Self-Taught Reasoner)는 모델이 reasoning을 생성하고, 정답 여부를 기준으로 reasoning을 개선하거나 재생성하는 반복 학습 아이디어입니다. 이 프로젝트에서는 STaR를 final fully automated training loop로 완성했다고 주장하지 않고, 다음 방향의 self-correction data design으로 검토했습니다.

검토/실험한 방향:

- generated reasoning에서 final answer를 추출
- checker로 correct/incorrect를 구분
- incorrect reasoning을 repair target 또는 rejected branch로 변환
- correct reasoning을 SFT row 또는 chosen branch로 사용
- RSP의 `decision_preferences`로 correct branch vs plausible wrong branch를 명시화

RSP와의 연결:

```text
wrong reasoning
-> failure point detection
-> corrected rule trace
-> chosen/rejected preference pair
```

따라서 STaR 관련 작업은 “full STaR training loop 구현 완료”가 아니라, self-correction과 preference data design이 RSP로 이어진 실험적 배경으로 설명합니다.

## 11. Multi-Adapter Evaluation

`multi_adapter_eval.ipynb`는 여러 adapter를 같은 eval input으로 비교하기 위한 notebook입니다.

구조:

- fixed eval sample
- adapter별 output directory
- same model path
- same decoding config
- per-adapter result collection

비교한 adapter 이름에는 historical score-like label이 포함되어 있지만, README에서는 이 숫자를 final claim으로 쓰지 않습니다. docs에서는 historical internal note로만 다룹니다.

## 12. RSP: Rule Selection Post-Training

RSP는 앞선 실험의 결론을 public-safe package로 정리한 형태입니다.

핵심 재정의:

```text
reasoning failure
-> rule decision point
-> correct branch vs plausible wrong branch
-> SFT + pairwise preference learning
```

RSP dataset:

- `anchor_sft`: broad behavior preservation
- `decision_sft`: correct rule trace supervision
- `decision_preferences`: chosen/rejected branch comparison

RSP trainer:

- completion-only SFT
- weighted rows
- lower-LR preference phase
- train-only safety flags
- adapter zip structural validation

## Historical Internal Notes

내부 실험 기록에는 0.85, 0.86, 0.09 같은 historical score-like notes가 존재합니다. 하지만 이 repo에서는 다음과 같이만 취급합니다.

- historical context
- rejected/superseded candidate evidence
- current RSP package의 final result가 아님

README에서는 이 숫자를 제거했고, 현재 public claim은 train-ready pipeline과 documented experiment history로 제한합니다.

## Claims to Avoid

사용하지 않을 표현:

- final score를 달성했다는 표현
- 최종 순위/수상 성과처럼 보이는 표현
- 개선을 보장하는 표현
- 0.86+ adapter가 현재 repo에서 확정 재현된다는 표현
- Huikang 원천 데이터 제작자처럼 보이는 표현
- STaR full loop 완료 구현 표현

사용할 표현:

- Huikang-style dataset analysis
- curated/restructured/augmented training package
- train-ready Nemotron adapter pipeline
- adapter conversion and SVD analysis
- rule-selection learning formulation
- evaluation/submission safety gates
