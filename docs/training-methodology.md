# Training Methodology

이 문서는 Nemotron adapter 학습 방법론을 정리합니다. 설명은 한국어 중심으로 작성하되, 구현과 연결되는 핵심 용어는 영어 원문을 유지합니다.

## Objective

목표는 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`에 적용 가능한 rank-32 LoRA adapter 학습 파이프라인을 구축하는 것입니다. 현재 repo는 최종 점수 달성을 주장하지 않고, 다음을 검증 가능한 산출물로 제시합니다.

- Huikang-compatible token/mask SFT 이해 및 재구성
- auxiliary dataset mixing 실험 기록
- RSP rule-selection dataset schema
- train-only adapter builder
- dataset, train shell, adapter structure verification

## Competition Constraints

NVIDIA Nemotron Model Reasoning Challenge의 공식 조건은 학습 방법보다 **최종 adapter compatibility**를 더 강하게 제한합니다.

| Constraint | 공식/실무 의미 |
| --- | --- |
| Base model | NVIDIA Nemotron-3-Nano-30B 계열 모델을 evaluator가 로드 |
| Submission artifact | `submission.zip` 안에 LoRA adapter 포함 |
| Adapter rank | `max_lora_rank <= 32` |
| Required config | `adapter_config.json` 필요 |
| Inference engine | vLLM 기반 evaluator |
| Decoding | temperature `0.0`, top_p `1.0` |
| Max model length | 8192 |
| Answer format | final answer를 `\boxed{}` 안에 넣는 것이 가장 안전 |
| Metric | exact string match 또는 numeric relative tolerance `1e-2` |

따라서 이 프로젝트의 training methodology는 단순히 loss를 낮추는 것이 아니라, rank-32 PEFT adapter로 변환 가능한 학습/검증 경로를 유지하는 데 초점을 둡니다.

## Huikang-Compatible SFT Format

Huikang 방식의 핵심은 text를 바로 SFTTrainer에 넣는 것이 아니라, pre-tokenized `tokens`와 `mask`를 사용한다는 점입니다.

```text
tokens = prompt_tokens + assistant_reasoning_tokens + final_answer_tokens
mask   = 0 for prompt, 1 for assistant completion
```

학습 row는 trainer 내부에서 다음처럼 변환됩니다.

```text
input_ids = tokens[:-1]
targets   = tokens[1:]
weights   = mask[1:]
```

이 구조의 의미:

- prompt token은 loss에서 제외됩니다.
- reasoning trace와 `\boxed{answer}` token만 loss를 받습니다.
- 모델은 문제 자체를 외우기보다, solver trace를 completion으로 재현하도록 학습됩니다.

이 format은 `decode_huikang_replay_samples.ipynb`, Huikang public notebook, `sft-my-data-balace.ipynb`, `rsp_train_huikang_compatible.py`를 통해 확인한 기준입니다.

## Model and Adapter Configuration

| Setting | Value |
| --- | --- |
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Adapter type | PEFT LoRA |
| Rank | 32 |
| Alpha | 32 |
| Dropout | 0.0 |
| Max sequence length | 8192 |
| Main precision route | BF16 LoRA |
| Resource probe route | 4-bit / QLoRA feasibility only |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `out_proj`, `up_proj`, `down_proj`, `lm_head` |
| Primary public script | `rsp_train_huikang_compatible.py` |

초기 일부 실험에서는 `lora_dropout=0.05`, shorter max length, text-format SFT도 사용했지만, Huikang-compatible 방향에서는 8192 context, rank 32, alpha 32, dropout 0.0, completion-only loss가 더 중요한 기준으로 정리되었습니다.

## Training Optimizations

실험과 public package에서 확인되는 주요 training optimization은 다음과 같습니다.

| Technique | Role | Current interpretation |
| --- | --- | --- |
| BF16 mixed precision | 30B-class adapter training memory/throughput 경로 | main full-training path |
| Gradient checkpointing | memory 절감 | enabled in training scripts |
| Gradient accumulation | 작은 micro batch로 effective batch 구성 | used in Kaggle/PRO 6000 scripts |
| Completion-only loss | prompt memorization 방지 | core methodology |
| Cosine LR / warmup | 안정적인 SFT schedule | used in RSP trainer |
| 4-bit / QLoRA | T4 resource feasibility 확인 | resource-constrained probe |
| Cut Cross Entropy style patch | logits materialization memory 절감 | used in Huikang-compatible notebook variants |
| MoE tied-gradient convention | Tinker-style expert LoRA behavior alignment | used in Huikang-compatible notebook variants |
| Sample packing | 짧은 sequences의 padding 낭비 감소 | 검토/일부 실험 context, current public script의 핵심 claim은 아님 |

Flash Attention, Liger Kernel 등은 30B-class training에서 중요한 optimization 후보로 검토했습니다. 다만 clean public repo의 현재 claim은 실제 코드 evidence가 있는 BF16/gradient checkpointing/completion-only SFT/pairwise preference learning 중심으로 제한합니다.

## PEFT Method Exploration

프로젝트 중 검토하거나 실험 대상으로 삼은 PEFT 계열은 다음과 같습니다.

| Method | 역할 | 이 repo에서의 claim level |
| --- | --- | --- |
| LoRA | 표준 저랭크 adapter. `ΔW = B @ A` 형태로 trainable delta를 근사 | main implementation path |
| QLoRA | 4-bit base model loading + LoRA training으로 memory 절감 | T4/limited GPU feasibility probe |
| DoRA | weight direction과 magnitude를 분리하는 LoRA 변형 | method review / candidate |
| rsLoRA | rank scaling 안정화. rank가 큰 경우의 training stability 보완 | method review / candidate |
| LoRA+ | LoRA A/B matrix에 다른 learning rate 적용 | method review / candidate |
| PiSSA | SVD 기반 initialization으로 빠른 수렴을 노리는 방식 | SVD analysis와 연결되는 reference method |
| AdaLoRA / VeRA | parameter budget을 더 줄이는 변형 | reviewed, not current public path |
| Residual LoRA | frozen existing adapter 위에 작은 residual adapter를 추가 학습 | teammate artifact, weak-domain repair experiment |
| Patch/Lambda LoRA | patch adapter를 학습한 뒤 lambda scale과 SVD merge로 rank 32 export | teammate artifact, adapter stability experiment |
| Ortho-LoRA | task-wise gradient projection으로 task interference 완화 | teammate artifact, experimental |

이 중 public clean repo에서 가장 강하게 주장할 수 있는 구현은 rank-32 LoRA, QLoRA feasibility probe, SVD 기반 adapter conversion analysis, RSP trainer입니다. Residual/Patch/Ortho-LoRA는 `team/minjaechoics/`에 보존한 teammate experiment artifact이며, DoRA/rsLoRA/LoRA+/PiSSA는 “검토한 방법론” 또는 “후속 실험 후보”로 표현합니다.

## Team Weak-Domain Repair Context

`team/minjaechoics/`의 teammate artifacts는 training methodology 측면에서 다음 insight를 제공합니다.

- `equation_numeric` aggregate를 numeric/symbolic, `hk_*`/`my_*` split으로 나눠 봐야 한다.
- symbolic branch-map failure는 arithmetic ability 문제가 아니라 rule branch selection과 symbol mapping 문제에 가깝다.
- 기존 rank-32 adapter를 직접 업데이트하면 protected domains가 흔들릴 수 있으므로 residual/patch LoRA, guarded replay, EWC, DPO, GRPO-style update가 검토되었다.
- final submission constraint 때문에 residual/patch signal은 SVD로 다시 rank 32 adapter에 합쳐야 한다.

이 내용은 RSP의 `anchor_sft`, `decision_sft`, `decision_preferences` 설계와 연결됩니다. 특히 anchor rows는 protected behavior 보존, decision/preference rows는 weak-domain branch selection 개선을 의도합니다.

## Dataset Families

### Huikang Replay Rows

Huikang snapshot에서 epoch-0 order를 복원하고, 각 problem id의 `synthetic.json`을 읽어 token/mask SFT row로 사용했습니다. 이 구조는 notebook에서 다음 방식으로 확인됩니다.

- `logprobs/index.jsonl`에서 `problem_id` order 읽기
- `tokens/<problem_id>/synthetic.json` 로드
- `tokens`, `mask` 길이 확인
- max length 8192 truncation
- `mask`가 모두 0인 row 제외

### CoT-Selected Rows

초기 CoT-selected SFT 실험에서는 generated CoT 중 final answer가 rule-based checker를 통과한 row만 사용했습니다.

이 방식의 목적:

- noisy generated reasoning 제거
- domain-specific checker로 correctness 확보
- answer-only가 아니라 reasoning trace 포함

### Auxiliary Replay Rows

Double-update notebook에서는 math replay와 E3 branch-map rows를 Huikang-style token/mask format으로 변환한 뒤 원본 stream에 interleave했습니다.

이 방식의 목적:

- broad reasoning coverage 추가
- equation/rule-selection 계열 보강
- auxiliary data가 target corpus를 완전히 압도하지 않게 mixing

### RSP Rows

Public package의 RSP dataset은 세 row family로 정리되어 있습니다.

| Row family | Count | Purpose |
| --- | ---: | --- |
| `anchor_sft` | 7,646 | broad behavior preservation |
| `decision_sft` | 2,666 | correct rule trace supervision |
| `decision_preferences` | 2,500 | chosen vs rejected rule branch learning |

## RSP Training Phases

### Phase 1: Weighted Completion-Only SFT

입력:

- `rsp_anchor_sft.jsonl`
- `rsp_decision_sft.jsonl`

핵심 동작:

- prompt에 boxed-answer instruction suffix를 보정
- completion이 최종 `\boxed{answer}`로 끝나도록 정규화
- prompt label은 `-100`으로 masking
- completion token에만 cross-entropy 적용
- row-level `sample_weight` 보존

### Phase 2: Pairwise Rule-Selection Preference Learning

입력:

- `rsp_decision_preferences.jsonl`

핵심 동작:

- 같은 prompt에 대해 `chosen` completion과 `rejected` completion을 비교
- completion-only 평균 log probability를 계산
- SimPO-style pairwise loss로 chosen branch를 선호하도록 학습
- SFT보다 낮은 learning rate와 짧은 epoch로 설정

이 phase는 “더 긴 reasoning을 쓰게 하는 것”보다 “올바른 rule branch를 고르게 하는 것”에 초점을 둡니다.

## Adapter Conversion and SVD

Huikang/Tinker 계열 adapter는 submission-compatible PEFT adapter와 key/shape가 다를 수 있습니다.

주요 이슈:

- Tinker adapter key prefix와 PEFT submission key prefix 차이
- fused expert weights를 per-expert weights로 변환해야 하는 문제
- `gate_proj` + `x_proj` fused projection을 `in_proj`로 맞추는 문제
- rank 64 형태를 rank 32 제한에 맞추기 위한 SVD compression

SVD 변환은 practical하지만 lossy입니다. 따라서 문서에서는 SVD를 “성능 향상 보장 기법”이 아니라 **adapter compatibility를 맞추기 위한 rank compression / transport experiment**로 설명합니다.

## Verification Before Training

`verify_rsp_dataset.py` checks:

- required files
- required fields
- unique IDs
- boxed answer consistency
- row family counts
- bit-manipulation decision trace constraints
- fail-closed execution flags

`verify_rsp_train_shell.py` checks:

- `SUBMISSION_ALLOWED = False`
- `EVALUATION_ALLOWED = False`
- locked target modules
- LoRA rank/alpha/dropout contract
- preference phase existence
- forbidden evaluation/submission behavior absence

After training, the same verifier can inspect adapter zip structure:

- `adapter_config.json`
- `adapter_model.safetensors`
- rank 32 LoRA A/B tensor shape
- target module set
- dropout/alpha/bias config

## What This Method Supports

현재 artifacts가 support하는 claim:

- Huikang-style SFT format을 분석하고 public package로 재구성했다.
- RSP dataset schema와 verifier가 있다.
- train-only Nemotron LoRA entrypoint가 있다.
- adapter output은 post-training structure gate로 검증할 수 있다.
- historical experiments는 context로 기록되어 있다.

## What This Method Does Not Verify

현재 artifacts만으로는 다음을 검증하지 않습니다.

- final RSP adapter가 선택되었다.
- final leaderboard score를 달성했다.
- 모든 domain에서 성능이 개선되었다.
- historical 0.85/0.86 notes가 current clean repo에서 재현된다.
- STaR full loop가 complete implementation으로 들어갔다.

이 부분은 공개 문서에서 명시적으로 claim boundary로 유지합니다.
