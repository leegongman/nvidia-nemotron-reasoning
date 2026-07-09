# Dataset Design

이 문서는 프로젝트의 데이터 설계와 provenance를 정리합니다. 핵심은 원천 데이터를 내가 만든 것처럼 주장하는 것이 아니라, **Huikang/Tong Hui Kang의 공개 데이터와 repository를 분석하고, 그 구조를 기반으로 내 학습/검증 실험용 데이터 패키지를 재구성했다**는 점입니다.

## 1. 원천 데이터와 Attribution

주요 원천은 다음 공개 자료입니다.

- Kaggle competition: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge>
- Kaggle dataset: <https://www.kaggle.com/datasets/huikang/huikang-nemotron-repository-snapshot>
- Kaggle author: <https://www.kaggle.com/huikang>
- GitHub source: <https://github.com/tonghuikang/nemotron>

Huikang dataset metadata 기준 license는 CC BY 4.0입니다. 이 프로젝트는 Huikang 원천 데이터의 제작자임을 주장하지 않습니다. 내가 수행한 작업은 다음에 가깝습니다.

- 공개 snapshot의 구조 분석
- token/mask SFT format 해석
- Kaggle notebook 기반 재현/변형 실험
- 보조 데이터셋 구성 및 mixing
- adapter training package와 verification gate 구현
- failure analysis를 RSP rule-selection dataset으로 재정의

## 2. Competition Dataset Structure

공식 대회 데이터는 rule transformation puzzle을 담은 CSV입니다. 공개 파일 기준 구조는 다음과 같습니다.

| File | Rows | Fields | 역할 |
| --- | ---: | --- | --- |
| `train.csv` | 9,500 | `id`, `prompt`, `answer` | puzzle과 ground-truth answer 제공 |
| `test.csv` | 3 sample rows | `id`, `prompt` | submission 작성용 sample. 실제 scoring 시 hidden test set으로 교체 |

`train.csv` prompt는 몇 개의 input-output examples와 풀어야 할 target instance를 포함합니다. 내가 확인한 prompt family는 다음과 같습니다.

| Domain family | Rows | 문제 성격 |
| --- | ---: | --- |
| Bit manipulation | 1,602 | 8-bit binary input/output transformation rule 추론 |
| Gravity | 1,597 | `d = 0.5*g*t^2` 형태에서 hidden gravitational constant 추론 |
| Unit conversion | 1,594 | Wonderland unit conversion rule 추론 |
| Text encryption | 1,576 | 예시 문장 쌍에서 substitution/encryption rule 복원 |
| Numeral conversion | 1,576 | 숫자를 Wonderland numeral system으로 변환 |
| Equation transformation | 1,555 | symbolic equation/string transformation rule 추론 |

이 구조 때문에 단순한 “수학 풀이”보다 **few-shot examples에서 규칙을 찾고, target instance에 같은 규칙을 적용하는 능력**이 중요합니다.

공식 evaluation은 model response에서 `\boxed{}` 안의 final answer를 우선 추출합니다. 문자열은 exact match, 수치는 relative tolerance `1e-2` 기준으로 채점되며, score는 correct ratio입니다. 최종 제출은 CSV 예측 파일이 아니라 rank 32 이하 LoRA adapter를 담은 `submission.zip`입니다.

## 3. Huikang 데이터 구조

Huikang snapshot에서 핵심적으로 확인한 학습 경로는 다음 형태입니다.

```text
nemotron-master/
└── training/
    └── sft/
        └── 04-08-16-14/
            ├── tokens/
            │   └── <problem_id>/
            │       └── synthetic.json
            └── logprobs/
                └── index.jsonl
```

`logprobs/index.jsonl`은 training order와 problem id를 연결하는 index 역할을 합니다. Kaggle notebook에서는 epoch 0 row를 읽어 원래 학습 순서를 복원했습니다.

각 `synthetic.json`은 대략 다음 정보를 포함합니다.

```json
{
  "tokens": [ ... ],
  "mask": [ ... ]
}
```

의미는 단순하지만 중요합니다.

| Field | 의미 |
| --- | --- |
| `tokens` | prompt + assistant reasoning + final answer가 합쳐진 token sequence |
| `mask=0` | prompt 영역. loss를 걸지 않음 |
| `mask=1` | assistant reasoning/completion 영역. loss를 적용 |

즉, Huikang-style SFT는 일반적인 text-only SFT가 아니라 **pre-tokenized completion-only SFT**입니다. 모델은 문제 prompt 자체를 외우는 것이 아니라, deterministic solver trace와 최종 `\boxed{answer}`를 재현하도록 학습됩니다.

## 4. Huikang 방식의 데이터 설계 핵심

Huikang 공개 writeup과 GitHub 구조에서 확인한 핵심은 다음입니다.

1. Domain별 solver trace
   - `reasoners/*.py`에서 gravity, unit conversion, numeral, cipher, equation, bit manipulation을 서로 다른 deterministic procedure로 풉니다.
   - 모델에게 “추론을 발견하라”고 맡기는 방식이 아니라, 사람이 설계한 solver trace를 모델이 재현하게 만드는 방식입니다.

2. Completion-only loss
   - prompt는 mask 0입니다.
   - reasoning trace와 final boxed answer는 mask 1입니다.
   - 이 구조는 prompt memorization보다 answer-producing trace reproduction에 가깝습니다.

3. Min-logprob 중심 분석
   - 평균 loss보다 특정 token의 낮은 logprob가 실전 실패로 이어질 수 있습니다.
   - Huikang writeup에서는 min logprob와 token-level failure inspection을 반복 개선 루프의 핵심으로 설명합니다.

4. Deterministic chain-of-thought
   - temperature 0.0 evaluation 환경에서는 다양한 답변보다 가장 가능성 높은 token sequence가 중요합니다.
   - 따라서 다양성보다 안정적인 trace 형식과 token-level reproducibility가 중요합니다.

## 5. 내가 재구성한 데이터 흐름

이 프로젝트의 데이터 실험은 Huikang 구조를 기준점으로 삼되, 여러 보조 데이터와 failure-specific rows를 추가하는 방향으로 진행되었습니다.

### 4.1 CoT-selected SFT

초기 실험에서는 LLM이 생성한 CoT 중 rule-based checker로 final answer가 맞는 샘플만 남기는 방식을 사용했습니다.

확인된 notebook:

- `leegongman/nemotron-sft-lora-with-cot-my-dataset`

주요 구조:

- prompt에 공식 boxed-answer suffix 추가
- domain별 prompt engineering 적용
- generated CoT에서 `\boxed{}` answer 추출
- type-specific checker로 정답 여부 확인
- correct CoT만 SFT row로 사용

이 실험은 “데이터 양”보다 “verified reasoning quality”가 중요하다는 방향을 확인하는 역할을 했습니다.

### 4.2 Huikang-style replay and balanced SFT

이후 실험은 Huikang token/mask corpus 구조를 더 직접적으로 따랐습니다.

확인된 notebook:

- `leegongman/sft-my-data-balace`

사용한 입력:

- `huikang/huikang-nemotron-repository-snapshot`
- `leegongman/merged-sft-data-balance`
- `metric/nemotron-3-nano-30b-a3b-bf16`

핵심은 `merged_sft_dataset/tokens`와 `merged_sft_dataset/logprobs/index.jsonl`을 Huikang 원본과 같은 방식으로 읽고, `tokens[:-1]`, `targets=tokens[1:]`, `weights=mask[1:]` 구조로 LoRA 학습을 수행했다는 점입니다.

### 4.3 Math replay auxiliary mixing

`end-to-end-finetuning-for-lb-0-83-double-update` notebook에서는 외부 math replay JSONL을 Nemotron chat template으로 tokenization하고, prompt 영역은 mask 0, assistant 영역은 mask 1로 변환했습니다.

주요 설계:

- source: `mohamedamr992/replay-math`
- output: `replay_math_tokenized.jsonl`
- target answer tokens cap: 약 2M answer tokens
- Huikang target examples 사이에 replay examples를 interleave

이 방식은 보조 reasoning coverage를 추가하되, 원본 target stream과 완전히 분리된 shuffle이 아니라 일정 간격으로 삽입하는 방식이었습니다.

### 4.4 Equation symbolic branch-map mixing

같은 double-update notebook에서는 equation symbolic branch-map row도 추가로 tokenization했습니다.

주요 설계:

- source: `equation_symbolic_sft_strict_plus_gold_guided.jsonl`
- output: `e3_branchmap_tokenized.jsonl`
- prompt/completion을 chat template으로 렌더링
- prompt는 mask 0, completion은 mask 1
- math replay 삽입 이후 다시 E3 examples를 interleave

이 실험은 equation 계열의 rule-selection failure를 보강하기 위한 보조 데이터 mixing입니다.

### 4.5 Domain-specific weighting

W1 provenance에서 equation과 bit domain에 대한 reweighting evidence가 확인됩니다.

```json
{
  "observed_domain_loss_weights": {
    "equation_numeric": 1.25,
    "bit_manipulation": 1.1
  }
}
```

문서에서는 이를 “domain-specific alpha”라고 부르기보다 **domain-specific sample/loss weighting**이라고 표현합니다. LoRA `lora_alpha`와 혼동될 수 있기 때문입니다.

### 4.6 RSP dataset

최종 public package는 RSP(Rule Selection Post-Training) 형태로 정리했습니다.

| File | 역할 |
| --- | --- |
| `rsp_anchor_sft.jsonl` | broad behavior 보존용 anchor rows |
| `rsp_decision_sft.jsonl` | correct rule trace 학습 |
| `rsp_decision_preferences.jsonl` | chosen/rejected branch preference 학습 |
| `rsp_manifest.json` | count, schema, provenance summary |

RSP의 핵심은 bit/equation failure를 “정답 생성 실패”가 아니라 **decision branch 선택 실패**로 재구성한 것입니다.

## 6. Public Repo에서 제외한 데이터

다음 파일들은 GitHub에 포함하지 않습니다.

- full Huikang snapshot copy
- generated full JSONL datasets
- `submission.zip`
- adapter checkpoint
- `.safetensors`, `.bin`, `.pt`, `.pth`
- Kaggle cache, Hugging Face cache
- raw private notebook outputs
- large replay datasets

Clean repo에는 작은 example rows만 포함합니다.

## 7. Claim Boundary

공개 문서에서 가능한 표현:

- Huikang-style dataset structure를 분석했다.
- 공개/외부 데이터 기반으로 학습 패키지를 재구성했다.
- 보조 데이터 mixing과 rule-selection dataset을 설계했다.
- train-ready adapter pipeline과 static verifier를 구현했다.

피해야 하는 표현:

- Huikang 원천 데이터를 내가 만들었다.
- 최종 adapter가 검증 완료되었다.
- final leaderboard score를 달성했다.
- STaR full loop를 완료 구현했다.
- SVD conversion이 성능을 보장한다.
