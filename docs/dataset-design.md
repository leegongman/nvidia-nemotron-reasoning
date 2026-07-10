# Dataset Design

이 문서는 이 프로젝트에서 학습 데이터를 어떻게 구성했는지 설명합니다. 특정 원천 데이터 이름보다 중요한 것은, 대회 puzzle을 어떤 학습 문제로 바꾸고 어떤 row family로 나누었는지입니다.

## 1. 목표

대회 prompt는 few-shot input-output examples를 보고 숨은 transformation rule을 찾아 target instance에 적용하는 구조입니다. 따라서 dataset 설계의 목표는 단순 answer memorization이 아니라 다음 세 가지입니다.

- 모델이 deterministic reasoning trace를 안정적으로 생성하게 만들기
- weak domain에서 잘못된 rule branch를 고르는 실패를 줄이기
- adapter update가 기존에 잘 맞히던 domain을 망가뜨리지 않게 anchor를 유지하기

## 2. Competition Seed Dataset

공식 `train.csv`는 `id`, `prompt`, `answer`를 포함하는 9,500개 puzzle입니다.

| Domain family | Rows | 설계상 의미 |
| --- | ---: | --- |
| Bit manipulation | 1,602 | output bit별 boolean/bitwise rule 선택 |
| Gravity | 1,597 | examples에서 hidden constant를 추정 |
| Unit conversion | 1,594 | unit mapping과 scale 변환 추론 |
| Text encryption | 1,576 | token/word substitution rule 복원 |
| Numeral conversion | 1,576 | numeral system formatting |
| Equation transformation | 1,555 | symbolic operator, branch-map, position mapping 추론 |

이 seed dataset은 task family와 answer format을 이해하는 기준으로 사용했습니다. 실제 adapter 학습에는 이 CSV를 그대로 넣는 방식보다, prompt/completion을 분리한 token/mask SFT corpus와 보조 rows를 사용했습니다.

## 3. Token/Mask SFT Format

학습 corpus의 핵심 format은 다음입니다.

```text
tokens/<problem_id>/synthetic.json
logprobs/index.jsonl
```

각 `synthetic.json`은 tokenized sequence와 loss mask를 가집니다.

```json
{
  "tokens": [101, 102, 103],
  "mask": [0, 0, 1]
}
```

의미:

| Field | 역할 |
| --- | --- |
| `tokens` | prompt + assistant reasoning + final answer token sequence |
| `mask=0` | prompt 영역. loss 제외 |
| `mask=1` | assistant reasoning/completion 영역. loss 적용 |

Trainer에서는 다음처럼 next-token row를 만듭니다.

```text
input_ids = tokens[:-1]
targets   = tokens[1:]
weights   = mask[1:]
```

이 방식은 prompt token을 외우는 방향의 loss를 줄이고, assistant trace와 최종 `\boxed{answer}`에만 supervised signal을 주기 위한 설계입니다.

## 4. Concrete Dataset Construction

내가 구성한 데이터셋의 핵심은 원천 corpus 이름이 아니라, 각 문제를 다음 학습 단위로 재해석한 방식입니다.

### 4.1 Seed Problem Parsing

각 competition row는 먼저 아래 정보로 나누어 보았습니다.

| Extracted field | 사용 목적 |
| --- | --- |
| `problem_id` | token/mask replay order와 manifest key |
| `domain_family` | bit, gravity, unit, text, numeral, equation 계열 구분 |
| few-shot examples | hidden transformation rule 후보 추론 |
| target instance | 실제 답을 적용해야 하는 input |
| `answer` | final `\boxed{}` normalization과 checker 기준 |

이 단계의 산출물은 “새 문제를 만든 것”이 아니라, 기존 problem을 SFT/pairwise 학습에 넣을 수 있도록 domain, answer, rule-decision 단위로 구조화한 것입니다.

### 4.2 Completion Trace Construction

각 SFT row는 다음 형태를 목표로 구성했습니다.

```text
prompt:
  task examples
  target input
  instruction to provide final answer in \boxed{}

completion:
  identify rule family
  choose branch / mapping / constant
  apply rule to target
  emit final \boxed{answer}
```

중요한 점은 completion이 단순 정답 문자열이 아니라, model이 자주 틀리는 **rule choice**를 드러내는 trace라는 점입니다.

### 4.3 Token/Mask Conversion

text pair가 준비되면 Nemotron tokenizer/chat template 기준으로 token/mask row를 만들었습니다.

1. prompt-only token length를 계산합니다.
2. prompt+assistant full token sequence를 만듭니다.
3. prompt 영역은 `mask=0`으로 둡니다.
4. assistant reasoning과 final answer는 `mask=1`로 둡니다.
5. max length 8192를 넘는 row는 제외하거나 truncate 후보로 분리합니다.
6. `mask`가 모두 0인 row, boxed answer가 없는 row, answer mismatch row는 제외합니다.

이 변환 덕분에 trainer는 일반 text SFT가 아니라 completion-only next-token objective로 동작합니다.

### 4.4 Auxiliary Mixing

보조 데이터는 원본 stream을 대체하지 않고, 약한 domain을 보강하는 방식으로 섞었습니다.

| Auxiliary family | 구성 방식 | Mixing 원칙 |
| --- | --- | --- |
| CoT-selected rows | 생성된 reasoning 중 checker 통과 row만 유지 | correctness 우선, noisy CoT 제거 |
| Math replay rows | chat template 적용 후 token/mask row로 변환 | 일정 간격 interleave, token budget cap |
| Equation branch-map rows | branch key, position map, symbol map을 completion에 명시 | equation weak domain 보강 |
| Target repair rows | 실제 실패 유형과 유사한 corrected trace 구성 | narrow update, anchor와 함께 사용 |

이 설계의 목적은 “데이터를 많이 넣는 것”이 아니라, weak-domain update가 기존에 맞히던 domain을 깨뜨리지 않도록 controlled mixing을 하는 것입니다.

### 4.5 Preference Pair Construction

Preference row는 random negative가 아니라 모델이 실제로 만들 법한 wrong branch를 rejected로 둡니다.

```text
same prompt
  chosen: correct branch -> correct boxed answer
  rejected: plausible wrong branch -> wrong boxed answer
```

예를 들어 equation symbolic 문제에서는 unknown operator를 단순 산술 연산으로 fallback하는 trace, punctuation symbol을 내부 code로 바꾼 뒤 복원하지 못하는 trace, bit 문제에서는 output bit별 rule이 아니라 하나의 global rule로 처리하는 trace를 rejected 후보로 둡니다.

## 5. Dataset Families

### 5.1 Anchor SFT

Anchor rows는 이미 안정적으로 동작하는 broad behavior를 보존하기 위한 rows입니다.

구성 기준:

- 다양한 domain을 포함
- final answer가 `\boxed{}`로 끝남
- prompt 영역은 mask 0
- reasoning/completion 영역은 mask 1
- weak-domain 보조 데이터보다 더 넓은 coverage 유지

역할:

- catastrophic forgetting 방지
- bit/gravity/unit/text/numeral 등 protected behavior 유지
- residual or preference update가 너무 좁아지는 것을 방지

### 5.2 Decision SFT

Decision SFT rows는 bit/equation 계열에서 “어떤 rule branch를 골라야 하는지”를 직접 학습시키기 위한 rows입니다.

구성 기준:

- prompt에 examples와 target instance 포함
- completion에는 branch 선택 근거와 적용 절차 포함
- final answer는 반드시 `\boxed{answer}`로 정규화
- symbolic equation에서는 operator fallback이 아니라 branch-map/position-map reasoning을 드러냄

예시 설계 의도:

```text
input[2]를 branch key로 선택
선택된 branch에서 output position별 source position 확인
position-specific symbol map 적용
최종 symbol sequence를 boxed answer로 출력
```

### 5.3 Decision Preferences

Preference rows는 같은 prompt에 대해 correct branch와 plausible wrong branch를 비교하도록 구성했습니다.

```json
{
  "prompt": "...",
  "chosen": "correct rule trace ... \\boxed{answer}",
  "rejected": "plausible but wrong rule trace ... \\boxed{wrong}",
  "chosen_answer": "answer",
  "rejected_answer": "wrong"
}
```

Rejected branch는 random wrong answer가 아니라 실제 모델이 자주 만드는 실패에 가깝게 구성합니다.

- unknown operator를 absolute difference로 처리
- symbolic operator를 concatenation으로 처리
- punctuation symbol을 내부 code letter로 바꾼 뒤 복원 실패
- bit output position을 독립 rule로 보지 않고 global rule 하나로 처리

이렇게 해야 preference phase가 “정답을 외우기”보다 “틀린 branch를 피하기”를 학습합니다.

## 6. Auxiliary Data Mixing

### CoT-selected SFT

생성된 CoT를 전부 쓰지 않고, rule-based checker를 통과한 rows만 남겼습니다.

구성 절차:

1. domain-specific prompt로 CoT 생성
2. `\boxed{}`에서 final answer 추출
3. domain별 checker로 정답 여부 확인
4. correct rows만 SFT 후보로 유지

목적은 데이터 양을 늘리는 것이 아니라 noisy reasoning을 줄이는 것입니다.

### Math Replay

외부 math replay rows는 Nemotron chat template으로 렌더링한 뒤 token/mask format으로 변환했습니다.

구성 방식:

- prompt-only token ids 계산
- full prompt+assistant token ids 계산
- prompt length까지 mask 0
- assistant completion은 mask 1
- answer token budget cap 적용
- target corpus 사이에 일정 간격으로 interleave

목적은 broad reasoning coverage를 추가하되, 대회 domain-specific trace를 압도하지 않게 하는 것입니다.

### Equation Branch-Map Rows

Equation symbolic failure를 보강하기 위해 branch-map rows를 추가했습니다.

구성 방식:

- prompt/completion pair 작성
- branch key, selected positions, symbol map 적용 과정을 completion에 포함
- max length 8192 초과 row 제외
- completion-only mask 적용
- math replay와 target rows 사이에 다시 interleave

목적은 equation 문제를 산술 fallback이 아니라 rule-selection/branch-map 문제로 학습시키는 것입니다.

## 7. Domain Weighting

일부 weak domain은 sample/loss weight를 높게 주었습니다.

```json
{
  "equation_numeric": 1.25,
  "bit_manipulation": 1.1
}
```

이 값은 LoRA `lora_alpha`가 아닙니다. 문서에서는 이를 **domain-specific sample/loss weighting**으로 표현합니다.

## 8. RSP Public Dataset

최종 public package는 RSP(Rule Selection Post-Training) 형태로 정리했습니다.

| File | Rows | 역할 |
| --- | ---: | --- |
| `rsp_anchor_sft.jsonl` | 7,646 | broad behavior 보존 |
| `rsp_decision_sft.jsonl` | 2,666 | correct rule trace 학습 |
| `rsp_decision_preferences.jsonl` | 2,500 | chosen/rejected branch preference 학습 |
| `rsp_manifest.json` | - | count, schema, provenance summary |

RSP의 핵심은 bit/equation failure를 “정답 생성 실패”가 아니라 **decision branch 선택 실패**로 재구성한 것입니다.

## 9. Public Repo Boundary

GitHub에는 full data를 포함하지 않습니다.

제외 대상:

- full token/mask corpus
- generated full JSONL datasets
- replay source datasets
- `submission.zip`
- adapter checkpoint
- `.safetensors`, `.bin`, `.pt`, `.pth`
- Kaggle/Hugging Face cache
- notebook outputs

Clean repo에는 schema를 보여주는 작은 example rows만 포함합니다.

## 10. Claim Boundary

가능한 표현:

- token/mask SFT corpus 구조를 분석하고 재구성했다.
- 보조 데이터 mixing과 domain weighting을 설계했다.
- rule-selection dataset과 preference pairs를 구성했다.
- train-ready adapter pipeline과 static verifier를 구현했다.

피해야 하는 표현:

- 외부 원천 corpus 자체를 직접 제작했다.
- 최종 adapter가 검증 완료되었다.
- final leaderboard score를 달성했다.
- STaR full loop를 완료 구현했다.
- SVD conversion이 성능을 보장한다.
