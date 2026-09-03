# Dataset

이 프로젝트의 데이터 설계는 "새 데이터를 많이 추가"하는 것보다, 기존 reasoning trace를 같은 token/mask 계약으로 재생하고 약한 rule family에 보조 신호를 삽입하는 데 초점을 두었습니다.

외부에서 받은 원본 데이터의 저작권·배포 범위를 이 저장소가 확장하지는 않습니다. 공개 저장소에는 원본 corpus를 포함하지 않고, 입력 형식과 변환 규칙만 설명합니다.

## Source and Attribution

이 프로젝트의 핵심 token/mask source corpus는 [Tong Huikang의 공개 Nemotron repository](https://github.com/tonghuikang/nemotron)에 기록된 Progress Prize submission pipeline에서 생성된 corpus를 기반으로 사용했습니다. 해당 repository의 공개 README와 코드가 설명하는 생성·검증 흐름을 분석했고, 그 corpus를 Kaggle snapshot에서 replay한 뒤 보조 데이터와 RSP 형태로 확장했습니다.

이 저장소는 해당 원본 데이터셋을 직접 처음부터 제작했다고 주장하지 않습니다. 원본 corpus, 전체 token 파일, 생성 로그와 checkpoint는 공개 저장소에 포함하지 않으며, 공개 범위는 구조 설명·작은 schema sample·우리의 변환/검증 코드로 제한합니다.

참고한 원본 구현은 [`reasoning.py`](https://github.com/tonghuikang/nemotron/blob/master/reasoning.py), [`augmentation.py`](https://github.com/tonghuikang/nemotron/blob/master/augmentation.py), [`corpus.py`](https://github.com/tonghuikang/nemotron/blob/master/corpus.py)입니다.

## Original Corpus Construction

원본 repository에서 확인되는 데이터 구성은 다음 단계로 나뉩니다.

1. `train.csv`의 prompt/answer와 문제 index를 category별로 정리합니다.
2. `reasoning.py`가 gravity, unit conversion, cipher, bit manipulation, numeral, equation numeric, cryptarithm 등 category에 맞는 deterministic reasoner를 선택합니다.
3. reasoner가 예시에서 규칙을 추론하고 target에 적용한 reasoning trace를 생성합니다.
4. 생성된 trace에서 `\\boxed{}` answer를 추출하고, binary/string은 엄격 비교하며 숫자는 tolerance 기반으로 검증합니다.
5. 정답 검증을 통과한 reasoning은 학습 completion으로 사용하고, 규칙을 찾지 못한 문제는 investigation 상태로 분리합니다.
6. `augmentation.py`는 spelling, concatenation, splitting, matching, lstrip 계열 augmenter가 만든 `[category]`, `[prompt]`, `[completion]` 레코드를 별도 보조 데이터로 생성합니다.

이 설계의 핵심은 정답만 수집하는 것이 아니라, **문제의 hidden rule을 찾아가는 reasoning trace를 category별 solver로 생성하고 검증한 뒤 학습 corpus로 변환**하는 데 있습니다.

## Tokenized Corpus Layout

`corpus.py`는 reasoning trace와 final answer를 completion으로 묶습니다.

```text
completion = reasoning trace + </think> + \\boxed{answer} + <|im_end|>
prompt     = chat template(problem prompt + boxed-answer instruction)
mask       = 0 for prompt tokens, 1 for completion tokens
```

공개 원본 코드에서 확인되는 산출물은 `corpus/<problem_id>/synthetic.jsonl` segment file과 `corpus.jsonl` metadata index입니다. 각 index row에는 category, masked/unmasked token count, total token count, answer, inclusion 여부가 기록되며, token limit은 8,192로 설정되어 있습니다.

## Snapshot Used In My Training

내 Kaggle 학습 recipe에서는 이 corpus의 tokenized/export snapshot을 다음 경로로 읽었습니다.

```text
tokens/<problem_id>/synthetic.json
logprobs/index.jsonl
```

이 snapshot은 공개 원본 코드의 `corpus/<problem_id>/synthetic.jsonl` 및 `corpus.jsonl`과 파일명·상위 디렉터리는 다르지만, prompt/completion token과 mask를 분리하고 문제별 metadata/order를 보존한다는 설계 목적은 같습니다. `logprobs/index.jsonl`은 notebook에서 epoch 0 training order와 문제별 logprob/loss 분석 기준으로 사용했습니다.

내가 구현한 부분은 이 snapshot의 epoch 0 순서를 replay하고 중복 problem ID를 제거한 뒤, 동일한 token/mask 계약으로 `replay_math`, subtraction sub-replay, equation branch-map rows와 preference pairs를 섞어 학습·검증 가능한 public package로 재구성한 것입니다.

## Visualization Resources

원본 repository는 [nemotron.huikang.dev](https://nemotron.huikang.dev)에서 Base, Synthetic, Corpus, Training, Metrics 화면을 제공합니다. 이 화면은 문제별 prompt와 parsed rule, reasoning trace, token mask, training logprob, category별 metric을 확인하는 참고 자료입니다. 본 저장소에는 해당 서비스의 데이터나 frontend를 복제하지 않고 원본 링크만 제공합니다.

## Seed Data

공식 `train.csv`는 9,500개 puzzle로 구성되어 있고, 각 row는 `id`, `prompt`, `answer`를 포함합니다. 노트북에서는 이 파일을 문제 식별자와 domain 추정의 기준으로 사용했습니다.

이 데이터는 다음을 파악하는 기준으로 사용했습니다.

- domain family
- answer format
- examples와 target 구조
- hidden rule 유형
- `\boxed{}` answer extraction 조건

## Source Token/Mask Corpus

학습 입력은 일반 JSONL completion이 아니라 다음 두 계층으로 구성된 tokenized corpus였습니다.

```text
tokens/<problem_id>/synthetic.json
logprobs/index.jsonl
```

- `synthetic.json`: `tokens`와 같은 길이의 `mask`를 보유합니다.
- `logprobs/index.jsonl`: epoch 0의 problem order와 문제별 loss/logprob 분석에 사용되는 index입니다.
- 노트북은 epoch 0에서 중복 problem id를 제거한 순서를 replay하고, `MAX_SEQ_LEN=8192`에서 자릅니다.

이 corpus의 설계 배경은 domain별 deterministic solver가 만든 reasoning을 assistant completion으로 저장하는 것입니다. reasoning 생성기는 gravity, unit conversion, cipher, bit manipulation, numeral, equation numeric, cryptarithm처럼 domain별로 분리되고, 정답은 `\boxed{}`에서 추출해 검증합니다.

## Token/Mask SFT

핵심 학습 format은 text-only SFT가 아니라 token/mask 구조입니다.

```json
{
  "tokens": ["prompt tokens + assistant reasoning + final answer"],
  "mask": ["0 for prompt, 1 for assistant completion"]
}
```

학습에서는 다음처럼 사용합니다.

```text
input_ids = tokens[:-1]
targets   = tokens[1:]
weights   = mask[1:]
```

이 방식의 목적은 prompt를 외우는 것이 아니라 assistant reasoning trace와 final answer에 loss를 주는 것입니다.

노트북의 실제 변환도 같은 규칙을 따릅니다.

1. assistant message를 포함한 chat template를 tokenizer로 렌더링합니다.
2. prompt 길이 이전에는 `0`, reasoning과 final content에는 `1`을 부여합니다.
3. 8,192 token을 넘는 row는 잘라내고, assistant loss token이 없는 row는 제외합니다.
4. `tokens[:-1]`, `tokens[1:]`, `mask[1:]`로 next-token loss와 weighted loss를 계산합니다.

## Auxiliary Data

추가로 구성하거나 섞어 본 데이터는 다음과 같습니다.

| 데이터 | 목적 |
| --- | --- |
| CoT-selected rows | generated CoT 중 checker를 통과한 row만 사용 |
| Math replay rows | general reasoning coverage 보강 |
| Equation branch-map rows | symbolic equation failure 보강 |
| Target repair rows | 실제 failure mode와 비슷한 corrected trace 구성 |
| Preference pairs | correct branch vs plausible wrong branch 학습 |

실제 recipe에 기록된 보조 데이터 흐름은 다음과 같습니다.

| 단계 | 노트북에서 한 일 |
| --- | --- |
| `replay_math` | message 기반 math reasoning을 동일한 chat template와 token/mask 형식으로 변환 |
| Math replay mix | answer loss token 약 2M을 상한으로 두고 원본 examples 사이에 일정 간격으로 interleave |
| Sub-replay | subtraction rule 후보를 생성해 24개 tokenized row로 만들고 별도 간격으로 interleave |
| Domain weighting | `equation_numeric` target token loss에 `1.20`, `bit_manipulation`에 `1.10` multiplier 적용 |

여기서 multiplier는 데이터 row를 복제하는 비율이 아니라, target token의 loss weight입니다. 따라서 원본 order와 prompt는 유지하면서 약한 domain의 gradient 신호만 조정합니다.

## RSP Dataset

최종 public package는 RSP(Rule Selection Post-Training) 형태로 정리했습니다.

| Family | 역할 |
| --- | --- |
| `anchor_sft` | 기존 behavior 보존 |
| `decision_sft` | 올바른 rule trace 학습 |
| `decision_preferences` | correct branch와 wrong branch 비교 |

실제 full dataset과 tokenized replay file은 GitHub에 포함하지 않고, `examples/`에 schema preview만 둡니다. recipe의 입력 경로는 Kaggle runtime에 연결된 데이터셋을 가리키며, 공개 저장소에서 그대로 실행되는 로컬 경로가 아닙니다.
