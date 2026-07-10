# equation_numeric 오답 분석 보고서

평가 결과 디렉토리: `/home/ubuntu/evaluator/results/authorized_submission_1_vllm_60pc_20260517_170748`

평가 모델: `FinetunedAdapter(Authrozied)/submission_1`

Backend: `vllm`

평가 샘플: `/home/ubuntu/evaluator/eval_60_per_category.jsonl`에서 뽑은 `equation_numeric` 60문항

## 요약

`equation_numeric`의 점수는 `43/60 = 0.7167`로, 이번 60개/category 평가에서 가장 낮았습니다.

다만 낮은 점수가 `equation_numeric` 전체에 고르게 퍼진 것은 아닙니다. 오답 17개는 모두 `symbolic_branch_*` 문제에서 나왔습니다.

| 구간 | 정답 / 전체 | 정확도 |
|---|---:|---:|
| `source_replay_*` 문제 | 26 / 26 | 1.0000 |
| `symbolic_branch_*` 문제 | 17 / 34 | 0.5000 |
| 숫자형 정답 | 37 / 41 | 0.9024 |
| 기호형 정답 | 6 / 19 | 0.3158 |
| `symbolic_branch_*` 숫자형 정답 | 12 / 16 | 0.7500 |
| `symbolic_branch_*` 기호형 정답 | 5 / 18 | 0.2778 |

핵심 결론은 이렇습니다. 모델은 일반적인 숫자 연산처럼 보이는 `source_replay_*` 문제는 거의 완벽하게 풀었지만, `symbolic_branch_*`의 기호 기반 branch-map 문제에서 크게 약했습니다. 이 문제들은 단순 사칙연산이나 두 피연산자 연산이 아니라, `input[2]`를 기준으로 분기한 뒤 특정 위치의 문자를 골라 위치별 mapping을 적용하는 구조인 경우가 많습니다. 반면 모델은 이를 숫자 연산, 문자열 이어붙이기, 역순 이어붙이기, 절댓값 차이 같은 작은 operator family로 해석하는 경향을 보였습니다.

## 주요 실패 유형

### 1. 기호형 branch-map 문제를 산술/operator 문제로 오해함

오답 중 상당수는 구두점 기호가 데이터 토큰, operator, 정답 문자로 동시에 등장합니다. 생성 답변을 보면 모델은 이런 기호들을 `A`, `B`, `C` 같은 내부 문자로 치환한 뒤, 가운데 문자를 `x`, `y`, `z` 같은 operator로 라벨링하려고 합니다.

이 휴리스틱은 `symbolic_branch_*` 기호형 문제에서 잘 맞지 않았습니다. 정답 풀이가 단순 operator 적용이 아니라 `input[2]`에 따른 분기와 위치별 symbol map을 요구하기 때문입니다.

예시: `symbolic_branch_00142`

- Target: `|[)$[`
- 정답: `|['{`
- 모델 예측: `|[)$[`
- 모델 행동: question operator를 concatenation으로 판단하고 입력과 거의 같은 문자열을 반환했습니다.
- reference 풀이: `input[2]=)` 기준으로 분기한 뒤 선택 위치에 mapping을 적용합니다.
  - `out[0] = map_0(input[0]=|) = |`
  - `out[1] = map_1(input[4]=[) = [`
  - `out[2] = map_2(input[1]=[) = '`
  - `out[3] = map_3(input[2]=)) = {`

즉 이 케이스는 단순 추출 문제가 아니라, 생성된 풀이 자체가 잘못된 문제 모델을 사용한 것입니다.

### 2. 질문 operator가 예시에 없으면 임의 fallback을 만듦

몇몇 숫자형 오답은 생성 답변 안에서 명시적으로 "question operator is not found in the examples"라고 말한 뒤, 임의 기본 연산을 적용했습니다.

| 문제 | 정답 | 예측 | 생성 답변의 fallback |
|---|---:|---:|---|
| `symbolic_branch_00414` | `53@` | `1051` | reversed operands + reversed result + `multiply+1` |
| `symbolic_branch_00403` | `99` | `53` | absolute difference |
| `symbolic_branch_00464` | `44` | `29` | absolute difference |
| `symbolic_branch_00187` | `7541` | `34` | absolute difference |

예시: `symbolic_branch_00403`

- 예시에는 operator `>`만 등장합니다.
- 질문은 `76!23`이고, operator `!`는 예시에 없습니다.
- 정답은 `99`입니다.
- 모델은 "question operator is not found in the examples"라고 판단한 뒤 `|76 - 23| = 53`을 반환했습니다.

하지만 reference response는 `input[2]=!`에 대한 hidden branch-map 규칙을 사용합니다. 따라서 이 문제도 산술적 절댓값 차이 문제가 아니었습니다.

### 3. 알 수 없는 기호 operator를 concatenation으로 처리함

기호형 문제에서는 모델이 "question operator is unknown"이라고 판단한 뒤 concatenation을 기본값으로 사용하는 경우가 많았습니다. 이것이 기호형 오답의 큰 비중을 차지합니다.

| 문제 | 정답 | 예측 | 모델 행동 |
|---|---|---|---|
| `symbolic_branch_00479` | `/}'` | <code>)&#124;''</code> | operator `+`를 unknown으로 보고 concatenation 적용 |
| `symbolic_branch_00469` | `'[` | `[&}]}` | operator `+`를 unknown으로 보고 concatenation 적용 |
| `symbolic_branch_00401` | <code>:[&#96;&#96;</code> | empty extracted prediction | operator `*`를 unknown으로 보고 잘못된 boxed output 생성 |
| `symbolic_branch_00757` | <code>&#124;)\</code> | <code>)#?&#124;</code> | operator `+`를 unknown으로 보고 concatenation 적용 |
| `symbolic_branch_00512` | `-^]` | `EB?H` | 내부 letter 표현을 기호로 되돌리지 못함 |
| `symbolic_branch_00554` | `?'@!` | `CzxAE` | 내부 code letter를 그대로 출력 |

특히 `CzxAE`, `EB?H` 같은 출력은 매우 진단적입니다. 모델이 기호를 내부 변수로 바꾼 뒤, 최종 답을 다시 원래 symbol alphabet으로 복원하지 못하고 내부 표현을 그대로 내보냈습니다.

### 4. 기호와 operator의 경계가 모호한 문제에서 parsing이 흔들림

prompt는 구두점이 매우 많은 문자열을 사용합니다. 모델은 반복적으로 "two symbol-digits, one operator, two symbol-digits" 구조를 가정했지만, 실제 문제에서는 어떤 문자가 데이터 기호이면서 동시에 operator처럼 보이기도 하고, 모델의 내부 매핑에 없는 문자도 등장합니다.

대표 예시는 다음과 같습니다.

- `symbolic_branch_00513`: 정답 `:]`, 예측 `^#\!(`. 모델은 operator를 concatenation으로 보고 입력과 비슷한 기호열을 반환했습니다.
- `symbolic_branch_00689`: 정답 <code>&#96;{</code>, 예측 <code>&#124;}</code>. 모델은 reverse concatenation류 규칙을 선택했지만 잘못 파싱한 조각에 적용했습니다.
- `symbolic_branch_00408`: 정답 `]#`, 예측 `\[%']`. 모델은 일부 예시에서 reverse concatenation을 과잉 일반화했지만, reference는 branch-map 변환이었습니다.

### 5. 답안 추출기는 부차적 위험 요소지만 주원인은 아님

evaluator의 `\boxed{...}` 추출 로직은 마지막 중괄호를 찾는 단순 휴리스틱입니다.

```python
boxed_starts = list(re.finditer(r"\\boxed\{", text))
...
last_brace = segment.rfind("}")
matches.append(segment[:last_brace] if last_brace != -1 else segment)
```

숫자형 정답에는 대체로 문제가 없지만, 정답 자체에 `{`, `}` 같은 기호가 포함되는 경우에는 취약할 수 있습니다. 이번 run에서 가장 분명한 추출 리스크는 다음 케이스입니다.

- `symbolic_branch_00401`
- 정답: <code>:[&#96;&#96;</code>
- raw output 마지막이 `\boxed{}:::]`처럼 malformed boxed text 형태였습니다.
- 추출된 prediction은 empty string이었습니다.

하지만 이 경우에도 raw output 자체가 정답 <code>:[&#96;&#96;</code>를 제대로 생성한 것은 아니었습니다. 따라서 추출기 취약성은 보조적인 문제이고, 낮은 점수의 주된 원인은 모델의 풀이 전략 실패입니다.

## 전체 오답 목록

| # | 문제 | 정답 | 예측 | 생성 답변 기반 진단 |
|---:|---|---|---|---|
| 1 | `symbolic_branch_00142` | <code>&#124;['{</code> | <code>&#124;[)$[</code> | branch-map 문제를 concatenation으로 오해하고 입력과 비슷한 문자열을 반환함 |
| 2 | `symbolic_branch_00479` | `/}'` | <code>)&#124;''</code> | question operator를 unknown으로 보고 concatenation 적용 |
| 3 | `symbolic_branch_00077` | `10` | `1` | operator `` ` ``에 대해 absolute-difference류 규칙을 선택했고, trailing zero를 놓침 |
| 4 | `symbolic_branch_00469` | `'[` | `[&}]}` | unknown symbolic operator에 concatenation을 적용했고 중괄호가 섞인 malformed output 생성 |
| 5 | `symbolic_branch_00513` | `:]` | `^#\!(` | operator를 concatenation으로 보고 입력형 기호열을 반환 |
| 6 | `symbolic_branch_00401` | <code>:[&#96;&#96;</code> | empty | 잘못된 symbolic rule과 malformed `\boxed{}...` 출력 때문에 empty로 추출됨 |
| 7 | `symbolic_branch_00484` | <code>$&#124;?</code> | `]!{?` | symbol/operator parsing 실패로 입력과 비슷한 변환열 생성 |
| 8 | `symbolic_branch_00111` | `-$'` | `/?>` | 내부 symbol mapping 실패. unknown symbol이 `?`로 남고 boxed 출력도 잘림 |
| 9 | `symbolic_branch_00689` | <code>&#96;{</code> | <code>&#124;}</code> | reverse-concatenation류 규칙을 잘못 파싱한 조각에 적용 |
| 10 | `symbolic_branch_00757` | <code>&#124;)\</code> | <code>)#?&#124;</code> | unknown operator에 concatenation 적용 |
| 11 | `symbolic_branch_00408` | `]#` | `\[%']` | 예시 일부에서 reverse concatenation을 과잉 일반화. 실제 reference는 branch-map |
| 12 | `symbolic_branch_00512` | `-^]` | `EB?H` | 내부 letter 표현을 최종 symbol로 되돌리지 못함 |
| 13 | `symbolic_branch_00414` | `53@` | `1051` | 예시에 없는 operator에 대해 `multiply+1` 산술 규칙을 임의 적용 |
| 14 | `symbolic_branch_00403` | `99` | `53` | 예시에 없는 operator에 대해 absolute difference를 기본값으로 사용 |
| 15 | `symbolic_branch_00464` | `44` | `29` | 예시에 없는 operator에 대해 absolute difference를 기본값으로 사용 |
| 16 | `symbolic_branch_00187` | `7541` | `34` | 예시에 없는 operator에 대해 absolute difference를 기본값으로 사용 |
| 17 | `symbolic_branch_00554` | `?'@!` | `CzxAE` | 내부 표현을 symbol로 복원하지 못하고 code letter를 그대로 출력 |

## 해석

이번 샘플에서는 `equation_numeric`이라는 카테고리명이 다소 오해를 부릅니다. 60문항 중 19문항은 숫자 정답이 아니라 기호형 정답입니다. 모델은 이 기호형 케이스에서 크게 약했습니다.

- 숫자형 정답: `37/41 = 90.2%`
- 기호형 정답: `6/19 = 31.6%`

생성 답변을 보면 모델의 일관된 전략은 다음과 같습니다.

1. 모든 prompt를 두 피연산자 equation으로 파싱한다.
2. 구두점 기호를 `A`, `B`, `C` 같은 내부 문자로 바꾼다.
3. concatenation, reverse concatenation, arithmetic difference, multiplication, multiply +/- 1 같은 작은 operator family를 찾는다.
4. operator가 보이지 않거나 애매하면 임의 fallback을 사용한다.

이 전략은 `source_replay_*` 산술형 문제에는 잘 맞았습니다. 하지만 `symbolic_branch_*` branch-map 문제에서는 틀립니다. 그쪽의 실제 규칙은 대체로 다음 구조에 가깝습니다.

1. `input[2]` 기준으로 branch를 고른다.
2. 특정 input position들을 선택한다.
3. position-specific symbol map을 적용한다.
4. 길이가 달라질 수 있는 output을 만든다.

## 개선 제안

1. `equation_numeric` 점수를 최소한 `source replay` vs `symbolic branch`, numeric-answer vs symbolic-answer로 나눠서 리포팅하는 것이 좋습니다. 단일 aggregate 점수 `0.7167`은 서로 다른 두 문제군을 섞어서 보여줍니다.

2. 평가 목적이 in-context arithmetic 능력 측정이라면, `symbolic_branch_*` 기호형 branch-map 문제는 필터링하거나 별도 카테고리로 분리하는 것이 좋습니다. 이 문제들은 순수 numeric equation이라 보기 어렵습니다.

3. 반대로 symbolic branch-map 능력을 평가하려는 목적이라면, `symbolic_equation_map` 또는 `equation_symbolic` 같은 별도 카테고리로 유지하는 편이 좋습니다.

4. 모델 개선을 위해서는 다음 형태의 reasoning 예시를 학습/평가 데이터에 더 넣는 것이 좋아 보입니다.
   - `input[2]` 기준 branch 선택
   - 선택된 input position mapping
   - 처음 보는 punctuation symbol 처리
   - symbolic example에서 산술 fallback을 사용하지 않는 패턴

5. symbolic-answer 평가를 더 신뢰하려면 answer extraction도 개선하는 것이 좋습니다. `segment.rfind("}")` 방식보다 brace-balanced extraction 또는 sentinel 기반 extraction이 안전합니다.

