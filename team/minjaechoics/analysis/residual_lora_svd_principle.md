# Residual LoRA + SVD Rank-32 Merge Principle

## 1. Objective

현재 가장 안정적으로 동작한 LoRA는 기존 `submission_1` adapter를 직접 수정하지 않고, 그 위에 아주 작은 보정용 LoRA를 추가로 학습한 뒤 다시 단일 rank-32 adapter로 압축한 방식이다.

목표는 다음과 같다.

- 기존 `submission_1`의 강한 성능을 보존한다.
- 약한 카테고리인 `equation_numeric`에만 작은 보정 신호를 더한다.
- 대회 평가 환경의 `max_lora_rank=32` 제한을 만족하는 단일 adapter로 제출한다.

## 2. Base Structure

기존 LoRA의 추론 식은 다음과 같다.

```text
W = W_base + DeltaW_submission_1
```

이번 residual 방식은 학습 중에 다음 구조를 사용했다.

```text
W = W_base + DeltaW_submission_1 + DeltaW_residual
```

여기서 각 부분의 역할은 다음과 같다.

```text
W_base              : 원본 Nemotron base model, freeze
DeltaW_submission_1 : 기존 rank-32 LoRA adapter, freeze
DeltaW_residual     : 새로 얹은 작은 residual LoRA, trainable
```

즉 학습 중 gradient는 `DeltaW_residual`에만 흐른다. 기존 adapter 자체는 직접 업데이트하지 않는다.

## 3. Residual Adapter Training Setup

실제 residual adapter는 다음 설정으로 만들어졌다.

```text
residual rank  : 2
residual alpha : 2
learning rate  : 1e-7
steps          : 4
batch size     : 2
```

학습 대상 module은 비교적 제한했다.

```text
trainable:
  in_proj
  out_proj
  q_proj
  k_proj
  v_proj
  o_proj

excluded:
  MoE up_proj
  MoE down_proj
  lm_head
```

`up_proj/down_proj`는 MoE expert 전체에 걸려 있어 파라미터 수와 영향 범위가 매우 크다. `lm_head`도 출력 분포 전체를 흔들 수 있다. 그래서 이번 residual LoRA는 작은 projection 계열에만 얹어 catastrophic drift를 줄였다.

## 4. Data Used

학습 데이터는 evaluator의 `debug_predictions`에서 만든 teacher-trace 데이터다.

```text
/home/ubuntu/additonal_tuning/datasets/sft_debug_teacher_trace_micro_v3/prepared_teacher_trace_micro.jsonl
```

주요 구성은 다음과 같다.

```text
equation_wrong_teacher_trace
equation_correct_teacher_replay
bit_wrong_teacher_trace
bit_correct_teacher_replay
drift_anchor_teacher_trace
```

핵심은 단순히 정답만 주입하는 것이 아니라, `reference_response`에 들어 있는 정답 풀이 흐름을 target으로 사용했다는 점이다.

## 5. Why Merge

학습 중 구조는 다음과 같다.

```text
DeltaW_submission_1 rank <= 32
DeltaW_residual     rank <= 2
```

둘을 그대로 더하면 이론적으로 rank가 최대 34가 될 수 있다.

```text
rank(DeltaW_submission_1 + DeltaW_residual) <= 34
```

하지만 대회 평가 환경은 다음 제한을 가진다.

```text
max_lora_rank = 32
```

따라서 residual adapter를 그대로 추가 adapter로 제출할 수 없다. 최종 제출을 위해서는 다음처럼 하나의 rank-32 adapter로 다시 압축해야 한다.

```text
DeltaW_final ~= DeltaW_submission_1 + scale * DeltaW_residual
rank(DeltaW_final) <= 32
```

이번 제출에 사용한 scale은 다음과 같다.

```text
scale = 0.03
```

## 6. SVD Rank-32 Recompression Principle

LoRA update는 보통 다음 형태다.

```text
DeltaW = scale * B @ A
```

기존 adapter와 residual adapter를 더하면 다음과 같다.

```text
DeltaW_sum = scale_old * B_old @ A_old
           + scale_res * B_res @ A_res
```

이를 하나의 low-rank 행렬 곱으로 합치기 위해 다음처럼 concat할 수 있다.

```text
B_cat = [sqrt(scale_old) * B_old, sqrt(scale_res) * B_res]
A_cat = [sqrt(scale_old) * A_old
         sqrt(scale_res) * A_res]

DeltaW_sum = B_cat @ A_cat
```

이 상태의 rank는 최대 `32 + 2 = 34`다. 그래서 `B_cat @ A_cat`을 SVD로 다시 rank 32에 가깝게 압축한다.

실제 구현은 큰 full matrix를 직접 만들지 않고, QR/SVD trick을 사용한다.

```text
B_cat = Qb @ Rb
A_cat.T = Qa @ Ra

DeltaW_sum = Qb @ (Rb @ Ra.T) @ Qa.T
```

중간 core matrix인 `Rb @ Ra.T`는 크기가 작다. 여기에 SVD를 적용하고 상위 32개 singular component만 남긴다.

```text
core = U @ S @ Vh

B_new = Qb @ U_32 @ sqrt(S_32)
A_new = sqrt(S_32) @ Vh_32 @ Qa.T
```

최종적으로 다음 adapter가 저장된다.

```text
DeltaW_final = B_new @ A_new
rank <= 32
```

이 adapter는 대회 vLLM이 하나의 일반 LoRA adapter로 로드할 수 있다.

## 7. Why This Approach Is Safe

직접 SFT 방식은 기존 rank-32 adapter 자체를 업데이트한다.

```text
DeltaW_submission_1 -> DeltaW_submission_1'
```

이 경우 작은 step이라도 기존 능력이 바로 흔들릴 수 있다. 실제로 `rank32_direct_safe_v1`은 1 step만으로도 `bit_manipulation` 정답 여러 개를 깨뜨렸다.

반면 residual 방식은 다음 장점이 있다.

- 기존 adapter를 freeze하므로 원본 능력을 직접 덮어쓰지 않는다.
- residual delta를 별도 파일로 보관할 수 있다.
- 병합 전 `scale`을 조절해 영향력을 작게 만들 수 있다.
- 최종 제출 시에는 단일 rank-32 adapter가 된다.

다만 완전히 무해한 것은 아니다. 최종적으로는 `DeltaW_final`이 logits를 바꾸기 때문에 scale이 너무 크면 다른 카테고리도 흔들릴 수 있다.

## 8. Currently Submitted Artifact

학습된 residual adapter:

```text
/home/ubuntu/additonal_tuning/outputs/residual_lora_svd_v1/residual_adapter
```

제출한 rank-32 병합 adapter:

```text
/home/ubuntu/additonal_tuning/outputs/residual_lora_svd_v1/merged_adapter_scale_0p03
```

제출 zip:

```text
/home/ubuntu/additonal_tuning/outputs/residual_lora_svd_v1/merged_adapter_scale_0p03/submission.zip
```

평가 결과:

```text
/home/ubuntu/evaluator/results/residual_lora_svd_v1_scale_0p03_60pc/summary.json
```

local eval summary:

```text
accuracy          : 0.9547619047619048
bit_manipulation  : 0.9666666666666667
equation_numeric  : 0.7166666666666667
other categories  : 1.0
```

원본 `submission_1`과 비교했을 때:

```text
fixed     : 0
regressed : 0
changed predictions: 6, all inside equation_numeric wrong cases
```

즉 `scale=0.03`은 성능을 개선하지는 않았지만, 기존 정답을 깨지 않는 안전한 병합 지점으로 확인되었다.

## 9. Future Improvements

현재 residual adapter는 안전하게 작동했지만 오답을 정답으로 바꾸지는 못했다. 다음 개선은 재학습보다 scale sweep이 먼저다.

```text
scale candidates:
  0.01
  0.05
  0.10
```

목표는 다음 조건을 만족하는 scale을 찾는 것이다.

```text
fixed > regressed
bit_manipulation 유지
equation_numeric 43/60 이상으로 상승
나머지 카테고리 60/60 유지
```

만약 scale sweep에서도 개선이 없다면, 다음 단계는 residual 학습 데이터 자체를 더 좁히는 것이다.

- full trace 대신 `S4: APPLY target ... \boxed{}` 중심의 짧은 target 사용
- equation_numeric 오답 중 branch-position 오류만 따로 분리
- punctuation/backtick 보존 케이스만 별도 residual adapter로 학습
- residual target module을 더 좁혀 `in_proj/out_proj`만 실험
