# Recipe Evidence

이 문서는 Kaggle full-recipe notebook의 실행 코드에서 확인되는 학습 recipe를 공개용으로 다시 정리한 것입니다. notebook 원본은 Kaggle runtime의 외부 입력 경로와 대형 산출물을 참조하므로 이 저장소에는 그대로 포함하지 않습니다.

## Input Staging

실행은 다음 입력을 Kaggle dataset으로 연결해 시작합니다.

| 입력 | 사용 목적 |
| --- | --- |
| Nemotron 30B BF16 model | base model |
| competition `train.csv` | 문제 id, prompt, answer, domain 추정 |
| token/mask corpus | 원본 reasoning SFT stream |
| `replay_math` JSONL | 보조 math reasoning stream |
| 24-row sub-replay | subtraction rule 보강 |
| local wheelhouse | `unsloth`, `trl`, `peft`, `transformers`, `bitsandbytes`, Mamba/Causal Conv 의존성 |

원본 corpus는 `tokens/<problem_id>/synthetic.json`과 `logprobs/index.jsonl`로 읽습니다. epoch 0의 order를 보존하고, 각 row를 `tokens[:-1]`, `tokens[1:]`, `mask[1:]`로 변환합니다.

Notebook 버전에 따라 Kaggle input에서 이 corpus를 직접 읽거나, 같은 구조로 미리 staging한 snapshot을 읽는 차이가 있습니다. 이 문서는 provider-specific 경로가 아니라 공통 입력 계약과 변환 순서를 기록합니다.

## Replay Conversion

`replay_math` row는 `messages`의 user/assistant 경계를 기준으로 chat template를 적용합니다.

```text
prompt tokens       -> weight 0
assistant reasoning -> weight 1
assistant answer    -> weight 1
```

전체 길이는 8,192 token으로 제한하고 assistant target token이 없는 row는 제외합니다. 해당 실행에서는 323개 replay row를 보존했고, trainable answer token 상한을 약 2M으로 두었습니다.

24개 sub-replay는 subtraction rule 후보를 직접 계산해 reasoning trace로 만든 뒤 같은 tokenizer와 mask 규칙으로 저장합니다. 이후 원본 stream에 일정 간격으로 interleave합니다.

## Model and LoRA

```text
base: NVIDIA Nemotron 3 Nano 30B A3B BF16
r=32, lora_alpha=32, lora_dropout=0.0
target: q/k/v/o_proj, in/out_proj, up/down_proj, lm_head
max_seq_length=8192
batch_size=32, micro_batch_size=4
```

MoE 모델에서 expert LoRA weight tying을 적용하고, PEFT wrapper가 `lm_head`를 누락하는 경우 수동으로 보완합니다. LoRA weight는 FP32, base model은 BF16로 cast를 검증합니다.

## Weighted Training

원본 stream에는 domain별 target-token loss multiplier를 적용합니다.

```python
DOMAIN_LOSS_WEIGHTS = {
    "equation_numeric": 1.20,
    "bit_manipulation": 1.10,
}
```

데이터 row나 prompt를 복제하는 방식이 아니라 `per_token_ce * padded_weights`로 loss를 계산합니다. 따라서 domain weighting과 LoRA alpha는 서로 다른 조정 축입니다.

## Memory and Optimization

- `bf16` autocast
- gradient checkpointing
- batch 32를 micro batch 4로 나누는 accumulation
- Cut Cross Entropy로 logits materialization을 피하는 per-token loss
- AdamW, `weight_decay=0.0`
- step에 따른 linear learning-rate decay

## Packaging

학습 후 `adapter_config.json`과 `adapter_model.safetensors`를 저장하고, 필요한 `lm_head` key namespace를 정리한 뒤 `submission.zip`으로 묶습니다. 이 저장물은 공개 저장소에 포함하지 않습니다.

## Boundary

이 문서가 증명하는 것은 해당 recipe가 코드에 구성되어 있었다는 사실입니다. 이 저장소는 최종 leaderboard 점수 재현, 모든 변형의 동일한 성능, 또는 최종 adapter 배포를 주장하지 않습니다.
