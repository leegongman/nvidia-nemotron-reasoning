# Training Tricks

실험에서 사용하거나 검토한 training optimization입니다. `사용`은 notebook/code에서 확인된 항목이고, `검토`는 아이디어나 별도 실험 방향으로 남은 항목입니다.

| Technique | 역할 |
| --- | --- |
| BF16 | 사용. base model과 autocast 학습 경로 |
| LoRA FP32 cast | 사용. LoRA parameter를 FP32로 유지하고 base는 BF16로 검증 |
| Gradient checkpointing | 사용. `FastLanguageModel.get_peft_model` 설정 |
| Micro-batch accumulation | 사용. batch 32를 micro batch 4개 단위로 나눠 gradient accumulation |
| Linear LR decay | 사용. step에 따라 learning rate를 선형 감소 |
| Cut Cross Entropy | 사용. logits materialization 없이 per-token CE 계산 |
| Sample packing | 검토/별도 후보. full recipe의 주 학습 경로로 확인되지는 않음 |
| Flash Attention | 검토/후보. full recipe는 eager attention 설정을 사용 |
| Liger Kernel | 검토/후보 |
| Cosine LR + warmup | 일반적인 비교 후보. 확인된 full recipe의 스케줄은 linear decay |

clean repo에서는 실제 코드 evidence가 있는 부분을 중심으로만 claim합니다.
