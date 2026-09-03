# PEFT Methods

## Methods Confirmed In The Primary Training Path

| Method | 역할 |
| --- | --- |
| LoRA | rank-32 adapter 학습의 기본 방법 |
| QLoRA | 4-bit 로딩 가능성을 확인한 별도 feasibility probe |

## Methods Found In Team And Follow-up Artifacts

다음 항목은 `team/minjaechoics/`에 보존한 팀 단위 또는 후속 실험 코드에서 확인됩니다. 기본 `scripts/train/` 경로의 단일 최종 recipe와 동일한 성숙도의 구현으로 묶지 않습니다.

| Method | 확인 위치 |
| --- | --- |
| Residual/Patch LoRA | `team/minjaechoics/additonal_tuning/`, `lambdalora/` |
| DPO-style preference tuning | `team/minjaechoics/additonal_tuning/sft_residual_lora_dpo.py` |
| EWC-constrained residual tuning | `team/minjaechoics/additonal_tuning/sft_residual_lora_ewc.py` |
| Ortho-LoRA | `team/minjaechoics/Ortho_LoRA/` |
| GRPO-style update | `team/minjaechoics/additonal_tuning/grpo_symbolic_fix.py` |

## Methods Reviewed During Design

| Method | 해석 |
| --- | --- |
| DoRA | LoRA delta를 direction/magnitude로 분리 |
| rsLoRA | rank scaling 안정화 |
| LoRA+ | LoRA A/B matrix에 다른 learning rate 적용 |
| PiSSA | SVD 기반 LoRA initialization |
| AdaLoRA / VeRA | parameter budget 절감형 변형 |

현재 public repo의 구현 중심은 LoRA, token/mask SFT, RSP trainer입니다.
