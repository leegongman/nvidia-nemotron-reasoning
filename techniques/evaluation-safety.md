# Evaluation Safety

대회에서는 adapter가 실제로 로드되고, final answer가 올바르게 추출되는지가 중요합니다.

## Checks

- `adapter_config.json` 존재
- `adapter_model.safetensors` 존재
- LoRA rank <= 32
- target module set 일치
- `\boxed{}` answer extraction
- vLLM LoRA loading compatibility
- evaluation과 submission을 training script에서 분리

## Related Code

- `scripts/eval/auto_evaluator.py`
- `scripts/eval/run_eval.sh`
- `scripts/package/verify_rsp_train_shell.py`
