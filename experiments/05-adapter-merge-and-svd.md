# 05. Adapter Merge and SVD

여러 adapter signal을 합치거나, submission-compatible rank-32 adapter로 맞추기 위해 SVD compression을 실험했습니다.

## Problem Addressed

- adapter key namespace 차이
- Tinker-style adapter와 PEFT adapter 구조 차이
- fused projection과 per-module projection 차이
- rank 32 제한
- `adapter_config.json`, `adapter_model.safetensors` 구조 검증

## Role of SVD

SVD는 성능 향상을 보장하는 기법이 아니라, 큰 delta나 merge된 delta를 rank 32 constraint에 맞추기 위한 compression/transport 방법으로 사용했습니다.

## Related Files

- `techniques/adapter-merge-svd.md`
- `scripts/package/verify_rsp_train_shell.py`
