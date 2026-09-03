# Reproducibility

이 repo는 최종 leaderboard 점수를 재현하는 package가 아니라, 실험 구조와 train-ready pipeline을 재현 가능하게 정리한 package입니다.

## Reproducible Here

- 코드 구조 확인
- 작은 example dataset schema 확인
- RSP dataset builder/verifier 실행
- train shell static verification
- trainer dry-run path 확인
- Kaggle/Vast payload 생성

## Requires External Artifacts

- Nemotron 30B base model
- full token/mask corpus
- full auxiliary dataset
- GPU runtime
- final adapter/checkpoint
- leaderboard scoring environment

## Commands

`examples/rsp_dataset_sample/`는 schema preview라서 count gate는 실패하는 것이 정상입니다. full verification은 private/full dataset을 `data/rsp_dataset`에 stage한 뒤 실행합니다.

```bash
python scripts/data/verify_rsp_dataset.py \
  --dataset-dir data/rsp_dataset \
  --json-output data/rsp_dataset/rsp_verification.json
```

Full training requires private/full dataset artifacts:

```bash
python scripts/train/rsp_train_tokenmask_compatible.py \
  --dataset-dir data/rsp_dataset \
  --model /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --dry-run
```

## Boundary

현재 repo만으로 확인할 수 있는 것은 implementation structure입니다. 최종 점수, 최종 adapter, hidden test 성능은 별도 evidence가 필요합니다.
