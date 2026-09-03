# Adapter Merge and SVD

adapter merge/SVD 실험의 목적은 여러 adapter signal을 제출 가능한 rank-32 LoRA 구조에 맞추는 것이었습니다.

## Methods Covered

- adapter delta merge
- lambda/scale sweep
- residual LoRA
- patch LoRA
- SVD rank compression
- module-wise merge
- adapter structure validation

## Why It Matters

대회 제출은 rank 32 이하 LoRA adapter입니다. 따라서 실험 중 얻은 signal이 더 큰 rank나 다른 namespace에 있으면, 최종 제출 구조에 맞게 변환해야 합니다.

## Cautions

SVD compression은 lossy입니다. 따라서 이 문서에서는 성능 보장 기법이 아니라 compatibility/transport technique으로 설명합니다.
