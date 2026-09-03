# 01. Token/Mask Replay SFT

초기 핵심 분석은 학습 데이터가 일반 text JSONL이 아니라 token/mask 기반 SFT corpus라는 점이었습니다.

## Core Structure

```text
tokens/<problem_id>/synthetic.json
logprobs/index.jsonl
```

각 row는 prompt와 assistant completion이 하나의 token sequence에 들어 있고, mask로 loss 영역을 구분합니다.

- prompt: `mask=0`
- assistant reasoning/final answer: `mask=1`

## Experiment Focus

- `logprobs/index.jsonl` order를 이용한 replay
- `tokens[:-1] -> tokens[1:]` next-token objective
- max length 8192 유지
- prompt memorization을 줄이기 위한 completion-only loss

## Observations

같은 데이터를 쓰더라도 text SFT와 token/mask SFT는 학습 신호가 다릅니다. 이 프로젝트에서는 token/mask format을 기준으로 이후 실험을 정리했습니다.
