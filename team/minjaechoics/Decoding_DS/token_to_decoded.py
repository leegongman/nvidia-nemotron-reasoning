#!/usr/bin/env python3
"""Decode tokenized SFT dataset rows into text.

Converted from token_to_decoded.ipynb with paths fixed for this workspace.

Defaults:
  dataset root : /home/ubuntu/dataset/merged_sft_dataset
  output dir   : /home/ubuntu/Decoding_DS
  tokenizer    : local Nemotron snapshot under /workspace/.hf_home

Outputs:
  prompts.jsonl        raw token/mask rows, one JSON object per problem
  decoded_prompts.csv  decoded full text and supervised/masked text
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_DATASET_ROOT = Path("/home/ubuntu/dataset/merged_sft_dataset")
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/Decoding_DS")
DEFAULT_MODEL_PATH = Path(
    "/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode tokenized SFT dataset into CSV.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompts-jsonl", default="prompts.jsonl")
    parser.add_argument("--output-csv", default="decoded_prompts.csv")
    parser.add_argument("--limit", type=int, default=0, help="Decode only the first N rows; 0 means all rows.")
    parser.add_argument("--skip-special-tokens", action="store_true")
    return parser.parse_args()


def load_index(dataset_root: Path) -> dict[str, dict]:
    index_path = dataset_root / "logprobs" / "index.jsonl"
    if not index_path.exists():
        return {}
    out: dict[str, dict] = {}
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row.get("problem_id", ""))
            if pid:
                out[pid] = row
    return out


def iter_token_rows(dataset_root: Path):
    token_root = dataset_root / "tokens"
    if not token_root.is_dir():
        raise FileNotFoundError(f"Token directory not found: {token_root}")

    index = load_index(dataset_root)
    paths = sorted(token_root.glob("*/synthetic.json"))
    if not paths:
        raise FileNotFoundError(f"No synthetic.json files found under: {token_root}")

    for path in paths:
        problem_id = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        meta = index.get(problem_id, {})
        yield {
            "problem_id": problem_id,
            "category": meta.get("category", ""),
            "segment": meta.get("segment", ""),
            "num_loss_tokens": meta.get("num_loss_tokens", ""),
            "total_loss": meta.get("total_loss", ""),
            "min_logprob": meta.get("min_logprob", ""),
            "tokens": data["tokens"],
            "mask": data["mask"],
        }


def extract_user_prompt(decoded_text: str) -> str:
    matches = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", decoded_text, flags=re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_assistant_text(decoded_text: str) -> str:
    marker = "<|im_start|>assistant"
    if marker not in decoded_text:
        return ""
    return decoded_text.split(marker, 1)[1].strip()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=False)

    prompts_path = output_dir / args.prompts_jsonl
    csv_path = output_dir / args.output_csv

    fieldnames = [
        "problem_id",
        "category",
        "segment",
        "num_loss_tokens",
        "total_loss",
        "min_logprob",
        "text",
        "masked_text",
        "user_prompt",
        "assistant_text",
        "tokens",
        "mask",
    ]

    count = 0
    with prompts_path.open("w", encoding="utf-8") as jf, csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for row in iter_token_rows(dataset_root):
            if args.limit and count >= args.limit:
                break

            tokens = [int(x) for x in row["tokens"]]
            mask = [int(x) for x in row["mask"]]
            if len(tokens) != len(mask):
                raise ValueError(f"tokens/mask length mismatch for {row['problem_id']}: {len(tokens)} != {len(mask)}")

            masked_tokens = [tok for tok, keep in zip(tokens, mask) if keep]
            text = tokenizer.decode(tokens, skip_special_tokens=args.skip_special_tokens)
            masked_text = tokenizer.decode(masked_tokens, skip_special_tokens=args.skip_special_tokens)

            raw_row = {"tokens": tokens, "mask": mask}
            jf.write(json.dumps(raw_row, ensure_ascii=False, separators=(",", ":")) + "\n")

            writer.writerow(
                {
                    "problem_id": row["problem_id"],
                    "category": row["category"],
                    "segment": row["segment"],
                    "num_loss_tokens": row["num_loss_tokens"],
                    "total_loss": row["total_loss"],
                    "min_logprob": row["min_logprob"],
                    "text": text,
                    "masked_text": masked_text,
                    "user_prompt": extract_user_prompt(text),
                    "assistant_text": extract_assistant_text(text),
                    "tokens": json.dumps(tokens, ensure_ascii=False, separators=(",", ":")),
                    "mask": json.dumps(mask, ensure_ascii=False, separators=(",", ":")),
                }
            )
            count += 1

    print(f"Decoded rows: {count}")
    print(f"Wrote JSONL: {prompts_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
