#!/usr/bin/env python3
"""Tokenize equation_numeric patch CSV into the token/mask dataset layout.

This is the inverse-side companion to Decoding_DS/token_to_decoded.py.  It reads
decoded patch rows and writes:

  tokens/<problem_id>/synthetic.json
  logprobs/index.jsonl
  manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "equation_numeric_patch_dataset" / "patch_decoded_prompts.csv"
DEFAULT_OUTPUT_ROOT = Path("/lambdalora/output/equation_numeric_patch_dataset_tokenized")
DEFAULT_MODEL_PATH = Path(
    "/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize equation_numeric patch CSV.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Tokenize only the first N rows; 0 means all rows.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def encode_text(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=add_special_tokens)]


def parse_int_list(raw: str, column_name: str) -> list[int]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{column_name} must be a JSON list")
    return [int(x) for x in value]


def find_subsequence(tokens: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(tokens):
        return -1
    first = needle[0]
    last_start = len(tokens) - len(needle)
    for start in range(last_start + 1):
        if tokens[start] == first and tokens[start : start + len(needle)] == needle:
            return start
    return -1


def mask_from_csv(row: dict[str, str], tokens: list[int]) -> list[int] | None:
    raw = row.get("mask", "").strip()
    if not raw:
        return None
    mask = parse_int_list(raw, "mask")
    if len(mask) != len(tokens):
        return None
    return [1 if int(x) else 0 for x in mask]


def mask_from_masked_text(tokenizer: Any, row: dict[str, str], tokens: list[int]) -> list[int] | None:
    masked_text = row.get("masked_text", "")
    if not masked_text:
        return None
    masked_tokens = encode_text(tokenizer, masked_text)
    start = find_subsequence(tokens, masked_tokens)
    if start < 0:
        return None
    mask = [0] * len(tokens)
    for idx in range(start, start + len(masked_tokens)):
        mask[idx] = 1
    return mask


def mask_from_assistant_span(tokenizer: Any, text: str, token_count: int) -> list[int] | None:
    marker = "<|im_start|>assistant\n"
    start_char = text.find(marker)
    if start_char < 0:
        return None
    prefix = text[: start_char + len(marker)]
    start = len(encode_text(tokenizer, prefix))
    if start > token_count:
        return None
    return [0] * start + [1] * (token_count - start)


def build_text(row: dict[str, str]) -> str:
    text = row.get("text", "")
    if text:
        return text
    user_prompt = row.get("user_prompt", "")
    assistant_text = row.get("assistant_text") or row.get("masked_text", "")
    return (
        "<|im_start|>system\n<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}"
    )


def build_mask(tokenizer: Any, row: dict[str, str], tokens: list[int], text: str) -> list[int]:
    for builder in (
        lambda: mask_from_csv(row, tokens),
        lambda: mask_from_masked_text(tokenizer, row, tokens),
        lambda: mask_from_assistant_span(tokenizer, text, len(tokens)),
    ):
        mask = builder()
        if mask is not None:
            return mask
    raise ValueError(f"Could not recover loss mask for {row.get('problem_id', '<missing-id>')}")


def prepare_output_root(output_root: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    token_root = output_root / "tokens"
    logprobs_dir = output_root / "logprobs"
    token_root.mkdir(parents=True, exist_ok=True)
    logprobs_dir.mkdir(parents=True, exist_ok=True)
    return token_root, logprobs_dir


def row_metadata(row: dict[str, str], problem_id: str, mask: list[int], tokens: list[int]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "epoch": int(row["epoch"]) if row.get("epoch") else 0,
        "step": int(row["step"]) if row.get("step") else 0,
        "problem_id": problem_id,
        "source_problem_id": row.get("source_problem_id", ""),
        "segment": row.get("segment") or "synthetic.jsonl",
        "category": row.get("category", ""),
        "patch_source": row.get("patch_source", ""),
        "reasoning_type": row.get("reasoning_type", ""),
        "branch_pos": row.get("branch_pos", ""),
        "repeat_index": row.get("repeat_index", ""),
        "num_tokens": len(tokens),
        "num_loss_tokens": int(sum(mask)),
    }
    return meta


def tokenize_csv(args: argparse.Namespace) -> None:
    input_csv = args.input_csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer/model path not found: {model_path}")

    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=args.trust_remote_code)
    token_root, logprobs_dir = prepare_output_root(output_root, args.overwrite)
    index_path = logprobs_dir / "index.jsonl"

    count = 0
    total_tokens = 0
    total_loss_tokens = 0
    max_tokens = 0
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    with input_csv.open("r", encoding="utf-8", newline="") as f, index_path.open("w", encoding="utf-8") as index_file:
        reader = csv.DictReader(f)
        for row in reader:
            if args.limit and count >= args.limit:
                break
            problem_id = row.get("problem_id", "").strip() or f"row_{count:06d}"
            text = build_text(row)
            tokens = encode_text(tokenizer, text)
            mask = build_mask(tokenizer, row, tokens, text)
            if len(tokens) != len(mask):
                raise ValueError(f"tokens/mask length mismatch for {problem_id}: {len(tokens)} != {len(mask)}")
            if not any(mask):
                raise ValueError(f"empty loss mask for {problem_id}")

            out_dir = token_root / problem_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "synthetic.json").write_text(
                json.dumps({"tokens": tokens, "mask": mask}, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            meta = row_metadata(row, problem_id, mask, tokens)
            index_file.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n")

            count += 1
            total_tokens += len(tokens)
            total_loss_tokens += sum(mask)
            max_tokens = max(max_tokens, len(tokens))

    manifest = {
        "created_at": started_at,
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "model_path": str(model_path),
        "rows": count,
        "total_tokens": total_tokens,
        "total_loss_tokens": total_loss_tokens,
        "max_tokens": max_tokens,
        "index_path": str(index_path),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Tokenized rows: {count}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total loss tokens: {total_loss_tokens:,}")
    print(f"Max sequence length: {max_tokens:,}")
    print(f"Tokenized dataset root: {output_root}")


def main() -> None:
    tokenize_csv(parse_args())


if __name__ == "__main__":
    main()
