#!/usr/bin/env python3
"""Evaluate every equation_numeric item with the same artifacts as auto_evaluator.

This intentionally keeps the user's requested filename spelling:
`equantion_numeric_eval.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import auto_evaluator as base


CATEGORY = "equation_numeric"
DEFAULT_TOKEN_SAMPLE = "/home/ubuntu/evaluator/equation_numeric_all.jsonl"
DEFAULT_TEXT_SAMPLE = "/home/ubuntu/evaluator/equation_numeric_all_text.jsonl"
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu/evaluator/results/"
    f"equation_numeric_all_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run competition-metric-style eval on every equation_numeric row."
    )
    parser.add_argument("--dataset-jsonl", default=os.environ.get("EVAL_DATASET", base.DEFAULT_DATASET))
    parser.add_argument("--token-sample-jsonl", default=os.environ.get("EVAL_TOKEN_SAMPLE", DEFAULT_TOKEN_SAMPLE))
    parser.add_argument("--text-sample-jsonl", default=os.environ.get("EVAL_TEXT_SAMPLE", DEFAULT_TEXT_SAMPLE))
    parser.add_argument("--output-dir", default=os.environ.get("EVAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", base.DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH", base.DEFAULT_ADAPTER))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--resample", type=base.str2bool, default=base.str2bool(os.environ.get("RESAMPLE", "false")))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sort-by-id", type=base.str2bool, default=base.str2bool(os.environ.get("SORT_BY_ID", "true")))

    parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default=os.environ.get("EVAL_BACKEND", "auto"))
    parser.add_argument("--max-lora-rank", type=int, default=int(os.environ.get("MAX_LORA_RANK", "32")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "7680")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("MAX_NUM_SEQS", "64")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "8192")))

    parser.add_argument("--cache-model", type=base.str2bool, default=base.str2bool(os.environ.get("CACHE_MODEL", "true")))
    parser.add_argument("--cache-workers", type=int, default=int(os.environ.get("CACHE_WORKERS", "16")))
    parser.add_argument("--cache-chunk-mb", type=int, default=int(os.environ.get("CACHE_CHUNK_MB", "1024")))

    parser.add_argument("--dtype", choices=["bf16", "fp16"], default=os.environ.get("DTYPE", "bf16"))
    parser.add_argument("--device-map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--load-in-4bit", type=base.str2bool, default=base.str2bool(os.environ.get("LOAD_IN_4BIT", "true")))
    parser.add_argument(
        "--print-full-generations",
        type=base.str2bool,
        default=base.str2bool(os.environ.get("PRINT_FULL_GENERATIONS", "false")),
        help="Print every raw model generation to stdout/log. Raw outputs are always saved.",
    )
    parser.add_argument(
        "--write-live-results",
        type=base.str2bool,
        default=base.str2bool(os.environ.get("WRITE_LIVE_RESULTS", "true")),
        help="Append debug_predictions.jsonl, predictions.csv, and raw_generations.txt after each sample.",
    )
    return parser.parse_args()


def prepare_equation_numeric_records(args: argparse.Namespace) -> list[dict]:
    from transformers import AutoTokenizer

    token_sample_path = Path(args.token_sample_jsonl)
    if args.resample or not token_sample_path.exists():
        rows = [
            row
            for row in base.load_eval_dataset(Path(args.dataset_jsonl))
            if str(row.get("category", "")) == CATEGORY
        ]
        if args.sort_by_id:
            rows.sort(key=lambda row: str(row.get("problem_id", "")))
        base.write_jsonl(token_sample_path, rows)
    else:
        rows = base.load_jsonl(token_sample_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    records = [base.tokenized_to_eval_record(row, tokenizer) for row in rows]
    base.write_jsonl(Path(args.text_sample_jsonl), records)

    counts = Counter(row["category"] for row in records)
    print(f"Wrote token sample: {token_sample_path}")
    print(f"Wrote text sample : {args.text_sample_jsonl}")
    print(f"target category  : {CATEGORY}")
    print(f"total examples   : {len(records)}")
    print("category counts:")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")

    if not records:
        raise RuntimeError(f"No records found for category={CATEGORY!r}")
    return records


def load_equation_numeric_records(args: argparse.Namespace) -> list[dict]:
    text_sample_path = Path(args.text_sample_jsonl)
    if args.resample or not text_sample_path.exists():
        return prepare_equation_numeric_records(args)

    records = base.load_jsonl(text_sample_path)
    if not records:
        raise RuntimeError(f"No records found in {text_sample_path}")
    non_target = sorted({str(row.get("category", "")) for row in records if row.get("category") != CATEGORY})
    if non_target:
        raise RuntimeError(f"Text sample contains non-{CATEGORY} categories: {non_target}")
    print(f"Loaded text sample: {text_sample_path}")
    print(f"target category   : {CATEGORY}")
    print(f"total examples    : {len(records)}")
    return records


def write_metadata(args: argparse.Namespace, records: list[dict]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "script": str(Path(__file__).resolve()),
        "category": CATEGORY,
        "examples": len(records),
        "dataset_jsonl": args.dataset_jsonl,
        "token_sample_jsonl": args.token_sample_jsonl,
        "text_sample_jsonl": args.text_sample_jsonl,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "backend": args.backend,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    records = load_equation_numeric_records(args)
    write_metadata(args, records)
    if args.prepare_only:
        return

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            backend = "vllm"
        except ImportError:
            backend = "transformers"
            print("vLLM is not installed; falling back to Transformers backend.")

    writer = base.LiveResultWriter(args.output_dir, enabled=args.write_live_results)

    if backend == "vllm":
        predictions = base.generate_predictions_vllm(args, records, writer)
    else:
        predictions = base.generate_predictions_transformers(args, records, writer)

    _, scored, summary = base.score_predictions(predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base.write_jsonl(output_dir / "debug_predictions.jsonl", scored)
    pd.DataFrame(scored)[["problem_id", "category", "answer", "prediction", "exact_match"]].to_csv(
        output_dir / "predictions.csv", index=False
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nMetric summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
