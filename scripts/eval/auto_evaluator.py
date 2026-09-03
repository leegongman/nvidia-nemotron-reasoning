#!/usr/bin/env python3
"""Metric-style evaluator for local Nemotron LoRA experiments.

It samples tokenized rows per category, decodes prompt/answer pairs from
tokens+mask, generates answers, then scores with the same boxed-answer
extraction and verification rules as the NVIDIA metric.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import multiprocessing
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODEL_PATH = (
    "/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
DEFAULT_DATASET = "/dataset/ortho_lora/bit_rest.jsonl"
DEFAULT_TOKEN_SAMPLE = "/home/ubuntu/evaluator/eval_20_per_category.jsonl"
DEFAULT_TEXT_SAMPLE = "/home/ubuntu/evaluator/eval_20_per_category_text.jsonl"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/evaluator/results"
DEFAULT_ADAPTER = "/home/ubuntu/Experiment_Output/final_adapter"


class ParticipantVisibleError(Exception):
    pass


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sample_per_category(rows: list[dict[str, Any]], per_category: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("category", "unknown"))].append(row)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for category in sorted(buckets):
        group = list(buckets[category])
        rng.shuffle(group)
        sampled.extend(group[: min(per_category, len(group))])
    return sampled


def extract_final_answer(text: str | None) -> str:
    r"""Extract final answer from response, matching the competition metric."""
    if text is None:
        return "NOT_FOUND"

    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches = []
    for i, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return (non_empty[-1] if non_empty else matches[-1].strip())

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return found[-1].strip()

    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    if found:
        return found[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def clean_answer(answer: str) -> str:
    answer = re.sub(r"<\|[^>]+?\|>", "", str(answer))
    return answer.strip()


def verify(stored_answer: str, predicted: str) -> bool:
    """Competition metric verification logic."""
    stored_answer = clean_answer(stored_answer)
    predicted = clean_answer(predicted)

    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def cache_model(
    path: str | Path,
    exts: tuple[str, ...] = (".bin", ".pt", ".safetensors"),
    num_workers: int | None = None,
    chunk_mb: int = 256,
) -> int:
    """Pre-read model weight files into OS page cache, as in the metric."""

    def warmup_file(fpath: Path) -> tuple[Path, int]:
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        try:
            with fpath.open("rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    total += len(data)
        except Exception as exc:
            print(f"Error reading {fpath}: {exc}")
        return fpath, total

    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file() and str(p).endswith(exts)) if path.is_dir() else []
    if not files and path.exists() and path.is_file():
        files = [path]
    if not files:
        print(f"No model files found to cache at: {path}")
        return 0

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    t0 = time.time()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, num_bytes = fut.result()
            total_bytes += num_bytes
            print(f"[{i}/{len(files)}] cached {fpath.name}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed if elapsed > 0 else 0.0
    print(f"[cache_model] total read ~= {gb:.2f} GB")
    print(f"[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s")
    return total_bytes


def extract_user_prompt(decoded_prompt: str) -> str:
    matches = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", decoded_prompt, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return re.sub(r"<\|[^>]+?\|>", "", decoded_prompt).strip()


def tokenized_to_eval_record(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    tokens = [int(x) for x in row["tokens"]]
    mask = [float(x) for x in row["mask"]]
    if len(tokens) != len(mask):
        raise ValueError(f"tokens/mask length mismatch for {row.get('problem_id')}")

    split = next((i for i, value in enumerate(mask) if float(value) > 0), len(mask))
    decoded_prompt = tokenizer.decode(tokens[:split], skip_special_tokens=False)
    decoded_response = tokenizer.decode(tokens[split:], skip_special_tokens=False)

    answer = clean_answer(extract_final_answer(decoded_response))
    return {
        "id": str(row.get("problem_id", "")),
        "problem_id": str(row.get("problem_id", "")),
        "category": str(row.get("category", "unknown")),
        "task": str(row.get("task", "")),
        "prompt": extract_user_prompt(decoded_prompt),
        "answer": answer,
        "reference_response": decoded_response,
    }


def prepare_eval_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    token_sample_path = Path(args.token_sample_jsonl)
    if args.resample or not token_sample_path.exists():
        rows = load_jsonl(Path(args.dataset_jsonl))
        sampled = sample_per_category(rows, per_category=args.per_category, seed=args.seed)
        write_jsonl(token_sample_path, sampled)
    else:
        sampled = load_jsonl(token_sample_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    records = [tokenized_to_eval_record(row, tokenizer) for row in sampled]
    write_jsonl(Path(args.text_sample_jsonl), records)

    counts = Counter(row["category"] for row in records)
    print(f"Wrote token sample: {token_sample_path}")
    print(f"Wrote text sample : {args.text_sample_jsonl}")
    print("category counts:")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    return records


def load_eval_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    text_sample_path = Path(args.text_sample_jsonl)
    if args.resample or not text_sample_path.exists():
        return prepare_eval_samples(args)
    return load_jsonl(text_sample_path)


def discover_adapter(adapter_path: str) -> str | None:
    if not adapter_path:
        return None
    path = Path(adapter_path).expanduser()
    if (path / "adapter_config.json").exists():
        return str(path)
    matches = glob.glob(str(path / "**" / "adapter_config.json"), recursive=True)
    if not matches:
        return None
    return str(Path(matches[0]).parent)


def build_prompts(tokenizer: Any, records: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    suffix = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
    for row in records:
        user_content = str(row["prompt"]) + suffix
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)
    return prompts


def score_prediction(row: dict[str, Any], raw_text: str) -> dict[str, Any]:
    pred = clean_answer(extract_final_answer(raw_text))
    exact = verify(str(row["answer"]), pred)
    return {**row, "raw_output": raw_text, "prediction": pred, "exact_match": int(exact)}


class LiveResultWriter:
    """Write per-sample eval artifacts while generation is still running."""

    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.jsonl_path = self.output_dir / "debug_predictions.jsonl"
        self.csv_path = self.output_dir / "predictions.csv"
        self.raw_path = self.output_dir / "raw_generations.txt"
        self.csv_fields = ["problem_id", "category", "answer", "prediction", "exact_match"]

        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text("", encoding="utf-8")
        self.raw_path.write_text("", encoding="utf-8")
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()

    def write(self, idx: int, total: int, row: dict[str, Any], running_acc: float) -> None:
        if not self.enabled:
            return

        append_jsonl(self.jsonl_path, row)
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writerow({field: row.get(field, "") for field in self.csv_fields})

        with self.raw_path.open("a", encoding="utf-8") as f:
            f.write(f"\n===== GENERATION {idx:03d}/{total:03d} =====\n")
            f.write(f"problem_id: {row['problem_id']}\n")
            f.write(f"category: {row['category']}\n")
            f.write(f"answer: {row['answer']}\n")
            f.write(f"prediction: {row['prediction']}\n")
            f.write(f"match: {row['exact_match']}\n")
            f.write(f"running_acc: {running_acc:.6f}\n")
            f.write("----- raw_output -----\n")
            f.write(str(row["raw_output"]))
            f.write("\n----- end_raw_output -----\n")


def emit_prediction(
    idx: int,
    total: int,
    row: dict[str, Any],
    correct_so_far: int,
    writer: LiveResultWriter,
    print_full_generation: bool,
) -> None:
    running_acc = correct_so_far / max(1, idx)
    print(
        f"\n[{idx:03d}/{total:03d}] {row['category']} {row['problem_id']}\n"
        f"answer={row['answer']}\n"
        f"prediction={row['prediction']}\n"
        f"match={row['exact_match']} running_acc={running_acc:.4f}",
        flush=True,
    )
    if print_full_generation:
        print("----- raw_output start -----", flush=True)
        print(str(row["raw_output"]), flush=True)
        print("----- raw_output end -----", flush=True)
    writer.write(idx, total, row, running_acc)


def generate_predictions_vllm(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    writer: LiveResultWriter,
) -> list[dict[str, Any]]:
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this Python environment. Install vllm or run with --backend transformers."
        ) from exc

    if args.cache_model:
        cache_model(args.model_path, num_workers=args.cache_workers, chunk_mb=args.cache_chunk_mb)

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ptxas = Path("/tmp/triton/backends/nvidia/bin/ptxas")
    if ptxas.exists():
        os.environ.setdefault("TRITON_PTXAS_PATH", str(ptxas))

    adapter = discover_adapter(args.adapter_path)
    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=1,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=adapter is not None,
        max_lora_rank=args.max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    prompts = build_prompts(llm.get_tokenizer(), records)
    lora_request = LoRARequest("adapter", 1, adapter) if adapter else None
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)

    predictions: list[dict[str, Any]] = []
    correct_so_far = 0
    total = len(records)
    for idx, (row, output) in enumerate(zip(records, outputs), start=1):
        raw_text = output.outputs[0].text
        scored = score_prediction(row, raw_text)
        correct_so_far += int(scored["exact_match"])
        emit_prediction(idx, total, scored, correct_so_far, writer, args.print_full_generations)
        predictions.append(scored)
    return predictions


def generate_predictions_transformers(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    writer: LiveResultWriter,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        "trust_remote_code": False,
        "device_map": args.device_map,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)

    adapter = discover_adapter(args.adapter_path)
    if adapter:
        from peft import PeftModel

        print(f"Loading adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    else:
        print("Adapter not found; evaluating base model only.")

    model.eval()
    prompts = build_prompts(tokenizer, records)
    predictions: list[dict[str, Any]] = []
    correct_so_far = 0
    total = len(records)
    for idx, (row, prompt) in enumerate(zip(records, prompts), start=1):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_model_len)
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        scored = score_prediction(row, raw_text)
        correct_so_far += int(scored["exact_match"])
        emit_prediction(idx, total, scored, correct_so_far, writer, args.print_full_generations)
        predictions.append(scored)
    return predictions


def score_predictions(predictions: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in predictions:
        exact = verify(str(row["answer"]), str(row["prediction"]))
        scored.append({**row, "exact_match": int(exact)})

    accuracy = sum(row["exact_match"] for row in scored) / max(1, len(scored))
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row["category"]) for row in scored}):
        rows = [row for row in scored if str(row["category"]) == category]
        by_category[category] = {
            "examples": len(rows),
            "accuracy": sum(row["exact_match"] for row in rows) / max(1, len(rows)),
        }

    summary = {"accuracy": accuracy, "examples": len(scored), "by_category": by_category}
    return accuracy, scored, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run competition-metric-style eval on sampled tokenized data.")
    parser.add_argument("--dataset-jsonl", default=os.environ.get("EVAL_DATASET", DEFAULT_DATASET))
    parser.add_argument("--token-sample-jsonl", default=os.environ.get("EVAL_TOKEN_SAMPLE", DEFAULT_TOKEN_SAMPLE))
    parser.add_argument("--text-sample-jsonl", default=os.environ.get("EVAL_TEXT_SAMPLE", DEFAULT_TEXT_SAMPLE))
    parser.add_argument("--output-dir", default=os.environ.get("EVAL_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH", DEFAULT_ADAPTER))
    parser.add_argument("--per-category", type=int, default=int(os.environ.get("PER_CATEGORY", "20")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--resample", type=str2bool, default=str2bool(os.environ.get("RESAMPLE", "false")))
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default=os.environ.get("EVAL_BACKEND", "auto"))
    parser.add_argument("--max-lora-rank", type=int, default=int(os.environ.get("MAX_LORA_RANK", "32")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "7680")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("MAX_NUM_SEQS", "64")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "8192")))

    parser.add_argument("--cache-model", type=str2bool, default=str2bool(os.environ.get("CACHE_MODEL", "true")))
    parser.add_argument("--cache-workers", type=int, default=int(os.environ.get("CACHE_WORKERS", "16")))
    parser.add_argument("--cache-chunk-mb", type=int, default=int(os.environ.get("CACHE_CHUNK_MB", "1024")))

    parser.add_argument("--dtype", choices=["bf16", "fp16"], default=os.environ.get("DTYPE", "bf16"))
    parser.add_argument("--device-map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--load-in-4bit", type=str2bool, default=str2bool(os.environ.get("LOAD_IN_4BIT", "true")))
    parser.add_argument(
        "--print-full-generations",
        type=str2bool,
        default=str2bool(os.environ.get("PRINT_FULL_GENERATIONS", "true")),
        help="Print every raw model generation to stdout/log, in addition to saving raw_generations.txt.",
    )
    parser.add_argument(
        "--write-live-results",
        type=str2bool,
        default=str2bool(os.environ.get("WRITE_LIVE_RESULTS", "true")),
        help="Append debug_predictions.jsonl, predictions.csv, and raw_generations.txt after each sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_eval_records(args)
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

    writer = LiveResultWriter(args.output_dir, enabled=args.write_live_results)

    if backend == "vllm":
        predictions = generate_predictions_vllm(args, records, writer)
    else:
        predictions = generate_predictions_transformers(args, records, writer)

    _, scored, summary = score_predictions(predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "debug_predictions.jsonl", scored)
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
