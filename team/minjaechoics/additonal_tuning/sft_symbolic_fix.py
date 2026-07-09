#!/usr/bin/env python3
"""Guarded direct SFT for the equation_numeric symbolic failure mode.

This script intentionally avoids SFTTrainer. It:
  - loads the original token/mask SFT rows from merged_sft_dataset,
  - oversamples my_equation_numeric symbolic/previous-wrong rows,
  - mixes in equation_numeric numeric/hk rows and other-category replay,
  - starts from the existing submission_1 LoRA adapter,
  - applies a small low-LR update using Unsloth + cut-cross-entropy,
  - saves a normal adapter directory with adapter_config.json for vLLM eval.

Default mix:
  50% targeted equation_numeric symbolic/previous-wrong rows
  30% submission_1-correct equation_numeric replay rows
  20% bit_manipulation replay rows, including the two observed drift anchors
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

from additional_tuning_common import (
    BASE_MODEL_PATH,
    COMPETITION_GPU_MEMORY_UTILIZATION,
    COMPETITION_MAX_LORA_RANK,
    COMPETITION_MAX_MODEL_LEN,
    COMPETITION_MAX_NUM_SEQS,
    COMPETITION_MAX_TOKENS,
    COMPETITION_TEMPERATURE,
    COMPETITION_TOP_P,
    DATASET_ROOT,
    DEFAULT_TARGET_MODULES,
    MixConfig,
    SUBMISSION_1_ROOT,
    annotate_rows,
    build_guarded_mix,
    discover_adapter,
    env_bool,
    env_float,
    env_int,
    env_path,
    env_str,
    load_token_dataset,
    load_unsloth_lora_model,
    log,
    make_sft_example,
    now,
    patch_cce_forward,
    reset_dir,
    save_adapter,
    save_mix_report,
    str2bool,
    write_jsonl,
    zip_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct guarded SFT for equation_numeric symbolic repair.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--dataset-root", default=env_str("DATASET_ROOT", DATASET_ROOT))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument(
        "--output-dir",
        default=str(
            env_path(
                "SFT_OUTPUT_DIR",
                f"/home/ubuntu/additonal_tuning/outputs/sft_symbolic_fix_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
            )
        ),
    )
    parser.add_argument("--prepared-jsonl", default=env_str("SFT_PREPARED_JSONL", ""))
    parser.add_argument("--resample", type=str2bool, default=env_bool("RESAMPLE", True))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))

    parser.add_argument("--total-examples", type=int, default=env_int("SFT_TOTAL_EXAMPLES", 900))
    parser.add_argument("--hard-ratio", type=float, default=env_float("SFT_HARD_RATIO", 0.50))
    parser.add_argument("--equation-replay-ratio", type=float, default=env_float("SFT_EQUATION_REPLAY_RATIO", 0.30))
    parser.add_argument("--bit-replay-ratio", type=float, default=env_float("SFT_BIT_REPLAY_RATIO", 0.20))
    parser.add_argument("--other-replay-ratio", type=float, default=env_float("SFT_OTHER_REPLAY_RATIO", 0.00))
    parser.add_argument(
        "--correct-equation-replay-only",
        type=str2bool,
        default=env_bool("SFT_CORRECT_EQUATION_REPLAY_ONLY", True),
    )
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

    parser.add_argument("--max-seq-len", type=int, default=env_int("MAX_MODEL_LEN", COMPETITION_MAX_MODEL_LEN))
    parser.add_argument("--lora-rank", type=int, default=env_int("LORA_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--lora-alpha", type=int, default=env_int("LORA_ALPHA", 32))
    parser.add_argument("--load-in-4bit", type=str2bool, default=env_bool("LOAD_IN_4BIT", False))
    parser.add_argument("--gradient-checkpointing", default=env_str("GRADIENT_CHECKPOINTING", "unsloth"))

    parser.add_argument("--num-steps", type=int, default=env_int("SFT_NUM_STEPS", 60))
    parser.add_argument("--batch-size", type=int, default=env_int("SFT_BATCH_SIZE", 8))
    parser.add_argument("--micro-batch-size", type=int, default=env_int("SFT_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("SFT_LR", 3e-6))
    parser.add_argument("--weight-decay", type=float, default=env_float("SFT_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("MAX_GRAD_NORM", 0.5))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("SFT_SHUFFLE", True))
    parser.add_argument("--logging-steps", type=int, default=env_int("LOGGING_STEPS", 1))
    parser.add_argument("--save-steps", type=int, default=env_int("SAVE_STEPS", 20))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def prepare_rows(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir)
    prepared = Path(args.prepared_jsonl) if args.prepared_jsonl else output_dir / "prepared_sft_mix.jsonl"
    if prepared.exists() and not args.resample:
        rows = []
        with prepared.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        save_mix_report(output_dir / "mix_report.txt", rows)
        log(f"Loaded prepared SFT mix: {prepared} ({len(rows)} rows)")
        return rows

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    all_rows = load_token_dataset(args.dataset_root)
    annotated = annotate_rows(all_rows, tokenizer)
    mix_cfg = MixConfig(
        total_examples=args.total_examples,
        hard_ratio=args.hard_ratio,
        equation_replay_ratio=args.equation_replay_ratio,
        bit_replay_ratio=args.bit_replay_ratio,
        other_replay_ratio=args.other_replay_ratio,
        seed=args.seed,
        include_previous_wrong=True,
        correct_equation_replay_only=args.correct_equation_replay_only,
    )
    rows = build_guarded_mix(annotated, mix_cfg)
    write_jsonl(prepared, rows)
    save_mix_report(output_dir / "mix_report.txt", rows)
    log(f"Wrote prepared SFT mix: {prepared} ({len(rows)} rows)")
    return rows


def build_examples(rows: list[dict], max_seq_len: int) -> list[dict]:
    examples = []
    skipped = 0
    for row in rows:
        ex = make_sft_example(row, max_seq_len)
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
    log(f"SFT examples: {len(examples)} (skipped={skipped})")
    groups = {}
    for ex in examples:
        groups.setdefault(ex.get("mix_group", ""), 0)
        groups[ex.get("mix_group", "")] += 1
    for group, count in sorted(groups.items()):
        log(f"  {group}: {count}")
    return examples


def save_metadata(args: argparse.Namespace, output_dir: Path, examples: list[dict]) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "purpose": "guarded SFT repair for equation_numeric symbolic branch-map failures",
        "examples": len(examples),
        "initial_adapter": args.initial_adapter,
        "model_path": args.model_path,
        "dataset_root": args.dataset_root,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "learning_rate": args.learning_rate,
            "hard_ratio": args.hard_ratio,
            "equation_replay_ratio": args.equation_replay_ratio,
            "bit_replay_ratio": args.bit_replay_ratio,
            "other_replay_ratio": args.other_replay_ratio,
            "correct_equation_replay_only": args.correct_equation_replay_only,
            "max_seq_len": args.max_seq_len,
            "load_in_4bit": args.load_in_4bit,
            "lora_rank": args.lora_rank,
        },
        "competition_eval_envelope": {
            "max_lora_rank": COMPETITION_MAX_LORA_RANK,
            "max_tokens": COMPETITION_MAX_TOKENS,
            "top_p": COMPETITION_TOP_P,
            "temperature": COMPETITION_TEMPERATURE,
            "max_num_seqs": COMPETITION_MAX_NUM_SEQS,
            "gpu_memory_utilization": COMPETITION_GPU_MEMORY_UTILIZATION,
            "max_model_len": COMPETITION_MAX_MODEL_LEN,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.lora_rank > COMPETITION_MAX_LORA_RANK:
        raise ValueError(f"lora_rank={args.lora_rank} exceeds competition max_lora_rank={COMPETITION_MAX_LORA_RANK}")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("--batch-size must be divisible by --micro-batch-size")

    random.seed(args.seed)
    output_dir = reset_dir(args.output_dir)
    log(f"[{now()}] output_dir={output_dir}")
    log(f"initial_adapter={args.initial_adapter}")

    rows = prepare_rows(args)
    examples = build_examples(rows, args.max_seq_len)
    save_metadata(args, output_dir, examples)
    if args.dry_run:
        log("DRY_RUN=true: stopped before model load.")
        return
    if not examples:
        raise RuntimeError("No SFT examples.")

    model, tokenizer, stack = load_unsloth_lora_model(
        args.model_path,
        args.initial_adapter,
        max_seq_len=args.max_seq_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=DEFAULT_TARGET_MODULES,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    torch = stack["torch"]
    patch_cce_forward(model, stack)

    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    indices = list(range(len(examples)))
    cursor = 0
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)

    train_log: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.num_steps + 1):
        if cursor + args.batch_size > len(indices):
            cursor = 0
            if args.shuffle:
                random.Random(args.seed + step).shuffle(indices)
        batch_indices = indices[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = [examples[i] for i in batch_indices]
        accum = math.ceil(len(batch) / args.micro_batch_size)
        total_loss_sum = 0.0
        total_weight_sum = 0.0
        step_t0 = time.time()

        for mb_start in range(0, len(batch), args.micro_batch_size):
            micro = batch[mb_start : mb_start + args.micro_batch_size]
            max_len = max(len(ex["tokens"]) for ex in micro)
            input_ids = torch.zeros(len(micro), max_len, dtype=torch.long, device=device)
            labels = torch.zeros(len(micro), max_len, dtype=torch.long, device=device)
            weights = torch.zeros(len(micro), max_len, dtype=torch.float32, device=device)
            attention_mask = torch.zeros(len(micro), max_len, dtype=torch.long, device=device)

            for i, ex in enumerate(micro):
                seq_len = len(ex["tokens"])
                input_ids[i, :seq_len] = torch.tensor(ex["tokens"], dtype=torch.long, device=device)
                labels[i, :seq_len] = torch.tensor(ex["targets"], dtype=torch.long, device=device)
                weights[i, :seq_len] = torch.tensor(ex["weights"], dtype=torch.float32, device=device)
                attention_mask[i, :seq_len] = 1

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                per_token_ce = model._cached_per_token_ce
                loss_sum = (per_token_ce * weights).sum()
                weight_sum = weights.sum()
                loss = loss_sum / weight_sum.clamp_min(1.0)
            (loss / accum).backward()
            total_loss_sum += float(loss_sum.detach().cpu())
            total_weight_sum += float(weight_sum.detach().cpu())
            del input_ids, labels, weights, attention_mask, per_token_ce, loss_sum, weight_sum, loss

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss_sum / max(total_weight_sum, 1.0)
        row = {
            "step": step,
            "loss": avg_loss,
            "tokens": int(total_weight_sum),
            "wall_sec": round(time.time() - step_t0, 3),
        }
        train_log.append(row)
        if step % args.logging_steps == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(f"[{now()}] step={step:04d}/{args.num_steps} loss={avg_loss:.5f} mem={mem:.1f}GB peak={peak:.1f}GB")

        if args.save_steps and step % args.save_steps == 0:
            ckpt_dir = output_dir / f"checkpoint-{step:04d}"
            save_adapter(model, tokenizer, ckpt_dir, stack)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in train_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_dir = output_dir / "final_adapter"
    save_adapter(model, tokenizer, final_dir, stack)
    if args.zip_submission:
        zip_path = zip_adapter(final_dir)
        log(f"Submission zip: {zip_path}")


if __name__ == "__main__":
    main()
