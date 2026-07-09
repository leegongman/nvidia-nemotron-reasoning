#!/usr/bin/env python3
"""Very small direct update inside the existing rank-32 adapter.

This is the conservative version of "update the existing rank-32 LoRA space".
It starts from submission_1, freezes broad/high-risk LoRA modules by default
(MoE up/down projections and lm_head), and only updates a small projection set.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

from additional_tuning_common import (
    BASE_MODEL_PATH,
    COMPETITION_MAX_LORA_RANK,
    COMPETITION_MAX_MODEL_LEN,
    COMPETITION_MAX_NUM_SEQS,
    COMPETITION_MAX_TOKENS,
    COMPETITION_GPU_MEMORY_UTILIZATION,
    COMPETITION_TEMPERATURE,
    COMPETITION_TOP_P,
    DEFAULT_TARGET_MODULES,
    PREVIOUS_EVAL_JSONL,
    SUBMISSION_1_ROOT,
    discover_adapter,
    env_bool,
    env_float,
    env_int,
    env_str,
    load_unsloth_lora_model,
    log,
    now,
    patch_cce_forward,
    reset_dir,
    save_adapter,
    str2bool,
    zip_adapter,
)
from sft_debug_teacher_trace_micro import (
    DEFAULT_DRIFT_DEBUG_JSONL,
    build_examples,
    next_batch_indices,
    prepare_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative direct rank-32 adapter SFT.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--source-debug-jsonl", default=env_str("SOURCE_DEBUG_JSONL", PREVIOUS_EVAL_JSONL))
    parser.add_argument("--drift-debug-jsonl", default=env_str("DRIFT_DEBUG_JSONL", DEFAULT_DRIFT_DEBUG_JSONL))
    parser.add_argument("--output-dir", default=env_str("DIRECT_SAFE_OUTPUT_DIR", "/home/ubuntu/additonal_tuning/outputs/sft_rank32_direct_safe"))
    parser.add_argument("--prepared-jsonl", default=env_str("DIRECT_SAFE_PREPARED_JSONL", ""))
    parser.add_argument("--resample", type=str2bool, default=env_bool("RESAMPLE", False))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))

    parser.add_argument("--anchor-ids", default=env_str("DIRECT_SAFE_ANCHOR_IDS", "hk_3302f383,hk_7283eb09,hk_c095f799-p0,my_bit_manipulation_00628,my_bit_manipulation_00933,my_equation_numeric_00239"))
    parser.add_argument("--wrong-repeat", type=int, default=env_int("DIRECT_SAFE_WRONG_REPEAT", 2))
    parser.add_argument("--bit-wrong-repeat", type=int, default=env_int("DIRECT_SAFE_BIT_WRONG_REPEAT", 2))
    parser.add_argument("--anchor-repeat", type=int, default=env_int("DIRECT_SAFE_ANCHOR_REPEAT", 2))
    parser.add_argument("--equation-replay-count", type=int, default=env_int("DIRECT_SAFE_EQUATION_REPLAY_COUNT", 12))
    parser.add_argument("--bit-replay-count", type=int, default=env_int("DIRECT_SAFE_BIT_REPLAY_COUNT", 16))
    parser.add_argument("--replay-repeat", type=int, default=env_int("DIRECT_SAFE_REPLAY_REPEAT", 1))
    parser.add_argument("--include-drift-no-boxed", type=str2bool, default=env_bool("DIRECT_SAFE_INCLUDE_DRIFT_NO_BOXED", True))
    parser.add_argument("--boxed-tail-weight", type=float, default=env_float("DIRECT_SAFE_BOXED_TAIL_WEIGHT", 1.0))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

    parser.add_argument("--max-seq-len", type=int, default=env_int("MAX_MODEL_LEN", COMPETITION_MAX_MODEL_LEN))
    parser.add_argument("--lora-rank", type=int, default=env_int("LORA_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--lora-alpha", type=int, default=env_int("LORA_ALPHA", 32))
    parser.add_argument("--load-in-4bit", type=str2bool, default=env_bool("LOAD_IN_4BIT", False))
    parser.add_argument("--gradient-checkpointing", default=env_str("GRADIENT_CHECKPOINTING", "unsloth"))

    parser.add_argument("--epochs", type=int, default=env_int("DIRECT_SAFE_EPOCHS", 1))
    parser.add_argument("--num-steps", type=int, default=env_int("DIRECT_SAFE_NUM_STEPS", 2))
    parser.add_argument("--batch-size", type=int, default=env_int("DIRECT_SAFE_BATCH_SIZE", 2))
    parser.add_argument("--micro-batch-size", type=int, default=env_int("DIRECT_SAFE_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("DIRECT_SAFE_LR", 8e-8))
    parser.add_argument("--weight-decay", type=float, default=env_float("DIRECT_SAFE_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("DIRECT_SAFE_MAX_GRAD_NORM", 0.05))
    parser.add_argument("--save-steps", type=int, default=env_int("DIRECT_SAFE_SAVE_STEPS", 1))
    parser.add_argument("--logging-steps", type=int, default=env_int("DIRECT_SAFE_LOGGING_STEPS", 1))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("DIRECT_SAFE_SHUFFLE", True))
    parser.add_argument("--trainable-name-regex", default=env_str("DIRECT_SAFE_TRAINABLE_REGEX", r"\.(in_proj|out_proj|q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\."))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def configure_trainable_params(model, pattern: str) -> list:
    regex = re.compile(pattern)
    trainable = []
    counts = Counter()
    for name, param in model.named_parameters():
        if ".lora_" not in name:
            param.requires_grad_(False)
            continue
        keep = bool(regex.search(name))
        param.requires_grad_(keep)
        if keep:
            trainable.append(param)
            for module_name in ("in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "lm_head"):
                if f".{module_name}." in name:
                    counts[module_name] += param.numel()
                    break
    log(f"Trainable direct LoRA tensors: {len(trainable)}")
    for name, count in sorted(counts.items()):
        log(f"  trainable {name}: {count:,} params")
    if not trainable:
        raise RuntimeError(f"No trainable LoRA params matched regex: {pattern}")
    return trainable


def save_run_metadata(args: argparse.Namespace, output_dir: Path, rows: list[dict], examples: list[dict], num_steps: int) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "method": "rank32_direct_safe",
        "initial_adapter": args.initial_adapter,
        "rows": len(rows),
        "examples": len(examples),
        "num_steps": num_steps,
        "trainable_name_regex": args.trainable_name_regex,
        "training": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "max_grad_norm": args.max_grad_norm,
            "boxed_tail_weight": args.boxed_tail_weight,
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
    if args.lora_rank != COMPETITION_MAX_LORA_RANK:
        raise ValueError("Direct-safe mode expects lora_rank=32 to stay in the existing adapter space.")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("--batch-size must be divisible by --micro-batch-size")

    output_dir = reset_dir(args.output_dir)
    log(f"[{now()}] output_dir={output_dir}")
    rows = prepare_rows(args)

    from transformers import AutoTokenizer

    prep_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    examples = build_examples(rows, prep_tokenizer, args.max_seq_len, args.boxed_tail_weight)
    if not examples:
        raise RuntimeError("No training examples.")

    num_steps = args.num_steps if args.num_steps > 0 else max(1, math.ceil(len(examples) / args.batch_size) * max(1, args.epochs))
    save_run_metadata(args, output_dir, rows, examples, num_steps)
    if args.dry_run:
        log("DRY_RUN=true: stopped before model load.")
        return

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
    trainable_params = configure_trainable_params(model, args.trainable_name_regex)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    rng = random.Random(args.seed)
    indices = list(range(len(examples)))
    if args.shuffle:
        rng.shuffle(indices)
    cursor = 0
    epoch = 0
    train_log: list[dict] = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, num_steps + 1):
        batch_indices, cursor, finished_epoch = next_batch_indices(indices, cursor, args.batch_size)
        if not batch_indices:
            epoch += 1
            cursor = 0
            if args.shuffle:
                random.Random(args.seed + epoch).shuffle(indices)
            batch_indices, cursor, finished_epoch = next_batch_indices(indices, cursor, args.batch_size)
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
        row = {"step": step, "loss": avg_loss, "tokens": int(total_weight_sum), "batch_size": len(batch), "epoch": epoch, "wall_sec": round(time.time() - step_t0, 3)}
        train_log.append(row)
        if step % args.logging_steps == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(f"[{now()}] step={step:04d}/{num_steps} loss={avg_loss:.5f} mem={mem:.1f}GB peak={peak:.1f}GB")
        if args.save_steps and step % args.save_steps == 0:
            save_adapter(model, tokenizer, output_dir / f"checkpoint-{step:04d}", stack)
        if finished_epoch:
            epoch += 1
            cursor = 0
            if args.shuffle:
                random.Random(args.seed + epoch).shuffle(indices)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in train_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_dir = output_dir / "final_adapter"
    save_adapter(model, tokenizer, final_dir, stack)
    if args.zip_submission:
        log(f"Submission zip: {zip_adapter(final_dir)}")


if __name__ == "__main__":
    main()
