#!/usr/bin/env python3
"""Residual LoRA SFT from the full my_equation_numeric branch-map dataset.

The previous micro runs only nudged the 17 local mistakes.  This script uses the
larger token dataset, decodes the verified symbolic branch-map traces, repacks
them with the exact competition prompt suffix, trains only a tiny residual LoRA
on top of frozen submission_1, then SVD-merges back to one rank-32 adapter.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
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
    PREVIOUS_EVAL_JSONL,
    SUBMISSION_1_ROOT,
    discover_adapter,
    env_bool,
    env_float,
    env_int,
    env_str,
    extract_final_answer,
    extract_user_prompt,
    is_numeric_answer,
    load_jsonl,
    load_token_dataset,
    load_unsloth_lora_model,
    import_training_stack,
    log,
    now,
    patch_cce_forward,
    reset_dir,
    split_prompt_response,
    str2bool,
    verify,
    write_jsonl,
)
from sft_debug_teacher_trace_micro import build_examples, next_batch_indices
from sft_residual_lora_guarded_replay import (
    DEFAULT_FRAGILE_IDS,
    DEFAULT_OLD_CATEGORIES,
    clean_prompt,
    guard_response,
    load_debug,
    parse_csv,
)
from sft_residual_lora_svd import activate_residual_adapter, merge_residual_to_rank32, save_residual_adapter


IM_END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual LoRA SFT from full branch-map dataset.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--dataset-root", default=env_str("DATASET_ROOT", DATASET_ROOT))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--source-debug-jsonl", default=env_str("SOURCE_DEBUG_JSONL", PREVIOUS_EVAL_JSONL))
    parser.add_argument("--output-dir", default=env_str("BRANCHMAP_OUTPUT_DIR", "/home/ubuntu/additonal_tuning/outputs/residual_branchmap_dataset"))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))
    parser.add_argument("--merge-only", type=str2bool, default=env_bool("MERGE_ONLY", False))
    parser.add_argument("--residual-adapter-dir", default=env_str("RESIDUAL_ADAPTER_DIR", ""))

    parser.add_argument("--target-count", type=int, default=env_int("BRANCHMAP_TARGET_COUNT", 360))
    parser.add_argument("--symbolic-only", type=str2bool, default=env_bool("BRANCHMAP_SYMBOLIC_ONLY", False))
    parser.add_argument("--force-local-wrong-repeat", type=int, default=env_int("BRANCHMAP_FORCE_WRONG_REPEAT", 3))
    parser.add_argument("--old-categories", default=env_str("BRANCHMAP_OLD_CATEGORIES", DEFAULT_OLD_CATEGORIES))
    parser.add_argument("--guard-max-per-category", type=int, default=env_int("BRANCHMAP_GUARD_MAX_PER_CATEGORY", 10))
    parser.add_argument("--guard-equation-correct-count", type=int, default=env_int("BRANCHMAP_GUARD_EQUATION_CORRECT_COUNT", 10))
    parser.add_argument("--fragile-ids", default=env_str("BRANCHMAP_FRAGILE_IDS", DEFAULT_FRAGILE_IDS))
    parser.add_argument("--fragile-repeat", type=int, default=env_int("BRANCHMAP_FRAGILE_REPEAT", 2))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

    parser.add_argument("--max-seq-len", type=int, default=env_int("MAX_MODEL_LEN", COMPETITION_MAX_MODEL_LEN))
    parser.add_argument("--base-lora-rank", type=int, default=env_int("BASE_LORA_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--base-lora-alpha", type=int, default=env_int("BASE_LORA_ALPHA", 32))
    parser.add_argument("--residual-rank", type=int, default=env_int("RESIDUAL_RANK", 4))
    parser.add_argument("--residual-alpha", type=int, default=env_int("RESIDUAL_ALPHA", 4))
    parser.add_argument("--residual-target-modules", default=env_str("RESIDUAL_TARGET_MODULES", "in_proj,out_proj,q_proj,k_proj,v_proj,o_proj"))
    parser.add_argument("--residual-scale", type=float, default=env_float("RESIDUAL_SCALE", 0.02))
    parser.add_argument("--target-rank", type=int, default=env_int("TARGET_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--load-in-4bit", type=str2bool, default=env_bool("LOAD_IN_4BIT", False))
    parser.add_argument("--gradient-checkpointing", default=env_str("GRADIENT_CHECKPOINTING", "unsloth"))

    parser.add_argument("--num-steps", type=int, default=env_int("BRANCHMAP_NUM_STEPS", 24))
    parser.add_argument("--batch-size", type=int, default=env_int("BRANCHMAP_BATCH_SIZE", 2))
    parser.add_argument("--micro-batch-size", type=int, default=env_int("BRANCHMAP_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--guard-batch-size", type=int, default=env_int("BRANCHMAP_GUARD_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("BRANCHMAP_LR", 1.5e-7))
    parser.add_argument("--weight-decay", type=float, default=env_float("BRANCHMAP_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("BRANCHMAP_MAX_GRAD_NORM", 0.05))
    parser.add_argument("--guard-loss-weight", type=float, default=env_float("BRANCHMAP_GUARD_LOSS_WEIGHT", 0.8))
    parser.add_argument("--boxed-tail-weight", type=float, default=env_float("BRANCHMAP_BOXED_TAIL_WEIGHT", 2.0))
    parser.add_argument("--logging-steps", type=int, default=env_int("BRANCHMAP_LOGGING_STEPS", 1))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("BRANCHMAP_SHUFFLE", True))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def normalize_response(text: str) -> str:
    response = str(text).strip()
    marker = "<|im_start|>assistant"
    if marker in response:
        response = response.split(marker, 1)[1].strip()
    if response.startswith("<think>"):
        response = response[len("<think>") :].lstrip()
    if not response.endswith(IM_END):
        response = response.rstrip() + IM_END
    return response


def decoded_dataset_rows(args: argparse.Namespace, tokenizer) -> list[dict]:
    rows = []
    for row in load_token_dataset(args.dataset_root):
        if row.get("category") != "equation_numeric":
            continue
        pid = str(row["problem_id"])
        if not pid.startswith("my_equation_numeric_"):
            continue
        tokens = [int(x) for x in row["tokens"]]
        mask = [float(x) for x in row["mask"]]
        prompt_tokens, response_tokens = split_prompt_response(tokens, mask)
        prompt = extract_user_prompt(tokenizer.decode(prompt_tokens, skip_special_tokens=False))
        response = normalize_response(tokenizer.decode(response_tokens, skip_special_tokens=False))
        answer = extract_final_answer(response).strip()
        if args.symbolic_only and is_numeric_answer(answer):
            continue
        rows.append(
            {
                "problem_id": pid,
                "category": "equation_numeric",
                "prompt": clean_prompt(prompt),
                "answer": answer,
                "response": response,
                "mix_group": "branchmap_dataset_symbolic" if not is_numeric_answer(answer) else "branchmap_dataset_numeric",
            }
        )
    return rows


def build_target_rows(args: argparse.Namespace, tokenizer) -> list[dict]:
    rng = random.Random(args.seed)
    rows = decoded_dataset_rows(args, tokenizer)
    by_id = {str(row["problem_id"]): row for row in rows}

    source = load_debug(args.source_debug_jsonl)
    wrong_ids = [
        pid
        for pid, row in source.items()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 0
    ]

    forced = []
    for pid in sorted(wrong_ids):
        if pid not in by_id:
            continue
        row = dict(by_id[pid])
        row["mix_group"] = "branchmap_local_wrong_forced"
        for idx in range(max(1, args.force_local_wrong_repeat)):
            item = dict(row)
            item["repeat_index"] = idx
            forced.append(item)

    forced_ids = {str(row["problem_id"]) for row in forced}
    pool = [row for row in rows if str(row["problem_id"]) not in forced_ids]
    rng.shuffle(pool)
    take = max(0, args.target_count - len(forced)) if args.target_count > 0 else len(pool)
    selected = forced + [dict(row) for row in pool[:take]]
    rng.shuffle(selected)
    return selected


def build_guard_rows(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    source = load_debug(args.source_debug_jsonl)
    guard_rows: list[dict] = []

    def add(row: dict, group: str, repeat: int = 1) -> None:
        for idx in range(max(1, repeat)):
            guard_rows.append(
                {
                    "problem_id": str(row["problem_id"]),
                    "category": str(row.get("category", "unknown")),
                    "prompt": clean_prompt(str(row["prompt"])),
                    "answer": str(row["answer"]).strip(),
                    "mix_group": group,
                    "repeat_index": idx,
                    "response": guard_response(row),
                }
            )

    old_categories = set(parse_csv(args.old_categories))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in source.values():
        category = str(row.get("category", "unknown"))
        if category in old_categories and int(row.get("exact_match", 0)) == 1:
            buckets[category].append(row)
    for category in sorted(buckets):
        group = list(buckets[category])
        rng.shuffle(group)
        for row in group[: args.guard_max_per_category]:
            add(row, f"guard_{category}_full")

    eq_correct = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 1
    ]
    rng.shuffle(eq_correct)
    for row in eq_correct[: args.guard_equation_correct_count]:
        add(row, "guard_equation_correct_full")

    seen = {str(row["problem_id"]) for row in guard_rows}
    for pid in parse_csv(args.fragile_ids):
        row = source.get(pid)
        if row and pid not in seen and int(row.get("exact_match", 0)) == 1:
            add(row, "guard_fragile_anchor_full", args.fragile_repeat)
            seen.add(pid)

    rng.shuffle(guard_rows)
    return guard_rows


def forward_weighted_loss(torch, model, device, micro: list[dict]) -> tuple[object, object, object]:
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
    del input_ids, labels, weights, attention_mask, per_token_ce
    return loss, loss_sum, weight_sum


def write_mix_report(output_dir: Path, target_rows: list[dict], guard_rows: list[dict]) -> None:
    with (output_dir / "mix_report.txt").open("w", encoding="utf-8") as f:
        for name, rows in (("target", target_rows), ("guard", guard_rows)):
            f.write(f"{name}_rows: {len(rows)}\n")
            for key, count in sorted(Counter(str(row.get("mix_group", "")) for row in rows).items()):
                f.write(f"  {key}: {count}\n")


def save_metadata(args: argparse.Namespace, output_dir: Path, target_rows: list[dict], guard_rows: list[dict], target_examples: list[dict], guard_examples: list[dict]) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "method": "residual_branchmap_dataset_sft",
        "initial_adapter": args.initial_adapter,
        "dataset_root": args.dataset_root,
        "target_rows": len(target_rows),
        "guard_rows": len(guard_rows),
        "target_examples": len(target_examples),
        "guard_examples": len(guard_examples),
        "target_mix": dict(Counter(str(row.get("mix_group", "")) for row in target_rows)),
        "guard_mix": dict(Counter(str(row.get("mix_group", "")) for row in guard_rows)),
        "residual_rank": args.residual_rank,
        "residual_alpha": args.residual_alpha,
        "residual_scale": args.residual_scale,
        "training": {
            "num_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "guard_batch_size": args.guard_batch_size,
            "guard_loss_weight": args.guard_loss_weight,
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
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def train(args: argparse.Namespace, output_dir: Path) -> Path:
    # Import Unsloth before Transformers.  Newer Transformers can import
    # torchao, which probes CUDA during import; doing that before Unsloth's
    # patches has intermittently failed on this container.
    import_training_stack()
    from transformers import AutoTokenizer

    prep_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    target_rows = build_target_rows(args, prep_tokenizer)
    guard_rows = build_guard_rows(args)
    write_jsonl(output_dir / "target_rows.jsonl", target_rows)
    write_jsonl(output_dir / "guard_rows.jsonl", guard_rows)
    write_mix_report(output_dir, target_rows, guard_rows)
    log(f"Target rows: {len(target_rows)}")
    log(f"Guard rows: {len(guard_rows)}")

    target_examples = build_examples(target_rows, prep_tokenizer, args.max_seq_len, args.boxed_tail_weight)
    guard_examples = build_examples(guard_rows, prep_tokenizer, args.max_seq_len, 1.0)
    save_metadata(args, output_dir, target_rows, guard_rows, target_examples, guard_examples)
    if args.dry_run:
        log("DRY_RUN=true: stopped before model load.")
        return output_dir / "residual_adapter"
    if not target_examples or not guard_examples:
        raise RuntimeError("Need both target and guard examples.")

    model, tokenizer, stack = load_unsloth_lora_model(
        args.model_path,
        args.initial_adapter,
        max_seq_len=args.max_seq_len,
        lora_rank=args.base_lora_rank,
        lora_alpha=args.base_lora_alpha,
        target_modules=DEFAULT_TARGET_MODULES,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    activate_residual_adapter(model, stack, args)
    torch = stack["torch"]
    patch_cce_forward(model, stack)
    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    rng = random.Random(args.seed)
    target_indices = list(range(len(target_examples)))
    guard_indices = list(range(len(guard_examples)))
    if args.shuffle:
        rng.shuffle(target_indices)
        rng.shuffle(guard_indices)
    target_cursor = 0
    guard_cursor = 0
    train_log: list[dict] = []

    for step in range(1, args.num_steps + 1):
        batch_indices, target_cursor, target_done = next_batch_indices(target_indices, target_cursor, args.batch_size)
        guard_batch_indices, guard_cursor, guard_done = next_batch_indices(guard_indices, guard_cursor, args.guard_batch_size)
        if not batch_indices:
            target_cursor = 0
            random.Random(args.seed + step).shuffle(target_indices)
            batch_indices, target_cursor, target_done = next_batch_indices(target_indices, target_cursor, args.batch_size)
        if not guard_batch_indices:
            guard_cursor = 0
            random.Random(args.seed + 1000 + step).shuffle(guard_indices)
            guard_batch_indices, guard_cursor, guard_done = next_batch_indices(guard_indices, guard_cursor, args.guard_batch_size)

        batch = [target_examples[i] for i in batch_indices]
        guard_batch = [guard_examples[i] for i in guard_batch_indices]
        optimizer.zero_grad(set_to_none=True)
        accum = math.ceil(len(batch) / args.micro_batch_size)
        step_task = 0.0
        step_guard = 0.0
        step_tokens = 0.0
        t0 = time.time()

        for mb_start in range(0, len(batch), args.micro_batch_size):
            micro = batch[mb_start : mb_start + args.micro_batch_size]
            task_loss, _, task_weight = forward_weighted_loss(torch, model, device, micro)
            guard_loss, _, _ = forward_weighted_loss(torch, model, device, guard_batch)
            loss = task_loss + args.guard_loss_weight * guard_loss
            (loss / accum).backward()
            step_task += float(task_loss.detach().cpu())
            step_guard += float(guard_loss.detach().cpu())
            step_tokens += float(task_weight.detach().cpu())
            del task_loss, guard_loss, loss, task_weight

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        row = {
            "step": step,
            "task_loss": step_task / max(accum, 1),
            "guard_loss": step_guard / max(accum, 1),
            "guard_weight": args.guard_loss_weight,
            "tokens": int(step_tokens),
            "wall_sec": round(time.time() - t0, 3),
        }
        train_log.append(row)
        if step % args.logging_steps == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(f"[{now()}] step={step:04d}/{args.num_steps} task={row['task_loss']:.5f} guard={row['guard_loss']:.5f} mem={mem:.1f}GB peak={peak:.1f}GB")
        if target_done:
            target_cursor = 0
            random.Random(args.seed + step).shuffle(target_indices)
        if guard_done:
            guard_cursor = 0
            random.Random(args.seed + 1000 + step).shuffle(guard_indices)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in train_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return save_residual_adapter(model, output_dir / "residual_adapter", stack, args)


def main() -> None:
    args = parse_args()
    if args.base_lora_rank != COMPETITION_MAX_LORA_RANK or args.target_rank != COMPETITION_MAX_LORA_RANK:
        raise ValueError("base_lora_rank and target_rank must be 32 for submission compatibility.")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("--batch-size must be divisible by --micro-batch-size")
    output_dir = reset_dir(args.output_dir)
    log(f"[{now()}] output_dir={output_dir}")
    if args.merge_only:
        if not args.residual_adapter_dir:
            raise ValueError("--merge-only requires --residual-adapter-dir")
        residual_dir = Path(args.residual_adapter_dir)
    else:
        residual_dir = train(args, output_dir)
        if args.dry_run:
            return
    merged_name = f"merged_adapter_scale_{str(args.residual_scale).replace('.', 'p')}"
    merge_residual_to_rank32(args.initial_adapter, residual_dir, output_dir / merged_name, args.target_rank, args.residual_scale, args.zip_submission)


if __name__ == "__main__":
    main()
