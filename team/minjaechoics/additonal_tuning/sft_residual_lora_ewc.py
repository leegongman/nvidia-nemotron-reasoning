#!/usr/bin/env python3
"""Residual LoRA with EWC guard loss and apply-only equation targets.

This is the "push equation_numeric harder, but protect old behavior" variant:

  - freeze base model and submission_1 adapter,
  - train only a tiny residual LoRA,
  - estimate diagonal Fisher on compact old-category guard answers,
  - train equation_numeric apply-only targets with EWC penalty,
  - SVD-merge residual into a single rank-32 adapter for vLLM submission.
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
    competition_prompt,
    discover_adapter,
    env_bool,
    env_float,
    env_int,
    env_str,
    extract_final_answer,
    load_jsonl,
    load_unsloth_lora_model,
    log,
    now,
    patch_cce_forward,
    reset_dir,
    str2bool,
    verify,
    write_jsonl,
)
from sft_debug_teacher_trace_micro import DEFAULT_DRIFT_DEBUG_JSONL, build_examples
from sft_residual_lora_svd import (
    activate_residual_adapter,
    merge_residual_to_rank32,
    parse_csv,
    save_residual_adapter,
)


IM_END = "<|im_end|>"
DEFAULT_OLD_CATEGORIES = "bit_manipulation,cipher,cryptarithm,gravity,numeral,unit_conversion"
DEFAULT_FRAGILE_IDS = "hk_3302f383,hk_7283eb09,hk_c095f799-p0,my_equation_numeric_00239"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual LoRA + EWC + apply-only equation SFT.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--source-debug-jsonl", default=env_str("SOURCE_DEBUG_JSONL", PREVIOUS_EVAL_JSONL))
    parser.add_argument("--drift-debug-jsonl", default=env_str("DRIFT_DEBUG_JSONL", DEFAULT_DRIFT_DEBUG_JSONL))
    parser.add_argument("--output-dir", default=env_str("EWC_OUTPUT_DIR", "/home/ubuntu/additonal_tuning/outputs/residual_lora_ewc"))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))
    parser.add_argument("--merge-only", type=str2bool, default=env_bool("MERGE_ONLY", False))
    parser.add_argument("--residual-adapter-dir", default=env_str("RESIDUAL_ADAPTER_DIR", ""))

    parser.add_argument("--wrong-repeat", type=int, default=env_int("EWC_WRONG_REPEAT", 4))
    parser.add_argument("--equation-replay-count", type=int, default=env_int("EWC_EQUATION_REPLAY_COUNT", 12))
    parser.add_argument("--equation-replay-repeat", type=int, default=env_int("EWC_EQUATION_REPLAY_REPEAT", 1))
    parser.add_argument("--fragile-ids", default=env_str("EWC_FRAGILE_IDS", DEFAULT_FRAGILE_IDS))
    parser.add_argument("--fragile-repeat", type=int, default=env_int("EWC_FRAGILE_REPEAT", 2))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

    parser.add_argument("--old-categories", default=env_str("EWC_OLD_CATEGORIES", DEFAULT_OLD_CATEGORIES))
    parser.add_argument("--fisher-max-per-category", type=int, default=env_int("EWC_FISHER_MAX_PER_CATEGORY", 6))
    parser.add_argument("--fisher-equation-correct-count", type=int, default=env_int("EWC_FISHER_EQUATION_CORRECT_COUNT", 8))
    parser.add_argument("--fisher-batch-size", type=int, default=env_int("EWC_FISHER_BATCH_SIZE", 1))
    parser.add_argument("--ewc-lambda", type=float, default=env_float("EWC_LAMBDA", 250.0))
    parser.add_argument("--normalize-fisher", type=str2bool, default=env_bool("EWC_NORMALIZE_FISHER", True))

    parser.add_argument("--max-seq-len", type=int, default=env_int("MAX_MODEL_LEN", COMPETITION_MAX_MODEL_LEN))
    parser.add_argument("--base-lora-rank", type=int, default=env_int("BASE_LORA_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--base-lora-alpha", type=int, default=env_int("BASE_LORA_ALPHA", 32))
    parser.add_argument("--residual-rank", type=int, default=env_int("RESIDUAL_RANK", 2))
    parser.add_argument("--residual-alpha", type=int, default=env_int("RESIDUAL_ALPHA", 2))
    parser.add_argument("--residual-target-modules", default=env_str("RESIDUAL_TARGET_MODULES", "in_proj,out_proj,q_proj,k_proj,v_proj,o_proj"))
    parser.add_argument("--residual-scale", type=float, default=env_float("RESIDUAL_SCALE", 0.05))
    parser.add_argument("--target-rank", type=int, default=env_int("TARGET_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--load-in-4bit", type=str2bool, default=env_bool("LOAD_IN_4BIT", False))
    parser.add_argument("--gradient-checkpointing", default=env_str("GRADIENT_CHECKPOINTING", "unsloth"))

    parser.add_argument("--num-steps", type=int, default=env_int("EWC_NUM_STEPS", 8))
    parser.add_argument("--batch-size", type=int, default=env_int("EWC_BATCH_SIZE", 2))
    parser.add_argument("--micro-batch-size", type=int, default=env_int("EWC_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("EWC_LR", 2e-7))
    parser.add_argument("--weight-decay", type=float, default=env_float("EWC_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("EWC_MAX_GRAD_NORM", 0.05))
    parser.add_argument("--boxed-tail-weight", type=float, default=env_float("EWC_BOXED_TAIL_WEIGHT", 2.0))
    parser.add_argument("--logging-steps", type=int, default=env_int("EWC_LOGGING_STEPS", 1))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("EWC_SHUFFLE", True))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def clean_prompt(prompt: str) -> str:
    return re.sub(r"<\|[^>]+?\|>", "", str(prompt)).strip()


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


def verified_reference(row: dict) -> str | None:
    answer = str(row.get("answer", "")).strip()
    candidate = row.get("reference_response")
    if candidate and verify(answer, extract_final_answer(str(candidate))):
        return normalize_response(str(candidate))
    raw = row.get("raw_output")
    if raw and int(row.get("exact_match", 0)) == 1 and verify(answer, extract_final_answer(str(raw))):
        return normalize_response(str(raw))
    return None


def apply_only_response(row: dict) -> str:
    answer = str(row["answer"]).strip()
    response = verified_reference(row)
    if response:
        match = re.search(r"\bS4:\s*APPLY\b", response)
        if match:
            return response[match.start() :].strip()
    return f"S4: APPLY target\nresult = {answer}\n</think>\n\\boxed{{{answer}}}{IM_END}"


def compact_answer_response(row: dict) -> str:
    answer = str(row["answer"]).strip()
    return f"Final answer: \\boxed{{{answer}}}{IM_END}"


def load_debug(path: str | Path) -> dict[str, dict]:
    rows = {}
    for row in load_jsonl(path):
        pid = str(row.get("problem_id", ""))
        if pid:
            rows[pid] = row
    return rows


def add_repeated(out: list[dict], row: dict, group: str, response: str, repeat: int) -> None:
    for idx in range(max(0, repeat)):
        out.append(
            {
                "problem_id": str(row["problem_id"]),
                "category": str(row.get("category", "unknown")),
                "prompt": clean_prompt(str(row["prompt"])),
                "answer": str(row["answer"]).strip(),
                "mix_group": group,
                "repeat_index": idx,
                "response": response,
            }
        )


def build_train_rows(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    source = load_debug(args.source_debug_jsonl)
    rows: list[dict] = []

    wrong_equation = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 0
    ]
    wrong_equation.sort(key=lambda row: str(row.get("problem_id", "")))
    for row in wrong_equation:
        add_repeated(rows, row, "equation_wrong_apply_only", apply_only_response(row), args.wrong_repeat)

    used = {str(row["problem_id"]) for row in rows}
    fragile_ids = [x.strip() for x in args.fragile_ids.split(",") if x.strip()]
    for pid in fragile_ids:
        row = source.get(pid)
        if not row or str(row.get("problem_id", "")) in used:
            continue
        if row.get("category") == "equation_numeric":
            response = apply_only_response(row)
        else:
            response = compact_answer_response(row)
        add_repeated(rows, row, "fragile_anchor_apply_or_compact", response, args.fragile_repeat)
        used.add(pid)

    eq_correct = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric"
        and int(row.get("exact_match", 0)) == 1
        and str(row.get("problem_id", "")) not in used
    ]
    for row in rng.sample(eq_correct, min(args.equation_replay_count, len(eq_correct))):
        add_repeated(rows, row, "equation_correct_apply_replay", apply_only_response(row), args.equation_replay_repeat)

    rng.shuffle(rows)
    return rows


def build_guard_rows(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed + 1009)
    source = load_debug(args.source_debug_jsonl)
    buckets: dict[str, list[dict]] = defaultdict(list)
    old_categories = set(parse_csv(args.old_categories))
    for row in source.values():
        category = str(row.get("category", "unknown"))
        if category in old_categories and int(row.get("exact_match", 0)) == 1:
            buckets[category].append(row)

    rows: list[dict] = []
    for category in sorted(buckets):
        group = list(buckets[category])
        rng.shuffle(group)
        for row in group[: args.fisher_max_per_category]:
            add_repeated(rows, row, f"fisher_guard_{category}", compact_answer_response(row), 1)

    eq_correct = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 1
    ]
    rng.shuffle(eq_correct)
    for row in eq_correct[: args.fisher_equation_correct_count]:
        add_repeated(rows, row, "fisher_guard_equation_correct", compact_answer_response(row), 1)

    fragile_ids = [x.strip() for x in args.fragile_ids.split(",") if x.strip()]
    existing = {str(row["problem_id"]) for row in rows}
    for pid in fragile_ids:
        row = source.get(pid)
        if row and int(row.get("exact_match", 0)) == 1 and pid not in existing:
            add_repeated(rows, row, "fisher_guard_fragile_anchor", compact_answer_response(row), 1)
            existing.add(pid)
    return rows


def write_mix_report(path: Path, rows: list[dict], guard_rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"train_rows: {len(rows)}\n")
        f.write(f"guard_rows: {len(guard_rows)}\n")
        for name, group_rows in (("train", rows), ("guard", guard_rows)):
            f.write(f"{name}_by_group:\n")
            for key, count in sorted(Counter(str(row.get("mix_group", "")) for row in group_rows).items()):
                f.write(f"  {key}: {count}\n")
            f.write(f"{name}_by_category:\n")
            for key, count in sorted(Counter(str(row.get("category", "")) for row in group_rows).items()):
                f.write(f"  {key}: {count}\n")


def ewc_penalty(torch, trainable_named: list[tuple[str, object]], fisher: dict[str, object], theta_star: dict[str, object]) -> object:
    penalty = None
    total = 0
    for name, param in trainable_named:
        if name not in fisher:
            continue
        value = (fisher[name] * (param.float() - theta_star[name]).pow(2)).sum()
        penalty = value if penalty is None else penalty + value
        total += param.numel()
    if penalty is None:
        device = trainable_named[0][1].device
        return torch.zeros((), dtype=torch.float32, device=device)
    return penalty / max(total, 1)


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


def estimate_fisher(torch, model, device, guard_examples: list[dict], trainable_named: list[tuple[str, object]], batch_size: int, normalize: bool) -> tuple[dict, dict, dict]:
    theta_star = {name: param.detach().clone().float() for name, param in trainable_named}
    fisher = {name: torch.zeros_like(param, dtype=torch.float32, device=param.device) for name, param in trainable_named}
    stats = {"examples": len(guard_examples), "batches": 0, "raw_mean": 0.0}
    if not guard_examples:
        return fisher, theta_star, stats

    model.zero_grad(set_to_none=True)
    for start in range(0, len(guard_examples), batch_size):
        batch = guard_examples[start : start + batch_size]
        loss, _, _ = forward_weighted_loss(torch, model, device, batch)
        loss.backward()
        for name, param in trainable_named:
            if param.grad is not None:
                fisher[name].add_(param.grad.detach().float().pow(2))
        model.zero_grad(set_to_none=True)
        stats["batches"] += 1
        del loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name in fisher:
        fisher[name].div_(max(stats["batches"], 1))
    total_sum = sum(float(value.sum().detach().cpu()) for value in fisher.values())
    total_numel = sum(value.numel() for value in fisher.values())
    raw_mean = total_sum / max(total_numel, 1)
    stats["raw_mean"] = raw_mean
    if normalize and raw_mean > 0:
        for name in fisher:
            fisher[name].div_(raw_mean)
        stats["normalized_mean"] = 1.0
    else:
        stats["normalized_mean"] = raw_mean
    return fisher, theta_star, stats


def save_metadata(args: argparse.Namespace, output_dir: Path, train_rows: list[dict], guard_rows: list[dict], train_examples: list[dict], guard_examples: list[dict], fisher_stats: dict | None) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "method": "residual_lora_ewc_apply_only",
        "initial_adapter": args.initial_adapter,
        "train_rows": len(train_rows),
        "guard_rows": len(guard_rows),
        "train_examples": len(train_examples),
        "guard_examples": len(guard_examples),
        "fisher_stats": fisher_stats,
        "residual_rank": args.residual_rank,
        "residual_alpha": args.residual_alpha,
        "residual_scale": args.residual_scale,
        "residual_target_modules": parse_csv(args.residual_target_modules),
        "training": {
            "num_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "max_grad_norm": args.max_grad_norm,
            "ewc_lambda": args.ewc_lambda,
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


def train_residual_ewc(args: argparse.Namespace, output_dir: Path) -> Path:
    train_rows = build_train_rows(args)
    guard_rows = build_guard_rows(args)
    write_jsonl(output_dir / "train_apply_only_rows.jsonl", train_rows)
    write_jsonl(output_dir / "fisher_guard_rows.jsonl", guard_rows)
    write_mix_report(output_dir / "mix_report.txt", train_rows, guard_rows)
    log(f"Train rows: {len(train_rows)}")
    log(f"Guard rows: {len(guard_rows)}")

    from transformers import AutoTokenizer

    prep_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    train_examples = build_examples(train_rows, prep_tokenizer, args.max_seq_len, args.boxed_tail_weight)
    guard_examples = build_examples(guard_rows, prep_tokenizer, args.max_seq_len, 1.0)
    if not train_examples:
        raise RuntimeError("No train examples.")
    save_metadata(args, output_dir, train_rows, guard_rows, train_examples, guard_examples, None)
    if args.dry_run:
        log("DRY_RUN=true: stopped before model load.")
        return output_dir / "residual_adapter"

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
    trainable_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_named:
        raise RuntimeError("No residual trainable parameters.")

    fisher, theta_star, fisher_stats = estimate_fisher(
        torch,
        model,
        device,
        guard_examples,
        trainable_named,
        batch_size=args.fisher_batch_size,
        normalize=args.normalize_fisher,
    )
    log(f"Fisher stats: {fisher_stats}")
    save_metadata(args, output_dir, train_rows, guard_rows, train_examples, guard_examples, fisher_stats)

    optimizer = torch.optim.AdamW([param for _, param in trainable_named], lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    indices = list(range(len(train_examples)))
    if args.shuffle:
        rng.shuffle(indices)
    cursor = 0
    train_log: list[dict] = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.num_steps + 1):
        batch_indices, cursor, finished_epoch = next_batch_indices(indices, cursor, args.batch_size)
        if not batch_indices:
            cursor = 0
            if args.shuffle:
                random.Random(args.seed + step).shuffle(indices)
            batch_indices, cursor, finished_epoch = next_batch_indices(indices, cursor, args.batch_size)
        batch = [train_examples[i] for i in batch_indices]
        accum = math.ceil(len(batch) / args.micro_batch_size)
        total_loss_sum = 0.0
        total_weight_sum = 0.0
        total_task_loss = 0.0
        total_ewc_loss = 0.0
        step_t0 = time.time()

        for mb_start in range(0, len(batch), args.micro_batch_size):
            micro = batch[mb_start : mb_start + args.micro_batch_size]
            task_loss, loss_sum, weight_sum = forward_weighted_loss(torch, model, device, micro)
            penalty = ewc_penalty(torch, trainable_named, fisher, theta_star)
            loss = task_loss + args.ewc_lambda * penalty
            (loss / accum).backward()
            total_loss_sum += float(loss_sum.detach().cpu())
            total_weight_sum += float(weight_sum.detach().cpu())
            total_task_loss += float(task_loss.detach().cpu())
            total_ewc_loss += float(penalty.detach().cpu())
            del task_loss, penalty, loss, loss_sum, weight_sum

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([param for _, param in trainable_named], args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss_sum / max(total_weight_sum, 1.0)
        row = {
            "step": step,
            "token_loss": avg_loss,
            "task_loss_mean": total_task_loss / max(accum, 1),
            "ewc_penalty_mean": total_ewc_loss / max(accum, 1),
            "ewc_lambda": args.ewc_lambda,
            "batch_size": len(batch),
            "tokens": int(total_weight_sum),
            "wall_sec": round(time.time() - step_t0, 3),
        }
        train_log.append(row)
        if step % args.logging_steps == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(
                f"[{now()}] step={step:04d}/{args.num_steps} token_loss={avg_loss:.5f} "
                f"task={row['task_loss_mean']:.5f} ewc={row['ewc_penalty_mean']:.8f} mem={mem:.1f}GB peak={peak:.1f}GB"
            )
        if finished_epoch:
            cursor = 0
            if args.shuffle:
                random.Random(args.seed + step).shuffle(indices)
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
        raise ValueError("This script keeps submission compatibility by using base_lora_rank=target_rank=32.")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("--batch-size must be divisible by --micro-batch-size")
    output_dir = reset_dir(args.output_dir)
    log(f"[{now()}] output_dir={output_dir}")

    if args.merge_only:
        if not args.residual_adapter_dir:
            raise ValueError("--merge-only requires --residual-adapter-dir")
        residual_dir = Path(args.residual_adapter_dir)
    else:
        residual_dir = train_residual_ewc(args, output_dir)
        if args.dry_run:
            return

    merged_name = f"merged_adapter_scale_{str(args.residual_scale).replace('.', 'p')}"
    merge_residual_to_rank32(
        args.initial_adapter,
        residual_dir,
        output_dir / merged_name,
        target_rank=args.target_rank,
        residual_scale=args.residual_scale,
        zip_submission=args.zip_submission,
    )


if __name__ == "__main__":
    main()
