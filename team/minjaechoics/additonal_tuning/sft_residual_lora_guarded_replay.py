#!/usr/bin/env python3
"""Residual LoRA with real guard replay and equation apply-only targets."""

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
from sft_debug_teacher_trace_micro import build_examples, next_batch_indices
from sft_residual_lora_svd import activate_residual_adapter, merge_residual_to_rank32, save_residual_adapter


IM_END = "<|im_end|>"
DEFAULT_OLD_CATEGORIES = "bit_manipulation,cipher,cryptarithm,gravity,numeral,unit_conversion"
DEFAULT_FRAGILE_IDS = "hk_3302f383,hk_7283eb09,hk_c095f799-p0,my_equation_numeric_00239"


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual LoRA guarded replay + apply-only SFT.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--source-debug-jsonl", default=env_str("SOURCE_DEBUG_JSONL", PREVIOUS_EVAL_JSONL))
    parser.add_argument("--output-dir", default=env_str("GUARD_OUTPUT_DIR", "/home/ubuntu/additonal_tuning/outputs/residual_lora_guarded_replay"))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))
    parser.add_argument("--merge-only", type=str2bool, default=env_bool("MERGE_ONLY", False))
    parser.add_argument("--residual-adapter-dir", default=env_str("RESIDUAL_ADAPTER_DIR", ""))

    parser.add_argument("--wrong-repeat", type=int, default=env_int("GUARD_WRONG_REPEAT", 4))
    parser.add_argument("--equation-replay-count", type=int, default=env_int("GUARD_EQUATION_REPLAY_COUNT", 12))
    parser.add_argument("--equation-replay-repeat", type=int, default=env_int("GUARD_EQUATION_REPLAY_REPEAT", 1))
    parser.add_argument("--old-categories", default=env_str("GUARD_OLD_CATEGORIES", DEFAULT_OLD_CATEGORIES))
    parser.add_argument("--guard-max-per-category", type=int, default=env_int("GUARD_MAX_PER_CATEGORY", 8))
    parser.add_argument("--guard-equation-correct-count", type=int, default=env_int("GUARD_EQUATION_CORRECT_COUNT", 8))
    parser.add_argument("--fragile-ids", default=env_str("GUARD_FRAGILE_IDS", DEFAULT_FRAGILE_IDS))
    parser.add_argument("--fragile-repeat", type=int, default=env_int("GUARD_FRAGILE_REPEAT", 2))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

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

    parser.add_argument("--num-steps", type=int, default=env_int("GUARD_NUM_STEPS", 8))
    parser.add_argument("--batch-size", type=int, default=env_int("GUARD_BATCH_SIZE", 2))
    parser.add_argument("--micro-batch-size", type=int, default=env_int("GUARD_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--guard-batch-size", type=int, default=env_int("GUARD_REPLAY_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("GUARD_LR", 1.5e-7))
    parser.add_argument("--weight-decay", type=float, default=env_float("GUARD_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("GUARD_MAX_GRAD_NORM", 0.05))
    parser.add_argument("--guard-loss-weight", type=float, default=env_float("GUARD_LOSS_WEIGHT", 0.7))
    parser.add_argument("--boxed-tail-weight", type=float, default=env_float("GUARD_BOXED_TAIL_WEIGHT", 2.0))
    parser.add_argument("--logging-steps", type=int, default=env_int("GUARD_LOGGING_STEPS", 1))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("GUARD_SHUFFLE", True))
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


def load_debug(path: str | Path) -> dict[str, dict]:
    rows = {}
    for row in load_jsonl(path):
        pid = str(row.get("problem_id", ""))
        if pid:
            rows[pid] = row
    return rows


def verified_reference(row: dict) -> str | None:
    answer = str(row.get("answer", "")).strip()
    reference = row.get("reference_response")
    if reference and verify(answer, extract_final_answer(str(reference))):
        return normalize_response(str(reference))
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


def guard_response(row: dict) -> str:
    answer = str(row["answer"]).strip()
    response = verified_reference(row)
    return response if response else f"Final answer: \\boxed{{{answer}}}{IM_END}"


def add_row(out: list[dict], row: dict, group: str, response: str, repeat: int) -> None:
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


def build_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rng = random.Random(args.seed)
    source = load_debug(args.source_debug_jsonl)
    train_rows: list[dict] = []
    guard_rows: list[dict] = []

    wrong_eq = sorted(
        [
            row
            for row in source.values()
            if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 0
        ],
        key=lambda row: str(row.get("problem_id", "")),
    )
    for row in wrong_eq:
        add_row(train_rows, row, "equation_wrong_apply_only", apply_only_response(row), args.wrong_repeat)

    eq_correct = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 1
    ]
    rng.shuffle(eq_correct)
    for row in eq_correct[: args.equation_replay_count]:
        add_row(train_rows, row, "equation_correct_apply_replay", apply_only_response(row), args.equation_replay_repeat)
    for row in eq_correct[: args.guard_equation_correct_count]:
        add_row(guard_rows, row, "guard_equation_correct_full", guard_response(row), 1)

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
            add_row(guard_rows, row, f"guard_{category}_full", guard_response(row), 1)

    seen_guard = {str(row["problem_id"]) for row in guard_rows}
    for pid in parse_csv(args.fragile_ids):
        row = source.get(pid)
        if row and pid not in seen_guard and int(row.get("exact_match", 0)) == 1:
            add_row(guard_rows, row, "guard_fragile_anchor_full", guard_response(row), args.fragile_repeat)
            seen_guard.add(pid)

    rng.shuffle(train_rows)
    rng.shuffle(guard_rows)
    return train_rows, guard_rows


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


def save_metadata(args: argparse.Namespace, output_dir: Path, train_rows: list[dict], guard_rows: list[dict], train_examples: list[dict], guard_examples: list[dict]) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "method": "residual_lora_guarded_replay_apply_only",
        "initial_adapter": args.initial_adapter,
        "train_rows": len(train_rows),
        "guard_rows": len(guard_rows),
        "train_examples": len(train_examples),
        "guard_examples": len(guard_examples),
        "residual_rank": args.residual_rank,
        "residual_alpha": args.residual_alpha,
        "residual_scale": args.residual_scale,
        "training": {
            "num_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "guard_batch_size": args.guard_batch_size,
            "guard_loss_weight": args.guard_loss_weight,
            "max_grad_norm": args.max_grad_norm,
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


def write_mix_report(output_dir: Path, train_rows: list[dict], guard_rows: list[dict]) -> None:
    with (output_dir / "mix_report.txt").open("w", encoding="utf-8") as f:
        for name, rows in (("train", train_rows), ("guard", guard_rows)):
            f.write(f"{name}_rows: {len(rows)}\n")
            for key, count in sorted(Counter(str(row.get("mix_group", "")) for row in rows).items()):
                f.write(f"  {key}: {count}\n")


def train(args: argparse.Namespace, output_dir: Path) -> Path:
    train_rows, guard_rows = build_rows(args)
    write_jsonl(output_dir / "train_rows.jsonl", train_rows)
    write_jsonl(output_dir / "guard_rows.jsonl", guard_rows)
    write_mix_report(output_dir, train_rows, guard_rows)
    log(f"Train rows: {len(train_rows)}")
    log(f"Guard rows: {len(guard_rows)}")

    from transformers import AutoTokenizer

    prep_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    train_examples = build_examples(train_rows, prep_tokenizer, args.max_seq_len, args.boxed_tail_weight)
    guard_examples = build_examples(guard_rows, prep_tokenizer, args.max_seq_len, 1.0)
    save_metadata(args, output_dir, train_rows, guard_rows, train_examples, guard_examples)
    if args.dry_run:
        log("DRY_RUN=true: stopped before model load.")
        return output_dir / "residual_adapter"
    if not train_examples or not guard_examples:
        raise RuntimeError("Need both train and guard examples.")

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
    train_indices = list(range(len(train_examples)))
    guard_indices = list(range(len(guard_examples)))
    if args.shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(guard_indices)
    train_cursor = 0
    guard_cursor = 0
    train_log: list[dict] = []

    for step in range(1, args.num_steps + 1):
        batch_indices, train_cursor, train_done = next_batch_indices(train_indices, train_cursor, args.batch_size)
        guard_batch_indices, guard_cursor, guard_done = next_batch_indices(guard_indices, guard_cursor, args.guard_batch_size)
        if not batch_indices:
            train_cursor = 0
            random.Random(args.seed + step).shuffle(train_indices)
            batch_indices, train_cursor, train_done = next_batch_indices(train_indices, train_cursor, args.batch_size)
        if not guard_batch_indices:
            guard_cursor = 0
            random.Random(args.seed + 1000 + step).shuffle(guard_indices)
            guard_batch_indices, guard_cursor, guard_done = next_batch_indices(guard_indices, guard_cursor, args.guard_batch_size)
        batch = [train_examples[i] for i in batch_indices]
        guard_batch = [guard_examples[i] for i in guard_batch_indices]
        optimizer.zero_grad(set_to_none=True)
        accum = math.ceil(len(batch) / args.micro_batch_size)
        step_task = 0.0
        step_guard = 0.0
        step_tokens = 0.0
        t0 = time.time()

        for mb_start in range(0, len(batch), args.micro_batch_size):
            micro = batch[mb_start : mb_start + args.micro_batch_size]
            task_loss, task_sum, task_weight = forward_weighted_loss(torch, model, device, micro)
            guard_loss, _, _ = forward_weighted_loss(torch, model, device, guard_batch)
            loss = task_loss + args.guard_loss_weight * guard_loss
            (loss / accum).backward()
            step_task += float(task_loss.detach().cpu())
            step_guard += float(guard_loss.detach().cpu())
            step_tokens += float(task_weight.detach().cpu())
            del task_loss, guard_loss, loss, task_sum, task_weight

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
        if train_done:
            train_cursor = 0
            random.Random(args.seed + step).shuffle(train_indices)
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
