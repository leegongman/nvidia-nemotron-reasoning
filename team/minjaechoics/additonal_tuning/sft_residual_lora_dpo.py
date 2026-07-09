#!/usr/bin/env python3
"""Tiny residual LoRA DPO for equation_numeric hard negatives.

This keeps the known-good submission_1 adapter frozen, trains only a rank-2
residual adapter, and exports a single rank-32 adapter by SVD recompression.
The DPO pairs are deliberately narrow:

    chosen   = compact verified apply-only answer
    rejected = compact continuation containing the old wrong prediction

Guard replay keeps a small set of old-category exact traces under CE loss.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
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
    DEFAULT_TARGET_MODULES,
    PREVIOUS_EVAL_JSONL,
    SUBMISSION_1_ROOT,
    competition_prompt,
    discover_adapter,
    env_bool,
    env_float,
    env_int,
    env_str,
    load_jsonl,
    load_unsloth_lora_model,
    log,
    now,
    patch_cce_forward,
    reset_dir,
    str2bool,
    write_jsonl,
)
from sft_debug_teacher_trace_micro import next_batch_indices, response_token_weights
from sft_residual_lora_guarded_replay import (
    DEFAULT_OLD_CATEGORIES,
    apply_only_response,
    clean_prompt,
    guard_response,
    load_debug,
    parse_csv,
)
from sft_residual_lora_svd import activate_residual_adapter, merge_residual_to_rank32, save_residual_adapter


IM_END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual LoRA DPO on equation_numeric hard negatives.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--source-debug-jsonl", default=env_str("SOURCE_DEBUG_JSONL", PREVIOUS_EVAL_JSONL))
    parser.add_argument("--output-dir", default=env_str("DPO_OUTPUT_DIR", "/home/ubuntu/additonal_tuning/outputs/residual_lora_dpo"))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))
    parser.add_argument("--merge-only", type=str2bool, default=env_bool("MERGE_ONLY", False))
    parser.add_argument("--residual-adapter-dir", default=env_str("RESIDUAL_ADAPTER_DIR", ""))

    parser.add_argument("--wrong-repeat", type=int, default=env_int("DPO_WRONG_REPEAT", 2))
    parser.add_argument("--old-categories", default=env_str("DPO_OLD_CATEGORIES", DEFAULT_OLD_CATEGORIES))
    parser.add_argument("--guard-max-per-category", type=int, default=env_int("DPO_GUARD_MAX_PER_CATEGORY", 6))
    parser.add_argument("--guard-equation-correct-count", type=int, default=env_int("DPO_GUARD_EQUATION_CORRECT_COUNT", 6))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

    parser.add_argument("--max-seq-len", type=int, default=env_int("MAX_MODEL_LEN", COMPETITION_MAX_MODEL_LEN))
    parser.add_argument("--base-lora-rank", type=int, default=env_int("BASE_LORA_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--base-lora-alpha", type=int, default=env_int("BASE_LORA_ALPHA", 32))
    parser.add_argument("--residual-rank", type=int, default=env_int("RESIDUAL_RANK", 2))
    parser.add_argument("--residual-alpha", type=int, default=env_int("RESIDUAL_ALPHA", 2))
    parser.add_argument("--residual-target-modules", default=env_str("RESIDUAL_TARGET_MODULES", "in_proj,out_proj,q_proj,k_proj,v_proj,o_proj"))
    parser.add_argument("--residual-scale", type=float, default=env_float("RESIDUAL_SCALE", 0.03))
    parser.add_argument("--target-rank", type=int, default=env_int("TARGET_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--load-in-4bit", type=str2bool, default=env_bool("LOAD_IN_4BIT", False))
    parser.add_argument("--gradient-checkpointing", default=env_str("GRADIENT_CHECKPOINTING", "unsloth"))

    parser.add_argument("--num-steps", type=int, default=env_int("DPO_NUM_STEPS", 6))
    parser.add_argument("--batch-size", type=int, default=env_int("DPO_BATCH_SIZE", 1))
    parser.add_argument("--guard-batch-size", type=int, default=env_int("DPO_GUARD_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("DPO_LR", 1e-7))
    parser.add_argument("--weight-decay", type=float, default=env_float("DPO_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("DPO_MAX_GRAD_NORM", 0.05))
    parser.add_argument("--beta", type=float, default=env_float("DPO_BETA", 0.10))
    parser.add_argument("--sft-weight", type=float, default=env_float("DPO_SFT_WEIGHT", 0.20))
    parser.add_argument("--guard-loss-weight", type=float, default=env_float("DPO_GUARD_LOSS_WEIGHT", 0.40))
    parser.add_argument("--boxed-tail-weight", type=float, default=env_float("DPO_BOXED_TAIL_WEIGHT", 2.0))
    parser.add_argument("--logging-steps", type=int, default=env_int("DPO_LOGGING_STEPS", 1))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("DPO_SHUFFLE", True))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def rejected_response(row: dict) -> str:
    wrong = str(row.get("prediction", "")).strip()
    if not wrong:
        wrong = "NOT_FOUND"
    return f"S4: APPLY target\nresult = {wrong}\n</think>\n\\boxed{{{wrong}}}{IM_END}"


def build_pair_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rng = random.Random(args.seed)
    source = load_debug(args.source_debug_jsonl)
    pairs: list[dict] = []

    wrong_eq = sorted(
        [
            row
            for row in source.values()
            if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 0
        ],
        key=lambda row: str(row.get("problem_id", "")),
    )
    for row in wrong_eq:
        for repeat_idx in range(max(1, args.wrong_repeat)):
            pairs.append(
                {
                    "problem_id": str(row["problem_id"]),
                    "category": "equation_numeric",
                    "prompt": clean_prompt(str(row["prompt"])),
                    "answer": str(row["answer"]).strip(),
                    "prediction": str(row.get("prediction", "")).strip(),
                    "repeat_index": repeat_idx,
                    "chosen_response": apply_only_response(row),
                    "rejected_response": rejected_response(row),
                }
            )

    guard_rows: list[dict] = []
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
            guard_rows.append(
                {
                    "problem_id": str(row["problem_id"]),
                    "category": category,
                    "prompt": clean_prompt(str(row["prompt"])),
                    "answer": str(row["answer"]).strip(),
                    "mix_group": f"guard_{category}_full",
                    "response": guard_response(row),
                }
            )

    eq_correct = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 1
    ]
    rng.shuffle(eq_correct)
    for row in eq_correct[: args.guard_equation_correct_count]:
        guard_rows.append(
            {
                "problem_id": str(row["problem_id"]),
                "category": "equation_numeric",
                "prompt": clean_prompt(str(row["prompt"])),
                "answer": str(row["answer"]).strip(),
                "mix_group": "guard_equation_correct_full",
                "response": guard_response(row),
            }
        )

    rng.shuffle(pairs)
    rng.shuffle(guard_rows)
    return pairs, guard_rows


def make_example(row: dict, response: str, tokenizer, max_seq_len: int, boxed_tail_weight: float, mix_group: str) -> dict | None:
    prompt_text = competition_prompt(tokenizer, row["prompt"])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    response_weights = response_token_weights(response, tokenizer, boxed_tail_weight)
    tokens = prompt_ids + response_ids
    mask = [0.0] * len(prompt_ids) + response_weights
    if len(tokens) > max_seq_len or len(tokens) < 2 or not response_ids:
        return None
    return {
        "problem_id": row["problem_id"],
        "category": row["category"],
        "mix_group": mix_group,
        "tokens": tokens[:-1],
        "targets": tokens[1:],
        "weights": mask[1:],
        "length": len(tokens) - 1,
    }


def build_pair_examples(pair_rows: list[dict], tokenizer, max_seq_len: int, boxed_tail_weight: float) -> list[dict]:
    examples: list[dict] = []
    skipped = 0
    for row in pair_rows:
        chosen = make_example(row, str(row["chosen_response"]), tokenizer, max_seq_len, boxed_tail_weight, "dpo_chosen")
        rejected = make_example(row, str(row["rejected_response"]), tokenizer, max_seq_len, boxed_tail_weight, "dpo_rejected")
        if chosen is None or rejected is None:
            skipped += 1
            continue
        examples.append({"row": row, "chosen": chosen, "rejected": rejected, "ref_margin": 0.0})
    log(f"DPO pairs: {len(examples)} (skipped={skipped})")
    return examples


def build_guard_examples(rows: list[dict], tokenizer, max_seq_len: int) -> list[dict]:
    examples = []
    skipped = 0
    for row in rows:
        ex = make_example(row, str(row["response"]), tokenizer, max_seq_len, 1.0, str(row["mix_group"]))
        if ex is None:
            skipped += 1
        else:
            examples.append(ex)
    log(f"Guard examples: {len(examples)} (skipped={skipped})")
    for group, count in sorted(Counter(ex["mix_group"] for ex in examples).items()):
        log(f"  {group}: {count}")
    return examples


def forward_logp_and_ce(torch, model, device, ex: dict) -> tuple[object, object, object]:
    input_ids = torch.tensor([ex["tokens"]], dtype=torch.long, device=device)
    labels = torch.tensor([ex["targets"]], dtype=torch.long, device=device)
    weights = torch.tensor([ex["weights"]], dtype=torch.float32, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        per_token_ce = model._cached_per_token_ce
        loss_sum = (per_token_ce * weights).sum()
        weight_sum = weights.sum().clamp_min(1.0)
        ce = loss_sum / weight_sum
        logp = -ce
    del input_ids, labels, weights, attention_mask, per_token_ce
    return logp, ce, weight_sum


def save_metadata(args: argparse.Namespace, output_dir: Path, pair_rows: list[dict], guard_rows: list[dict], dpo_examples: list[dict], guard_examples: list[dict]) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "method": "residual_lora_dpo_equation_hard_negative",
        "initial_adapter": args.initial_adapter,
        "pair_rows": len(pair_rows),
        "guard_rows": len(guard_rows),
        "dpo_examples": len(dpo_examples),
        "guard_examples": len(guard_examples),
        "residual_rank": args.residual_rank,
        "residual_alpha": args.residual_alpha,
        "residual_scale": args.residual_scale,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "sft_weight": args.sft_weight,
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


def train(args: argparse.Namespace, output_dir: Path) -> Path:
    pair_rows, guard_rows = build_pair_rows(args)
    write_jsonl(output_dir / "dpo_pairs.jsonl", pair_rows)
    write_jsonl(output_dir / "guard_rows.jsonl", guard_rows)
    log(f"Pair rows: {len(pair_rows)}")
    log(f"Guard rows: {len(guard_rows)}")

    from transformers import AutoTokenizer

    prep_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    dpo_examples = build_pair_examples(pair_rows, prep_tokenizer, args.max_seq_len, args.boxed_tail_weight)
    guard_examples = build_guard_examples(guard_rows, prep_tokenizer, args.max_seq_len)
    save_metadata(args, output_dir, pair_rows, guard_rows, dpo_examples, guard_examples)
    if args.dry_run:
        log("DRY_RUN=true: stopped before model load.")
        return output_dir / "residual_adapter"
    if not dpo_examples or not guard_examples:
        raise RuntimeError("Need both DPO and guard examples.")

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

    # Use the frozen initial point as the DPO reference margin.  The residual
    # adapter starts as an effectively zero delta, so this is cheap and stable.
    with torch.no_grad():
        for item in dpo_examples:
            chosen_logp, _, _ = forward_logp_and_ce(torch, model, device, item["chosen"])
            rejected_logp, _, _ = forward_logp_and_ce(torch, model, device, item["rejected"])
            item["ref_margin"] = float((chosen_logp - rejected_logp).detach().cpu())
            del chosen_logp, rejected_logp

    rng = random.Random(args.seed)
    dpo_indices = list(range(len(dpo_examples)))
    guard_indices = list(range(len(guard_examples)))
    if args.shuffle:
        rng.shuffle(dpo_indices)
        rng.shuffle(guard_indices)
    dpo_cursor = 0
    guard_cursor = 0
    train_log: list[dict] = []

    for step in range(1, args.num_steps + 1):
        pair_indices, dpo_cursor, dpo_done = next_batch_indices(dpo_indices, dpo_cursor, args.batch_size)
        guard_batch_indices, guard_cursor, guard_done = next_batch_indices(guard_indices, guard_cursor, args.guard_batch_size)
        if not pair_indices:
            dpo_cursor = 0
            random.Random(args.seed + step).shuffle(dpo_indices)
            pair_indices, dpo_cursor, dpo_done = next_batch_indices(dpo_indices, dpo_cursor, args.batch_size)
        if not guard_batch_indices:
            guard_cursor = 0
            random.Random(args.seed + 1000 + step).shuffle(guard_indices)
            guard_batch_indices, guard_cursor, guard_done = next_batch_indices(guard_indices, guard_cursor, args.guard_batch_size)

        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        step_dpo = 0.0
        step_sft = 0.0
        step_guard = 0.0
        denom = max(1, len(pair_indices))

        for pair_idx in pair_indices:
            pair = dpo_examples[pair_idx]
            chosen_logp, chosen_ce, _ = forward_logp_and_ce(torch, model, device, pair["chosen"])
            rejected_logp, _, _ = forward_logp_and_ce(torch, model, device, pair["rejected"])
            ref_margin = torch.tensor(float(pair["ref_margin"]), dtype=torch.float32, device=device)
            margin = (chosen_logp - rejected_logp) - ref_margin
            dpo_loss = -torch.nn.functional.logsigmoid(args.beta * margin)
            loss = dpo_loss + args.sft_weight * chosen_ce
            if guard_batch_indices:
                guard_losses = []
                for guard_idx in guard_batch_indices:
                    _, guard_ce, _ = forward_logp_and_ce(torch, model, device, guard_examples[guard_idx])
                    guard_losses.append(guard_ce)
                guard_loss = torch.stack(guard_losses).mean()
                loss = loss + args.guard_loss_weight * guard_loss
                step_guard += float(guard_loss.detach().cpu())
                del guard_loss, guard_losses
            (loss / denom).backward()
            step_dpo += float(dpo_loss.detach().cpu())
            step_sft += float(chosen_ce.detach().cpu())
            del chosen_logp, chosen_ce, rejected_logp, ref_margin, margin, dpo_loss, loss

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()

        row = {
            "step": step,
            "dpo_loss": step_dpo / denom,
            "chosen_ce": step_sft / denom,
            "guard_ce": step_guard / max(1, denom),
            "beta": args.beta,
            "wall_sec": round(time.time() - t0, 3),
        }
        train_log.append(row)
        if step % args.logging_steps == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(f"[{now()}] step={step:04d}/{args.num_steps} dpo={row['dpo_loss']:.5f} chosen_ce={row['chosen_ce']:.5f} guard={row['guard_ce']:.5f} mem={mem:.1f}GB peak={peak:.1f}GB")
        if dpo_done:
            dpo_cursor = 0
            random.Random(args.seed + step).shuffle(dpo_indices)
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
