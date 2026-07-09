#!/usr/bin/env python3
"""Teacher-trace micro SFT from evaluator debug_predictions.jsonl.

This is a stricter successor to sft_debug_correction_micro.py.  Instead of
training only on short "final answer" corrections, it uses the evaluator debug
row's verified reference_response as the assistant target whenever available.
Those traces contain the exact symbolic branch, punctuation handling, and
boxed-answer termination that the weak equation_numeric cases need.
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
    env_path,
    env_str,
    extract_final_answer,
    load_jsonl,
    load_unsloth_lora_model,
    log,
    now,
    patch_cce_forward,
    reset_dir,
    save_adapter,
    str2bool,
    verify,
    write_jsonl,
    zip_adapter,
)


DEFAULT_DRIFT_DEBUG_JSONL = (
    "/home/ubuntu/evaluator/results/"
    "sft_guarded_repair_v2_60pc_20260518_053443/debug_predictions.jsonl"
)

DEFAULT_ANCHOR_IDS = (
    "hk_3302f383",
    "hk_7283eb09",
    "hk_c095f799-p0",
    "my_bit_manipulation_00628",
    "my_bit_manipulation_00933",
    "my_equation_numeric_00239",
)

IM_END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-trace micro SFT from debug_predictions.jsonl.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--source-debug-jsonl", default=env_str("SOURCE_DEBUG_JSONL", PREVIOUS_EVAL_JSONL))
    parser.add_argument("--drift-debug-jsonl", default=env_str("DRIFT_DEBUG_JSONL", DEFAULT_DRIFT_DEBUG_JSONL))
    parser.add_argument(
        "--output-dir",
        default=str(
            env_path(
                "TEACHER_TRACE_OUTPUT_DIR",
                f"/home/ubuntu/additonal_tuning/outputs/sft_debug_teacher_trace_micro_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
            )
        ),
    )
    parser.add_argument("--prepared-jsonl", default=env_str("TEACHER_TRACE_PREPARED_JSONL", ""))
    parser.add_argument("--resample", type=str2bool, default=env_bool("RESAMPLE", True))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))

    parser.add_argument("--anchor-ids", default=env_str("TEACHER_TRACE_ANCHOR_IDS", ",".join(DEFAULT_ANCHOR_IDS)))
    parser.add_argument("--wrong-repeat", type=int, default=env_int("TEACHER_TRACE_WRONG_REPEAT", 3))
    parser.add_argument("--bit-wrong-repeat", type=int, default=env_int("TEACHER_TRACE_BIT_WRONG_REPEAT", 3))
    parser.add_argument("--anchor-repeat", type=int, default=env_int("TEACHER_TRACE_ANCHOR_REPEAT", 4))
    parser.add_argument("--equation-replay-count", type=int, default=env_int("TEACHER_TRACE_EQUATION_REPLAY_COUNT", 18))
    parser.add_argument("--bit-replay-count", type=int, default=env_int("TEACHER_TRACE_BIT_REPLAY_COUNT", 24))
    parser.add_argument("--replay-repeat", type=int, default=env_int("TEACHER_TRACE_REPLAY_REPEAT", 1))
    parser.add_argument("--boxed-tail-weight", type=float, default=env_float("TEACHER_TRACE_BOXED_TAIL_WEIGHT", 3.0))
    parser.add_argument("--include-drift-no-boxed", type=str2bool, default=env_bool("TEACHER_TRACE_INCLUDE_DRIFT_NO_BOXED", True))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

    parser.add_argument("--max-seq-len", type=int, default=env_int("MAX_MODEL_LEN", COMPETITION_MAX_MODEL_LEN))
    parser.add_argument("--lora-rank", type=int, default=env_int("LORA_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--lora-alpha", type=int, default=env_int("LORA_ALPHA", 32))
    parser.add_argument("--load-in-4bit", type=str2bool, default=env_bool("LOAD_IN_4BIT", False))
    parser.add_argument("--gradient-checkpointing", default=env_str("GRADIENT_CHECKPOINTING", "unsloth"))

    parser.add_argument("--epochs", type=int, default=env_int("TEACHER_TRACE_EPOCHS", 1))
    parser.add_argument("--num-steps", type=int, default=env_int("TEACHER_TRACE_NUM_STEPS", 0))
    parser.add_argument("--batch-size", type=int, default=env_int("TEACHER_TRACE_BATCH_SIZE", 4))
    parser.add_argument("--micro-batch-size", type=int, default=env_int("TEACHER_TRACE_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("TEACHER_TRACE_LR", 4e-7))
    parser.add_argument("--weight-decay", type=float, default=env_float("TEACHER_TRACE_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("TEACHER_TRACE_MAX_GRAD_NORM", 0.20))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("TEACHER_TRACE_SHUFFLE", True))
    parser.add_argument("--logging-steps", type=int, default=env_int("TEACHER_TRACE_LOGGING_STEPS", 1))
    parser.add_argument("--save-steps", type=int, default=env_int("TEACHER_TRACE_SAVE_STEPS", 6))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def clean_prompt(prompt: str) -> str:
    return re.sub(r"<\|[^>]+?\|>", "", str(prompt)).strip()


def boxed(answer: str) -> str:
    return f"\\boxed{{{answer}}}"


def normalize_response(text: str) -> str:
    response = str(text).strip()
    marker = "<|im_start|>assistant"
    if marker in response:
        response = response.split(marker, 1)[1].strip()
    # The Nemotron chat template with enable_thinking=True already ends the
    # prompt with "<think>\n".  The supervised continuation must begin after
    # that marker; otherwise SFT learns an impossible "<think><think>" prefix.
    if response.startswith("<think>"):
        response = response[len("<think>") :].lstrip()
    if not response.endswith(IM_END):
        response = response.rstrip() + IM_END
    return response


def is_verified_target(text: str | None, answer: str) -> bool:
    if not text:
        return False
    extracted = extract_final_answer(str(text))
    return verify(str(answer), str(extracted))


def fallback_response(row: dict) -> str:
    answer = str(row["answer"]).strip()
    category = str(row.get("category", "unknown"))
    if category == "equation_numeric":
        lines = [
            "Select the branch from the operator character in the input.",
            "Copy literal punctuation symbols exactly into the output.",
            f"Final answer: {boxed(answer)}",
        ]
        return "\n".join(lines) + IM_END
    if category == "bit_manipulation":
        return f"Resolve each bit position independently, then output the 8-bit string.\nFinal answer: {boxed(answer)}{IM_END}"
    return f"Final answer: {boxed(answer)}{IM_END}"


def teacher_response(row: dict) -> tuple[str, str, str]:
    """Return (response, response_source, extracted_answer)."""
    answer = str(row["answer"]).strip()
    candidates = [
        ("reference_response", row.get("reference_response")),
        ("raw_output_exact", row.get("raw_output") if int(row.get("exact_match", 0)) == 1 else None),
    ]
    for source_name, candidate in candidates:
        if is_verified_target(candidate, answer):
            response = normalize_response(str(candidate))
            return response, source_name, extract_final_answer(response)

    response = fallback_response(row)
    return response, "fallback_boxed_answer", extract_final_answer(response)


def load_debug(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    for row in load_jsonl(path):
        pid = str(row.get("problem_id", ""))
        if pid:
            rows[pid] = row
    return rows


def add_repeated(out: list[dict], row: dict, group: str, repeat: int) -> None:
    response, response_source, extracted = teacher_response(row)
    for idx in range(max(0, repeat)):
        out.append(
            {
                "problem_id": str(row["problem_id"]),
                "category": str(row.get("category", "unknown")),
                "prompt": clean_prompt(str(row["prompt"])),
                "answer": str(row["answer"]).strip(),
                "source_prediction": str(row.get("prediction", "")),
                "source_exact_match": int(row.get("exact_match", 0)),
                "mix_group": group,
                "repeat_index": idx,
                "response_source": response_source,
                "target_extracted_answer": str(extracted),
                "response": response,
            }
        )


def build_rows(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    source = load_debug(args.source_debug_jsonl)
    drift = load_debug(args.drift_debug_jsonl)
    if not source:
        raise FileNotFoundError(f"No source debug rows loaded from {args.source_debug_jsonl}")

    rows: list[dict] = []

    wrong_equation = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 0
    ]
    wrong_equation.sort(key=lambda row: str(row.get("problem_id", "")))
    for row in wrong_equation:
        add_repeated(rows, row, "equation_wrong_teacher_trace", args.wrong_repeat)

    wrong_bit = [
        row
        for row in source.values()
        if row.get("category") == "bit_manipulation" and int(row.get("exact_match", 0)) == 0
    ]
    wrong_bit.sort(key=lambda row: str(row.get("problem_id", "")))
    for row in wrong_bit:
        add_repeated(rows, row, "bit_wrong_teacher_trace", args.bit_wrong_repeat)

    anchor_ids = [pid.strip() for pid in args.anchor_ids.split(",") if pid.strip()]
    if args.include_drift_no_boxed and drift:
        for pid, row in drift.items():
            raw = str(row.get("raw_output", ""))
            if "\\boxed{" not in raw:
                anchor_ids.append(pid)

    seen_anchor: set[str] = set()
    for pid in anchor_ids:
        if pid in seen_anchor:
            continue
        seen_anchor.add(pid)
        row = source.get(pid) or drift.get(pid)
        if row is None:
            log(f"WARNING: anchor id not found in debug rows: {pid}")
            continue
        add_repeated(rows, row, "drift_anchor_teacher_trace", args.anchor_repeat)

    used = {str(row["problem_id"]) for row in rows}
    eq_replay = [
        row
        for row in source.values()
        if row.get("category") == "equation_numeric"
        and int(row.get("exact_match", 0)) == 1
        and str(row.get("problem_id", "")) not in used
    ]
    bit_replay = [
        row
        for row in source.values()
        if row.get("category") == "bit_manipulation"
        and int(row.get("exact_match", 0)) == 1
        and str(row.get("problem_id", "")) not in used
    ]
    for row in rng.sample(eq_replay, min(args.equation_replay_count, len(eq_replay))):
        add_repeated(rows, row, "equation_correct_teacher_replay", args.replay_repeat)
    for row in rng.sample(bit_replay, min(args.bit_replay_count, len(bit_replay))):
        add_repeated(rows, row, "bit_correct_teacher_replay", args.replay_repeat)

    rng.shuffle(rows)
    return rows


def write_mix_report(output_dir: Path, rows: list[dict]) -> None:
    report = {
        "rows": len(rows),
        "unique_problem_ids": len({str(row.get("problem_id", "")) for row in rows}),
        "by_group": Counter(str(row.get("mix_group", "")) for row in rows),
        "by_category": Counter(str(row.get("category", "")) for row in rows),
        "by_response_source": Counter(str(row.get("response_source", "")) for row in rows),
    }
    with (output_dir / "mix_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"rows: {report['rows']}\n")
        f.write(f"unique_problem_ids: {report['unique_problem_ids']}\n")
        for key in ("by_group", "by_category", "by_response_source"):
            f.write(f"{key}:\n")
            for name, count in sorted(report[key].items()):
                f.write(f"  {name}: {count}\n")


def prepare_rows(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir)
    prepared = Path(args.prepared_jsonl) if args.prepared_jsonl else output_dir / "prepared_teacher_trace_micro.jsonl"
    if prepared.exists() and not args.resample:
        rows = load_jsonl(prepared)
        log(f"Loaded prepared teacher-trace rows: {prepared} ({len(rows)} rows)")
    else:
        rows = build_rows(args)
        write_jsonl(prepared, rows)
        log(f"Wrote prepared teacher-trace rows: {prepared} ({len(rows)} rows)")
    write_mix_report(output_dir, rows)
    return rows


def response_token_weights(response: str, tokenizer, boxed_tail_weight: float) -> list[float]:
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    weights = [1.0] * len(response_ids)
    if boxed_tail_weight <= 1.0:
        return weights

    boxed_pos = response.rfind("\\boxed{")
    if boxed_pos < 0:
        return weights

    prefix_ids = tokenizer(response[:boxed_pos], add_special_tokens=False)["input_ids"]
    for idx in range(min(len(prefix_ids), len(weights)), len(weights)):
        weights[idx] = boxed_tail_weight
    return weights


def build_examples(rows: list[dict], tokenizer, max_seq_len: int, boxed_tail_weight: float) -> list[dict]:
    examples = []
    skipped = 0
    skipped_by_group: Counter[str] = Counter()
    for row in rows:
        prompt_text = competition_prompt(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response = str(row["response"])
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        response_weights = response_token_weights(response, tokenizer, boxed_tail_weight)
        tokens = prompt_ids + response_ids
        mask = [0.0] * len(prompt_ids) + response_weights
        if len(tokens) > max_seq_len:
            skipped += 1
            skipped_by_group[str(row.get("mix_group", ""))] += 1
            continue
        if len(tokens) < 2 or not response_ids:
            skipped += 1
            skipped_by_group[str(row.get("mix_group", ""))] += 1
            continue
        examples.append(
            {
                "problem_id": row["problem_id"],
                "category": row["category"],
                "mix_group": row["mix_group"],
                "response_source": row.get("response_source", ""),
                "tokens": tokens[:-1],
                "targets": tokens[1:],
                "weights": mask[1:],
                "length": len(tokens) - 1,
                "supervised": len(response_ids),
            }
        )

    log(f"Teacher-trace SFT examples: {len(examples)} (skipped={skipped})")
    for group, count in sorted(Counter(ex["mix_group"] for ex in examples).items()):
        log(f"  {group}: {count}")
    if skipped_by_group:
        for group, count in sorted(skipped_by_group.items()):
            log(f"  skipped {group}: {count}")
    return examples


def save_metadata(args: argparse.Namespace, output_dir: Path, rows: list[dict], examples: list[dict], num_steps: int) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "purpose": "debug_predictions teacher-trace micro SFT",
        "rows": len(rows),
        "examples": len(examples),
        "unique_problem_ids": len({str(row.get("problem_id", "")) for row in rows}),
        "by_group": dict(Counter(str(row.get("mix_group", "")) for row in rows)),
        "by_response_source": dict(Counter(str(row.get("response_source", "")) for row in rows)),
        "source_debug_jsonl": args.source_debug_jsonl,
        "drift_debug_jsonl": args.drift_debug_jsonl,
        "initial_adapter": args.initial_adapter,
        "model_path": args.model_path,
        "training": {
            "num_steps": num_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "boxed_tail_weight": args.boxed_tail_weight,
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


def next_batch_indices(indices: list[int], cursor: int, batch_size: int) -> tuple[list[int], int, bool]:
    if cursor >= len(indices):
        return [], 0, True
    end = min(cursor + batch_size, len(indices))
    return indices[cursor:end], end, end >= len(indices)


def main() -> None:
    args = parse_args()
    if args.lora_rank > COMPETITION_MAX_LORA_RANK:
        raise ValueError(f"lora_rank={args.lora_rank} exceeds competition max_lora_rank={COMPETITION_MAX_LORA_RANK}")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("--batch-size must be divisible by --micro-batch-size")

    output_dir = reset_dir(args.output_dir)
    log(f"[{now()}] output_dir={output_dir}")
    log(f"initial_adapter={args.initial_adapter}")

    rows = prepare_rows(args)
    from transformers import AutoTokenizer

    prep_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    examples = build_examples(rows, prep_tokenizer, args.max_seq_len, args.boxed_tail_weight)
    if not examples:
        raise RuntimeError("No teacher-trace SFT examples.")

    num_steps = args.num_steps
    if num_steps <= 0:
        num_steps = max(1, math.ceil(len(examples) / args.batch_size) * max(1, args.epochs))
    save_metadata(args, output_dir, rows, examples, num_steps)
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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
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
        row = {
            "step": step,
            "loss": avg_loss,
            "tokens": int(total_weight_sum),
            "batch_size": len(batch),
            "epoch": epoch,
            "wall_sec": round(time.time() - step_t0, 3),
        }
        train_log.append(row)
        if step % args.logging_steps == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(f"[{now()}] step={step:04d}/{num_steps} loss={avg_loss:.5f} mem={mem:.1f}GB peak={peak:.1f}GB")

        if args.save_steps and step % args.save_steps == 0:
            ckpt_dir = output_dir / f"checkpoint-{step:04d}"
            save_adapter(model, tokenizer, ckpt_dir, stack)

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
        zip_path = zip_adapter(final_dir)
        log(f"Submission zip: {zip_path}")


if __name__ == "__main__":
    main()
