#!/usr/bin/env python3
"""Direct GRPO-style tuning for equation_numeric symbolic failures.

This does not use GRPOTrainer. It implements a small grouped policy-gradient
loop directly:
  1. sample a guarded prompt mix from merged_sft_dataset,
  2. generate N completions per prompt, default N=8,
  3. score each completion with the competition metric,
  4. normalize rewards within the prompt group,
  5. update only LoRA weights using completion-token logprobs.

The script is intentionally conservative. Use this after a small SFT pass, or
point --initial-adapter at submission_1 if you want to test GRPO directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
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
    competition_prompt,
    discover_adapter,
    env_bool,
    env_float,
    env_int,
    env_path,
    env_str,
    extract_final_answer,
    load_token_dataset,
    load_unsloth_lora_model,
    log,
    now,
    patch_cce_forward,
    reset_dir,
    restore_forward,
    save_adapter,
    save_mix_report,
    str2bool,
    verify,
    write_jsonl,
    zip_adapter,
)


FALLBACK_PATTERNS = [
    "absolute difference",
    "default to concatenation",
    "question operator is not found",
    "question operator is unknown",
    "we default",
    "multiply+1",
    "multiply-1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct GRPO-style symbolic equation repair.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--dataset-root", default=env_str("DATASET_ROOT", DATASET_ROOT))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument(
        "--output-dir",
        default=str(
            env_path(
                "GRPO_OUTPUT_DIR",
                f"/home/ubuntu/additonal_tuning/outputs/grpo_symbolic_fix_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
            )
        ),
    )
    parser.add_argument("--prepared-jsonl", default=env_str("GRPO_PREPARED_JSONL", ""))
    parser.add_argument("--resample", type=str2bool, default=env_bool("RESAMPLE", True))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))

    parser.add_argument("--total-prompts", type=int, default=env_int("GRPO_TOTAL_PROMPTS", 128))
    parser.add_argument("--hard-ratio", type=float, default=env_float("GRPO_HARD_RATIO", 0.50))
    parser.add_argument("--equation-replay-ratio", type=float, default=env_float("GRPO_EQUATION_REPLAY_RATIO", 0.20))
    parser.add_argument("--other-replay-ratio", type=float, default=env_float("GRPO_OTHER_REPLAY_RATIO", 0.30))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))

    parser.add_argument("--max-seq-len", type=int, default=env_int("MAX_MODEL_LEN", COMPETITION_MAX_MODEL_LEN))
    parser.add_argument("--max-prompt-tokens", type=int, default=env_int("GRPO_MAX_PROMPT_TOKENS", 512))
    parser.add_argument("--max-completion-tokens", type=int, default=env_int("GRPO_MAX_COMPLETION_TOKENS", 1536))
    parser.add_argument("--loss-max-completion-tokens", type=int, default=env_int("LOSS_MAX_COMPLETION_TOKENS", 768))
    parser.add_argument("--num-generations", type=int, default=env_int("NUM_GENERATIONS", 8))
    parser.add_argument("--rollout-temperature", type=float, default=env_float("ROLLOUT_TEMPERATURE", 0.7))
    parser.add_argument("--rollout-top-p", type=float, default=env_float("ROLLOUT_TOP_P", 1.0))
    parser.add_argument("--sequential-rollouts", type=str2bool, default=env_bool("SEQUENTIAL_ROLLOUTS", True))

    parser.add_argument("--lora-rank", type=int, default=env_int("LORA_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--lora-alpha", type=int, default=env_int("LORA_ALPHA", 32))
    parser.add_argument("--load-in-4bit", type=str2bool, default=env_bool("LOAD_IN_4BIT", False))
    parser.add_argument("--gradient-checkpointing", default=env_str("GRADIENT_CHECKPOINTING", "unsloth"))

    parser.add_argument("--num-steps", type=int, default=env_int("GRPO_NUM_STEPS", 64))
    parser.add_argument("--train-micro-batch-size", type=int, default=env_int("GRPO_TRAIN_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("GRPO_LR", 1e-6))
    parser.add_argument("--weight-decay", type=float, default=env_float("GRPO_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("MAX_GRAD_NORM", 0.5))
    parser.add_argument("--exact-reward", type=float, default=env_float("EXACT_REWARD", 1.0))
    parser.add_argument("--boxed-reward", type=float, default=env_float("BOXED_REWARD", 0.05))
    parser.add_argument("--missing-boxed-penalty", type=float, default=env_float("MISSING_BOXED_PENALTY", -0.10))
    parser.add_argument("--symbolic-fallback-penalty", type=float, default=env_float("SYMBOLIC_FALLBACK_PENALTY", -0.10))
    parser.add_argument("--internal-code-penalty", type=float, default=env_float("INTERNAL_CODE_PENALTY", -0.15))
    parser.add_argument("--skip-zero-advantage", type=str2bool, default=env_bool("SKIP_ZERO_ADVANTAGE", True))
    parser.add_argument("--logging-steps", type=int, default=env_int("LOGGING_STEPS", 1))
    parser.add_argument("--save-steps", type=int, default=env_int("SAVE_STEPS", 0))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def prepare_prompts(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir)
    prepared = Path(args.prepared_jsonl) if args.prepared_jsonl else output_dir / "prepared_grpo_prompts.jsonl"
    if prepared.exists() and not args.resample:
        rows = []
        with prepared.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        log(f"Loaded prepared GRPO prompts: {prepared} ({len(rows)} rows)")
        return rows

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    all_rows = load_token_dataset(args.dataset_root)
    annotated = annotate_rows(all_rows, tokenizer)
    mix_cfg = MixConfig(
        total_examples=args.total_prompts,
        hard_ratio=args.hard_ratio,
        equation_replay_ratio=args.equation_replay_ratio,
        other_replay_ratio=args.other_replay_ratio,
        seed=args.seed,
        include_previous_wrong=True,
    )
    rows = build_guarded_mix(annotated, mix_cfg)

    formatted: list[dict] = []
    for row in rows:
        prompt_text = competition_prompt(tokenizer, row["prompt"], system_prompt=None)
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if prompt_len > args.max_prompt_tokens:
            continue
        formatted.append(
            {
                "problem_id": row["problem_id"],
                "category": row["category"],
                "mix_group": row.get("mix_group", ""),
                "prompt": row["prompt"],
                "prompt_formatted": prompt_text,
                "prompt_tokens": prompt_len,
                "answer": row["answer"],
                "is_symbolic_answer": row.get("is_symbolic_answer", False),
            }
        )

    if not formatted:
        raise RuntimeError(
            f"No prompts left after max_prompt_tokens={args.max_prompt_tokens}. "
            "Increase --max-prompt-tokens if needed."
        )
    write_jsonl(prepared, formatted)
    save_mix_report(output_dir / "mix_report.txt", rows)
    log(f"Wrote prepared GRPO prompts: {prepared} ({len(formatted)} rows)")
    return formatted


def reward_completion(
    completion: str,
    gold: str,
    is_symbolic: bool,
    args: argparse.Namespace,
) -> tuple[float, dict]:
    pred = extract_final_answer(completion)
    exact = verify(gold, pred)
    reward = args.exact_reward if exact else 0.0

    has_boxed = "\\boxed{" in completion
    reward += args.boxed_reward if has_boxed else args.missing_boxed_penalty

    lowered = completion.lower()
    fallback_hit = is_symbolic and any(pattern in lowered for pattern in FALLBACK_PATTERNS)
    if fallback_hit and not exact:
        reward += args.symbolic_fallback_penalty

    internal_code = False
    if is_symbolic and not exact:
        stripped = pred.strip()
        internal_code = bool(re.fullmatch(r"[A-Zxyz?]{2,}", stripped))
        if internal_code:
            reward += args.internal_code_penalty

    info = {
        "prediction": pred,
        "exact": int(exact),
        "has_boxed": int(has_boxed),
        "fallback_hit": int(fallback_hit),
        "internal_code": int(internal_code),
        "reward": reward,
    }
    return reward, info


def normalize_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < 1e-6:
        return [0.0 for _ in rewards]
    return [(r - mean) / std for r in rewards]


def generate_completions(model, tokenizer, stack, prompt: str, args: argparse.Namespace) -> list[str]:
    torch = stack["torch"]
    FastLanguageModel = stack["FastLanguageModel"]
    restore_forward(model)
    FastLanguageModel.for_inference(model)

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    completions: list[str] = []

    with torch.inference_mode():
        if args.sequential_rollouts:
            for _ in range(args.num_generations):
                out = model.generate(
                    **encoded,
                    max_new_tokens=args.max_completion_tokens,
                    do_sample=True,
                    temperature=args.rollout_temperature,
                    top_p=args.rollout_top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                completions.append(tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True))
        else:
            out = model.generate(
                **encoded,
                max_new_tokens=args.max_completion_tokens,
                do_sample=True,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            for i in range(out.shape[0]):
                completions.append(tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True))

    del encoded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    FastLanguageModel.for_training(model)
    patch_cce_forward(model, stack)
    return completions


def build_policy_examples(tokenizer, prompt: str, completions: list[str], advantages: list[float], args: argparse.Namespace) -> list[dict]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    examples: list[dict] = []
    eos = tokenizer.eos_token or ""
    for completion, advantage in zip(completions, advantages):
        comp_ids = tokenizer.encode(completion + eos, add_special_tokens=False)
        if not comp_ids:
            continue
        comp_ids = comp_ids[: args.loss_max_completion_tokens]
        full = (prompt_ids + comp_ids)[: args.max_seq_len]
        if len(full) <= len(prompt_ids) or len(full) < 2:
            continue
        input_ids = full[:-1]
        labels = full[1:]
        weights = []
        used = 0
        for i in range(len(input_ids)):
            is_completion_target = i + 1 >= len(prompt_ids)
            if is_completion_target and used < args.loss_max_completion_tokens:
                weights.append(1.0)
                used += 1
            else:
                weights.append(0.0)
        if used == 0:
            continue
        examples.append(
            {
                "tokens": input_ids,
                "targets": labels,
                "weights": weights,
                "advantage": float(advantage),
            }
        )
    return examples


def train_on_policy_examples(model, stack, optimizer, examples: list[dict], args: argparse.Namespace) -> dict:
    torch = stack["torch"]
    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    accum = math.ceil(len(examples) / args.train_micro_batch_size)
    total_loss = 0.0
    total_tokens = 0.0

    for mb_start in range(0, len(examples), args.train_micro_batch_size):
        micro = examples[mb_start : mb_start + args.train_micro_batch_size]
        max_len = max(len(ex["tokens"]) for ex in micro)
        input_ids = torch.zeros(len(micro), max_len, dtype=torch.long, device=device)
        labels = torch.zeros(len(micro), max_len, dtype=torch.long, device=device)
        weights = torch.zeros(len(micro), max_len, dtype=torch.float32, device=device)
        advantages = torch.tensor([ex["advantage"] for ex in micro], dtype=torch.float32, device=device)
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
            per_example_loss = (per_token_ce * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            loss = (advantages * per_example_loss).mean()

        (loss / accum).backward()
        total_loss += float(loss.detach().cpu()) * len(micro)
        total_tokens += float(weights.sum().detach().cpu())
        del input_ids, labels, weights, advantages, attention_mask, per_token_ce, per_example_loss, loss

    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "policy_loss": total_loss / max(1, len(examples)),
        "loss_tokens": int(total_tokens),
    }


def save_metadata(args: argparse.Namespace, output_dir: Path, prompts: list[dict]) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "purpose": "direct GRPO-style repair for equation_numeric symbolic branch-map failures",
        "prompts": len(prompts),
        "initial_adapter": args.initial_adapter,
        "model_path": args.model_path,
        "dataset_root": args.dataset_root,
        "training": {
            "num_steps": args.num_steps,
            "num_generations": args.num_generations,
            "learning_rate": args.learning_rate,
            "max_seq_len": args.max_seq_len,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_completion_tokens": args.max_completion_tokens,
            "loss_max_completion_tokens": args.loss_max_completion_tokens,
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
    if args.num_generations < 2:
        raise ValueError("GRPO needs at least 2 generations per prompt; 8 is recommended.")

    random.seed(args.seed)
    output_dir = reset_dir(args.output_dir)
    log(f"[{now()}] output_dir={output_dir}")
    log(f"initial_adapter={args.initial_adapter}")

    prompts = prepare_prompts(args)
    save_metadata(args, output_dir, prompts)
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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    rollout_path = output_dir / "rollouts.jsonl"
    train_log_path = output_dir / "train_log.jsonl"
    rng = random.Random(args.seed)
    skipped_zero_adv = 0

    for step in range(1, args.num_steps + 1):
        row = prompts[(step - 1) % len(prompts)]
        if step > 1 and (step - 1) % len(prompts) == 0:
            rng.shuffle(prompts)

        step_t0 = time.time()
        completions = generate_completions(model, tokenizer, stack, row["prompt_formatted"], args)
        rewards: list[float] = []
        infos: list[dict] = []
        for completion in completions:
            reward, info = reward_completion(completion, row["answer"], bool(row["is_symbolic_answer"]), args)
            rewards.append(reward)
            infos.append(info)
        advantages = normalize_advantages(rewards)

        rollout_record = {
            "step": step,
            "problem_id": row["problem_id"],
            "category": row["category"],
            "mix_group": row.get("mix_group", ""),
            "answer": row["answer"],
            "rewards": rewards,
            "advantages": advantages,
            "infos": infos,
            "completions": completions,
        }
        with rollout_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")

        if args.skip_zero_advantage and all(abs(a) < 1e-8 for a in advantages):
            skipped_zero_adv += 1
            update_info = {"policy_loss": 0.0, "loss_tokens": 0, "skipped_zero_advantage": 1}
        else:
            policy_examples = build_policy_examples(tokenizer, row["prompt_formatted"], completions, advantages, args)
            update_info = train_on_policy_examples(model, stack, optimizer, policy_examples, args)
            update_info["skipped_zero_advantage"] = 0

        exact_count = sum(info["exact"] for info in infos)
        log_row = {
            "step": step,
            "problem_id": row["problem_id"],
            "mix_group": row.get("mix_group", ""),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "exact_count": exact_count,
            "num_generations": len(completions),
            "wall_sec": round(time.time() - step_t0, 3),
            **update_info,
        }
        with train_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

        if step % args.logging_steps == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            pred_preview = infos[0]["prediction"] if infos else ""
            log(
                f"[{now()}] step={step:04d}/{args.num_steps} "
                f"reward_mean={log_row['reward_mean']:.3f} exact={exact_count}/{len(completions)} "
                f"loss={log_row['policy_loss']:.5f} pred0={pred_preview!r} "
                f"mem={mem:.1f}GB peak={peak:.1f}GB"
            )

        if args.save_steps and step % args.save_steps == 0:
            ckpt_dir = output_dir / f"checkpoint-{step:04d}"
            save_adapter(model, tokenizer, ckpt_dir, stack)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_dir = output_dir / "final_adapter"
    save_adapter(model, tokenizer, final_dir, stack)
    if args.zip_submission:
        zip_path = zip_adapter(final_dir)
        log(f"Submission zip: {zip_path}")
    log(f"Skipped zero-advantage groups: {skipped_zero_adv}/{args.num_steps}")


if __name__ == "__main__":
    main()
