#!/usr/bin/env python3
"""Train a tiny residual LoRA on top of frozen submission_1, then SVD-merge.

The training model keeps the original rank-32 adapter active and frozen, adds a
small "residual" adapter on selected projection modules, and trains only that
residual.  The exported adapter is a single rank-32 LoRA:

    Delta_final ~= Delta_submission_1 + residual_scale * Delta_residual

The recompression uses a low-rank QR/SVD identity, avoiding full huge-matrix SVD.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
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
    import_training_stack,
    load_unsloth_lora_model,
    log,
    now,
    patch_cce_forward,
    reset_dir,
    str2bool,
    zip_adapter,
)
from sft_debug_teacher_trace_micro import (
    DEFAULT_DRIFT_DEBUG_JSONL,
    build_examples,
    next_batch_indices,
    prepare_rows,
)


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual LoRA + rank-32 SVD recompression.")
    parser.add_argument("--model-path", default=env_str("MODEL_PATH", BASE_MODEL_PATH))
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--source-debug-jsonl", default=env_str("SOURCE_DEBUG_JSONL", PREVIOUS_EVAL_JSONL))
    parser.add_argument("--drift-debug-jsonl", default=env_str("DRIFT_DEBUG_JSONL", DEFAULT_DRIFT_DEBUG_JSONL))
    parser.add_argument("--output-dir", default=env_str("RESIDUAL_OUTPUT_DIR", "/home/ubuntu/additonal_tuning/outputs/sft_residual_lora_svd"))
    parser.add_argument("--prepared-jsonl", default=env_str("RESIDUAL_PREPARED_JSONL", ""))
    parser.add_argument("--resample", type=str2bool, default=env_bool("RESAMPLE", False))
    parser.add_argument("--dry-run", type=str2bool, default=env_bool("DRY_RUN", False))
    parser.add_argument("--merge-only", type=str2bool, default=env_bool("MERGE_ONLY", False))
    parser.add_argument("--residual-adapter-dir", default=env_str("RESIDUAL_ADAPTER_DIR", ""))

    parser.add_argument("--anchor-ids", default=env_str("RESIDUAL_ANCHOR_IDS", "hk_3302f383,hk_7283eb09,hk_c095f799-p0,my_bit_manipulation_00628,my_bit_manipulation_00933,my_equation_numeric_00239"))
    parser.add_argument("--wrong-repeat", type=int, default=env_int("RESIDUAL_WRONG_REPEAT", 2))
    parser.add_argument("--bit-wrong-repeat", type=int, default=env_int("RESIDUAL_BIT_WRONG_REPEAT", 2))
    parser.add_argument("--anchor-repeat", type=int, default=env_int("RESIDUAL_ANCHOR_REPEAT", 2))
    parser.add_argument("--equation-replay-count", type=int, default=env_int("RESIDUAL_EQUATION_REPLAY_COUNT", 12))
    parser.add_argument("--bit-replay-count", type=int, default=env_int("RESIDUAL_BIT_REPLAY_COUNT", 16))
    parser.add_argument("--replay-repeat", type=int, default=env_int("RESIDUAL_REPLAY_REPEAT", 1))
    parser.add_argument("--include-drift-no-boxed", type=str2bool, default=env_bool("RESIDUAL_INCLUDE_DRIFT_NO_BOXED", True))
    parser.add_argument("--boxed-tail-weight", type=float, default=env_float("RESIDUAL_BOXED_TAIL_WEIGHT", 1.0))
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

    parser.add_argument("--epochs", type=int, default=env_int("RESIDUAL_EPOCHS", 1))
    parser.add_argument("--num-steps", type=int, default=env_int("RESIDUAL_NUM_STEPS", 4))
    parser.add_argument("--batch-size", type=int, default=env_int("RESIDUAL_BATCH_SIZE", 2))
    parser.add_argument("--micro-batch-size", type=int, default=env_int("RESIDUAL_MICRO_BATCH_SIZE", 1))
    parser.add_argument("--learning-rate", type=float, default=env_float("RESIDUAL_LR", 1e-7))
    parser.add_argument("--weight-decay", type=float, default=env_float("RESIDUAL_WEIGHT_DECAY", 0.0))
    parser.add_argument("--max-grad-norm", type=float, default=env_float("RESIDUAL_MAX_GRAD_NORM", 0.05))
    parser.add_argument("--logging-steps", type=int, default=env_int("RESIDUAL_LOGGING_STEPS", 1))
    parser.add_argument("--shuffle", type=str2bool, default=env_bool("RESIDUAL_SHUFFLE", True))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def activate_residual_adapter(model, stack: dict, args: argparse.Namespace) -> None:
    target_modules = parse_csv(args.residual_target_modules)
    cfg = stack["LoraConfig"](
        r=args.residual_rank,
        lora_alpha=args.residual_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model.add_adapter("residual", cfg)
    model.base_model.set_adapter(["default", "residual"], inference_mode=False)

    trainable = Counter()
    trainable_count = 0
    for name, param in model.named_parameters():
        keep = ".lora_" in name and ".residual." in name
        param.requires_grad_(keep)
        if keep:
            param.data = param.data.float()
            trainable_count += 1
            for module_name in target_modules:
                if f".{module_name}." in name:
                    trainable[module_name] += param.numel()
                    break
    log(f"Residual adapter active with frozen default adapter. Trainable tensors: {trainable_count}")
    for name, count in sorted(trainable.items()):
        log(f"  residual {name}: {count:,} params")


def normalize_adapter_key(key: str) -> str:
    key = key.replace("base_model.model.backbone.", "base_model.model.model.")
    key = key.replace(".lora_A.residual.weight", ".lora_A.weight")
    key = key.replace(".lora_B.residual.weight", ".lora_B.weight")
    key = key.replace(".lora_A.default.weight", ".lora_A.weight")
    key = key.replace(".lora_B.default.weight", ".lora_B.weight")
    return key


def save_residual_adapter(model, output_dir: Path, stack: dict, args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key, value in model.state_dict().items():
        if ".residual.weight" not in key or ".lora_" not in key:
            continue
        tensors[normalize_adapter_key(key)] = value.detach().cpu().float()
    if not tensors:
        raise RuntimeError("No residual LoRA tensors found to save.")
    stack["save_file"](tensors, output_dir / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": args.residual_alpha,
        "lora_dropout": 0.0,
        "r": args.residual_rank,
        "target_modules": parse_csv(args.residual_target_modules),
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved residual adapter: {output_dir} ({len(tensors)} tensors)")
    return output_dir


def load_lora_scale(config_path: Path, default_rank: int) -> float:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(cfg.get("r", default_rank))
    alpha = float(cfg.get("lora_alpha", rank))
    return alpha / max(rank, 1)


def compressed_sum_pair(torch, old_a, old_b, res_a, res_b, old_scale: float, res_scale: float, target_rank: int):
    old_a = old_a.float()
    old_b = old_b.float()
    res_a = res_a.float()
    res_b = res_b.float()
    old_factor = math.sqrt(max(old_scale, 0.0))
    res_factor = math.sqrt(max(res_scale, 0.0))
    b_cat = torch.cat([old_b * old_factor, res_b * res_factor], dim=1)
    a_cat = torch.cat([old_a * old_factor, res_a * res_factor], dim=0)
    q_b, r_b = torch.linalg.qr(b_cat, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_cat.t(), mode="reduced")
    core = r_b @ r_a.t()
    u, s, vh = torch.linalg.svd(core, full_matrices=False)
    keep = min(target_rank, s.numel())
    sqrt_s = torch.sqrt(s[:keep].clamp_min(0.0))
    new_b = (q_b @ (u[:, :keep] * sqrt_s.unsqueeze(0))).contiguous()
    new_a = ((sqrt_s.unsqueeze(1) * vh[:keep, :]) @ q_a.t()).contiguous()
    if keep < target_rank:
        pad_b = torch.zeros(new_b.shape[0], target_rank - keep, dtype=new_b.dtype)
        pad_a = torch.zeros(target_rank - keep, new_a.shape[1], dtype=new_a.dtype)
        new_b = torch.cat([new_b, pad_b], dim=1)
        new_a = torch.cat([new_a, pad_a], dim=0)
    return new_a, new_b


def copy_adapter_sidecars(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.name == "adapter_model.safetensors":
            continue
        target = dst / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def merge_residual_to_rank32(initial_adapter: str | Path, residual_adapter: str | Path, output_dir: str | Path, target_rank: int, residual_scale: float, zip_submission: bool) -> Path:
    stack = import_training_stack()
    torch = stack["torch"]
    load_file = stack["load_file"]
    save_file = stack["save_file"]
    initial_adapter = Path(initial_adapter)
    residual_adapter = Path(residual_adapter)
    output_dir = Path(output_dir)
    copy_adapter_sidecars(initial_adapter, output_dir)

    old = load_file(initial_adapter / "adapter_model.safetensors")
    res_raw = load_file(residual_adapter / "adapter_model.safetensors")
    res = {normalize_adapter_key(k): v for k, v in res_raw.items()}
    old_scale = load_lora_scale(initial_adapter / "adapter_config.json", target_rank)
    res_base_scale = load_lora_scale(residual_adapter / "adapter_config.json", 1)
    effective_res_scale = residual_scale * res_base_scale

    merged = {k: v.detach().cpu().float() for k, v in old.items()}
    merged_pairs = 0
    for a_key, res_a in sorted(res.items()):
        if not a_key.endswith(".lora_A.weight"):
            continue
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if a_key not in old or b_key not in old or b_key not in res:
            log(f"WARNING: skipping unmatched residual tensor pair: {a_key}")
            continue
        new_a, new_b = compressed_sum_pair(
            torch,
            old[a_key],
            old[b_key],
            res_a,
            res[b_key],
            old_scale=old_scale,
            res_scale=effective_res_scale,
            target_rank=target_rank,
        )
        merged[a_key] = new_a
        merged[b_key] = new_b
        merged_pairs += 1

    cfg_path = output_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["r"] = target_rank
    cfg["lora_alpha"] = target_rank
    cfg["inference_mode"] = True
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    save_file(merged, output_dir / "adapter_model.safetensors")
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "residual_lora_svd_recompress",
        "initial_adapter": str(initial_adapter),
        "residual_adapter": str(residual_adapter),
        "target_rank": target_rank,
        "old_scale": old_scale,
        "residual_base_scale": res_base_scale,
        "user_residual_scale": residual_scale,
        "effective_residual_scale": effective_res_scale,
        "merged_pairs": merged_pairs,
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Merged residual into rank-{target_rank} adapter: {output_dir} (pairs={merged_pairs}, scale={residual_scale})")
    if zip_submission:
        log(f"Submission zip: {zip_adapter(output_dir)}")
    return output_dir


def save_run_metadata(args: argparse.Namespace, output_dir: Path, rows: list[dict], examples: list[dict], num_steps: int) -> None:
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "method": "residual_lora_train_then_svd",
        "initial_adapter": args.initial_adapter,
        "rows": len(rows),
        "examples": len(examples),
        "num_steps": num_steps,
        "residual_rank": args.residual_rank,
        "residual_alpha": args.residual_alpha,
        "residual_target_modules": parse_csv(args.residual_target_modules),
        "residual_scale": args.residual_scale,
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
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def train_residual(args: argparse.Namespace, output_dir: Path) -> Path:
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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No residual trainable parameters.")
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
        residual_dir = train_residual(args, output_dir)
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
