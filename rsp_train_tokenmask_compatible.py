#!/usr/bin/env python3
"""RSP train-only Nemotron LoRA entrypoint.

This script is deliberately train-only.  It builds a rank-32 token/mask-compatible
adapter from the RSP data package with two objectives:

1. weighted completion-only SFT over anchor_sft + decision_sft rows
2. weighted pairwise rule-selection preference loss over decision_preference rows

It never evaluates on the competition set and never submits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TOKENMASK_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
SUBMISSION_ALLOWED = False
EVALUATION_ALLOWED = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row at {path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows


def boxed_answer(text: str) -> str:
    matches = BOXED_RE.findall(str(text))
    return matches[-1].strip() if matches else ""


def normalize_prompt(prompt: str) -> str:
    prompt = str(prompt)
    return prompt if PROMPT_SUFFIX.strip() in prompt else prompt + PROMPT_SUFFIX


def normalize_completion(text: str, expected_answer: str) -> str:
    body = str(text).replace("<|im_end|>", "").rstrip()
    expected_answer = str(expected_answer).strip()
    if boxed_answer(body) != expected_answer:
        body = BOXED_RE.sub("", body).rstrip()
        body = f"{body}\n\\boxed{{{expected_answer}}}"
    if not body.endswith("<|im_end|>"):
        body = body.rstrip() + "<|im_end|>"
    return body


def tokenize_prompt(tokenizer: Any, prompt: str, enable_thinking: bool) -> list[int]:
    messages = [{"role": "user", "content": normalize_prompt(prompt)}]
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(value, list):
        return [int(item) for item in value]
    input_ids = value["input_ids"] if isinstance(value, dict) or hasattr(value, "__getitem__") else value.input_ids
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(item) for item in input_ids]


def sft_record(row: dict[str, Any], tokenizer: Any, max_seq_length: int, enable_thinking: bool) -> dict[str, Any] | None:
    answer = str(row["final_answer"]).strip()
    prompt_ids = tokenize_prompt(tokenizer, str(row["prompt"]), enable_thinking)
    completion = normalize_completion(str(row["completion"]), answer)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    input_ids = (prompt_ids + completion_ids)[:max_seq_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
    if all(label == -100 for label in labels):
        return None
    return {
        "id": str(row["id"]),
        "domain": str(row["domain"]),
        "row_type": str(row["row_type"]),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "sample_weight": float(row.get("sample_weight", 1.0)),
        "total_tokens": len(input_ids),
        "loss_tokens": sum(label != -100 for label in labels),
    }


def preference_record(
    row: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
    enable_thinking: bool,
) -> dict[str, Any] | None:
    prompt_ids = tokenize_prompt(tokenizer, str(row["prompt"]), enable_thinking)

    def encode_branch(key: str, answer_key: str) -> tuple[list[int], list[int]]:
        completion = normalize_completion(str(row[key]), str(row[answer_key]))
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        input_ids = (prompt_ids + completion_ids)[:max_seq_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
        return input_ids, labels

    chosen_ids, chosen_labels = encode_branch("chosen", "chosen_answer")
    rejected_ids, rejected_labels = encode_branch("rejected", "rejected_answer")
    if all(label == -100 for label in chosen_labels) or all(label == -100 for label in rejected_labels):
        return None
    return {
        "id": str(row["id"]),
        "domain": str(row["domain"]),
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": [1] * len(chosen_ids),
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": [1] * len(rejected_ids),
        "rejected_labels": rejected_labels,
        "sample_weight": float(row.get("sample_weight", 1.0)),
    }


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    output = Path.cwd() / "rsp_verification_for_training.json"
    cmd = [
        sys.executable,
        "verify_rsp_dataset.py",
        "--dataset-dir",
        str(dataset_dir),
        "--json-output",
        str(output),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"RSP dataset verification failed:\n{result.stdout}\n{result.stderr}")
    data = json.loads(output.read_text(encoding="utf-8"))
    if data.get("rsp_dataset_valid") is not True:
        raise RuntimeError(f"RSP dataset verification did not return valid=true: {data}")
    return data


def write_submission_zip(adapter_dir: Path, output_zip: Path) -> None:
    if SUBMISSION_ALLOWED:
        raise RuntimeError("SUBMISSION_ALLOWED must remain False in the train entrypoint")
    required = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapter_model.safetensors"
    if not required.is_file():
        raise FileNotFoundError(required)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    include_suffixes = {".json", ".safetensors", ".model", ".txt"}
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(adapter_dir.rglob("*")):
            if path.is_file() and path.suffix in include_suffixes:
                archive.write(path, path.relative_to(adapter_dir))


def quantile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = [int(row.get("total_tokens", 0)) for row in records]
    loss_tokens = [int(row.get("loss_tokens", 0)) for row in records]
    return {
        "rows": len(records),
        "domains": dict(Counter(row.get("domain") for row in records)),
        "row_types": dict(Counter(row.get("row_type") for row in records)),
        "total_tokens": {
            "sum": sum(total_tokens),
            "p50": quantile(total_tokens, 0.50),
            "p90": quantile(total_tokens, 0.90),
            "p99": quantile(total_tokens, 0.99),
            "max": max(total_tokens) if total_tokens else 0,
        },
        "loss_tokens": {
            "sum": sum(loss_tokens),
            "p50": quantile(loss_tokens, 0.50),
            "p90": quantile(loss_tokens, 0.90),
            "p99": quantile(loss_tokens, 0.99),
            "max": max(loss_tokens) if loss_tokens else 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RSP token/mask-compatible rank-32 LoRA trainer")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/rsp_dataset"))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/rsp_adapter"))
    parser.add_argument("--submission-zip", type=Path, default=Path("/kaggle/working/submission.zip"))
    parser.add_argument("--audit-json", type=Path, default=Path("/kaggle/working/rsp_training_audit.json"))
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--sft-learning-rate", type=float, default=1.6e-4)
    parser.add_argument("--preference-learning-rate", type=float, default=3.5e-5)
    parser.add_argument("--sft-epochs", type=float, default=1.0)
    parser.add_argument("--preference-epochs", type=float, default=0.35)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--preference-batch-size", type=int, default=1)
    parser.add_argument("--preference-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--simpo-beta", type=float, default=2.0)
    parser.add_argument("--simpo-gamma", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-4bit", action="store_true")
    parser.add_argument("--disable-thinking-template", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.lora_rank != 32 or args.lora_alpha != 32 or float(args.lora_dropout) != 0.0:
        raise ValueError("RSP contract requires rank32/alpha32/dropout0")
    if args.max_seq_length != 8192:
        raise ValueError("RSP contract requires max_seq_length=8192")
    if args.preference_epochs <= 0:
        raise ValueError("RSP requires a positive pairwise preference phase")

    verification = verify_dataset(args.dataset_dir)
    anchor_rows = read_jsonl(args.dataset_dir / "rsp_anchor_sft.jsonl")
    decision_rows = read_jsonl(args.dataset_dir / "rsp_decision_sft.jsonl")
    preference_rows = read_jsonl(args.dataset_dir / "rsp_decision_preferences.jsonl")

    random.Random(args.seed).shuffle(anchor_rows)
    random.Random(args.seed + 1).shuffle(decision_rows)
    random.Random(args.seed + 2).shuffle(preference_rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enable_thinking = not args.disable_thinking_template

    sft_rows = anchor_rows + decision_rows
    sft_records = [
        record
        for row in sft_rows
        for record in [sft_record(row, tokenizer, args.max_seq_length, enable_thinking)]
        if record is not None
    ]
    pref_records = [
        record
        for row in preference_rows
        for record in [preference_record(row, tokenizer, args.max_seq_length, enable_thinking)]
        if record is not None
    ]
    if len(sft_records) < 8500:
        raise RuntimeError(f"too few tokenized SFT records: {len(sft_records)}")
    if len(pref_records) < 1000:
        raise RuntimeError(f"too few tokenized preference records: {len(pref_records)}")

    audit = {
        "candidate_id": "rsp-rule-selection-post-training",
        "current_minimum_goal_achieved": "no",
        "submission_allowed": False,
        "evaluation_allowed": False,
        "dataset_verification": verification,
        "dataset_hashes": {
            "rsp_anchor_sft.jsonl": sha256(args.dataset_dir / "rsp_anchor_sft.jsonl"),
            "rsp_decision_sft.jsonl": sha256(args.dataset_dir / "rsp_decision_sft.jsonl"),
            "rsp_decision_preferences.jsonl": sha256(args.dataset_dir / "rsp_decision_preferences.jsonl"),
        },
        "adapter_contract": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": TOKENMASK_TARGET_MODULES,
            "max_seq_length": args.max_seq_length,
            "precision": "bf16",
        },
        "objective": {
            "phase_1": "weighted completion-only SFT",
            "phase_2": "weighted pairwise SimPO-style rule-selection preference",
            "simpo_beta": args.simpo_beta,
            "simpo_gamma": args.simpo_gamma,
        },
        "sft_summary": summarize_records(sft_records),
        "preference_summary": {
            "rows": len(pref_records),
            "domains": dict(Counter(row.get("domain") for row in pref_records)),
        },
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    def make_training_args(**kwargs: Any) -> Any:
        import inspect

        params = inspect.signature(TrainingArguments.__init__).parameters
        filtered = {key: value for key, value in kwargs.items() if key in params}
        return TrainingArguments(**filtered)

    quantization_config = None
    if args.enable_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # PRO 6000 should hold the BF16 LoRA train path on one GPU. `auto` can
    # conservatively dispatch modules to CPU/disk, which turns runtime problems
    # into slow or invalid mixed-placement failures.
    device_map: Any = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.enable_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TOKENMASK_TARGET_MODULES,
        ),
    )
    model.print_trainable_parameters()

    sft_dataset = Dataset.from_list(
        [
            {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
                "labels": row["labels"],
                "sample_weight": row["sample_weight"],
            }
            for row in sft_records
        ]
    )

    def pad_matrix(values: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(item) for item in values)
        return torch.tensor([item + [pad_value] * (max_len - len(item)) for item in values], dtype=torch.long)

    def sft_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_matrix([row["input_ids"] for row in batch], tokenizer.pad_token_id),
            "attention_mask": pad_matrix([row["attention_mask"] for row in batch], 0),
            "labels": pad_matrix([row["labels"] for row in batch], -100),
            "sample_weight": torch.tensor([float(row["sample_weight"]) for row in batch], dtype=torch.float32),
        }

    class WeightedSFTTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
            weights = inputs.pop("sample_weight").to(model.device).float()
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = inputs["labels"][:, 1:].contiguous()
            mask = labels.ne(-100)
            safe_labels = labels.masked_fill(~mask, 0)
            token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                safe_labels.view(-1),
                reduction="none",
            ).view(labels.size())
            per_row = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = (per_row * weights).sum() / weights.sum().clamp_min(1.0)
            return (loss, outputs) if return_outputs else loss

    sft_args = make_training_args(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.sft_learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit" if args.enable_4bit else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    WeightedSFTTrainer(model=model, args=sft_args, train_dataset=sft_dataset, data_collator=sft_collate).train()

    def pref_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "chosen_input_ids": pad_matrix([row["chosen_input_ids"] for row in batch], tokenizer.pad_token_id),
            "chosen_attention_mask": pad_matrix([row["chosen_attention_mask"] for row in batch], 0),
            "chosen_labels": pad_matrix([row["chosen_labels"] for row in batch], -100),
            "rejected_input_ids": pad_matrix([row["rejected_input_ids"] for row in batch], tokenizer.pad_token_id),
            "rejected_attention_mask": pad_matrix([row["rejected_attention_mask"] for row in batch], 0),
            "rejected_labels": pad_matrix([row["rejected_labels"] for row in batch], -100),
            "sample_weight": torch.tensor([float(row["sample_weight"]) for row in batch], dtype=torch.float32),
        }

    def average_logprob(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        mask = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~mask, 0)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    pref_loader = DataLoader(
        pref_records,
        batch_size=args.preference_batch_size,
        shuffle=True,
        collate_fn=pref_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.preference_learning_rate)
    model.train()
    pref_steps_per_epoch = math.ceil(len(pref_loader) / max(1, args.preference_gradient_accumulation_steps))
    pref_total_steps = max(1, int(pref_steps_per_epoch * args.preference_epochs))
    completed_steps = 0
    optimizer.zero_grad(set_to_none=True)
    while completed_steps < pref_total_steps:
        for step, batch in enumerate(pref_loader, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            chosen = average_logprob(batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"])
            rejected = average_logprob(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
            weights = batch["sample_weight"].float()
            margins = args.simpo_beta * (chosen - rejected) - args.simpo_gamma
            loss = (-F.logsigmoid(margins) * weights).sum() / weights.sum().clamp_min(1.0)
            loss = loss / args.preference_gradient_accumulation_steps
            loss.backward()
            if step % args.preference_gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                if completed_steps % 10 == 0:
                    print(f"preference_step={completed_steps}/{pref_total_steps} loss={float(loss.detach().cpu()):.6f}")
                if completed_steps >= pref_total_steps:
                    break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_submission_zip(args.output_dir, args.submission_zip)
    audit["artifact"] = {
        "adapter_dir": str(args.output_dir),
        "submission_zip": str(args.submission_zip),
        "submission_zip_sha256": sha256(args.submission_zip),
        "preference_optimizer_steps": completed_steps,
    }
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RSP train-only adapter saved: {args.output_dir}")
    print(f"RSP train-only submission.zip saved: {args.submission_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
