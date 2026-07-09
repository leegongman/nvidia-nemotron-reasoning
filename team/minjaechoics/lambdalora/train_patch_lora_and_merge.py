#!/usr/bin/env python3
"""One-file patch LoRA pipeline for the Nemotron competition.

This script combines:
  1. optional CSV -> token/mask dataset retokenization,
  2. Unsloth or PEFT patch-LoRA training,
  3. frozen existing adapter A + trainable patch adapter B,
  4. A_final = A + lambda * B export,
  5. SVD rank-32 compression for max_lora_rank=32,
  6. saved-adapter rank checks and eval config manifest.

Default submission-safe behavior:
  - backend: unsloth
  - patch rank: 8
  - lambda values: 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50
  - merge mode: svd
  - svd rank: 32
  - rank check after every exported lambda adapter

Run:
  python train_patch_lora_rank32_pipeline.py \
    --existing-adapter "/home/ubuntu/FinetunedAdapter(Authrozied)/submission_1"
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "equation_numeric_patch_dataset" / "patch_decoded_prompts.csv"
DEFAULT_OUTPUT_DIR = Path("/lambdalora/output")
DEFAULT_DATASET_ROOT = DEFAULT_OUTPUT_DIR / "equation_numeric_patch_dataset_tokenized"
DEFAULT_EXISTING_ADAPTER = Path("/home/ubuntu/ubuntu/FinetunedAdapter(Authrozied)/submission_1")
DEFAULT_MODEL_PATH = Path(
    "/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
DEFAULT_LAMBDAS = "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50"
COMPETITION_EVAL_CONFIG = {
    "max_lora_rank": 32,
    "max_tokens": 7680,
    "top_p": 1.0,
    "temperature": 0.0,
    "max_num_seqs": 64,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
}
NEMOTRON_TARGET_MODULES = [
    "out_proj",
    "v_proj",
    "q_proj",
    "down_proj",
    "embed_tokens",
    "k_proj",
    "in_proj",
    "up_proj",
    "o_proj",
    "lm_head",
    "gate_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-file rank-32 patch LoRA pipeline.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--existing-adapter", type=Path, default=DEFAULT_EXISTING_ADAPTER)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["unsloth", "peft"], default="unsloth")
    parser.add_argument("--existing-name", default="existing")
    parser.add_argument("--patch-name", default="patch")
    parser.add_argument("--lambda-values", default=DEFAULT_LAMBDAS)
    parser.add_argument("--merge-mode", choices=["svd", "cat", "linear"], default="svd")
    parser.add_argument("--svd-rank", type=int, default=32)
    parser.add_argument("--patch-rank", type=int, default=8)
    parser.add_argument("--patch-alpha", type=int, default=0, help="Defaults to existing adapter alpha.")
    parser.add_argument("--patch-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="", help="Comma-separated override. Defaults to existing config.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--skip-too-long", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=112843)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--unsloth-force-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--retokenize", action="store_true", help="Force rebuilding tokenized dataset from --input-csv.")
    parser.add_argument("--check-adapter-dir", type=Path, default=None, help="Only check this adapter and exit.")
    parser.add_argument("--allow-rank-unsafe", action="store_true", help="Allow cat/linear or svd_rank > 32.")
    return parser.parse_args()


def resolve_workspace_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path.resolve()

    text = str(path)
    workspace_home = Path("/home/ubuntu/ubuntu")
    plain_home = "/home/ubuntu/"
    if text.startswith(plain_home):
        candidate = workspace_home / text[len(plain_home) :]
        if candidate.exists():
            print(f"Resolved {path} -> {candidate}")
            return candidate.resolve()
    return path


def discover_adapter_dir(root: Path) -> Path:
    root = resolve_workspace_path(root)
    if (root / "adapter_config.json").exists() and (root / "adapter_model.safetensors").exists():
        return root.resolve()

    matches = sorted(root.glob("**/adapter_config.json")) if root.exists() else []
    adapters = [path.parent for path in matches if (path.parent / "adapter_model.safetensors").exists()]
    if not adapters:
        raise FileNotFoundError(f"No adapter_config.json + adapter_model.safetensors found under: {root}")
    print(f"Discovered adapter under wrapper dir: {adapters[0]}")
    return adapters[0].resolve()


def normalize_paths(args: argparse.Namespace) -> None:
    args.model_path = resolve_workspace_path(args.model_path)
    args.input_csv = resolve_workspace_path(args.input_csv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.check_adapter_dir is not None:
        args.check_adapter_dir = discover_adapter_dir(args.check_adapter_dir)
    elif args.existing_adapter is not None:
        args.existing_adapter = discover_adapter_dir(args.existing_adapter)


def validate_required_paths(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {args.model_path}")
    token_root = args.dataset_root / "tokens"
    needs_csv = args.retokenize or not token_root.is_dir() or not any(token_root.glob("*/synthetic.json"))
    if needs_csv and not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found for tokenization: {args.input_csv}")


def maybe_fallback_backend(args: argparse.Namespace) -> None:
    if args.backend != "unsloth":
        return
    if importlib.util.find_spec("unsloth") is not None:
        return
    print("Unsloth is not importable in this Python environment; falling back to --backend peft.")
    args.backend = "peft"


def parse_lambda_values(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one lambda value is required")
    return values


def lambda_dir_name(value: float) -> str:
    normalized = f"{value:.4f}".rstrip("0").rstrip(".")
    return "lambda_" + normalized.replace("-", "m").replace(".", "p")


def validate_submission_merge_args(args: argparse.Namespace) -> None:
    rank_safe = args.merge_mode == "svd" and 0 < args.svd_rank <= COMPETITION_EVAL_CONFIG["max_lora_rank"]
    if rank_safe or args.allow_rank_unsafe:
        return
    raise ValueError(
        "Submission requires final adapter rank <= max_lora_rank=32. "
        "Use --merge-mode svd --svd-rank 32, or pass --allow-rank-unsafe for experiments."
    )


def find_subsequence(tokens: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(tokens):
        return -1
    first = needle[0]
    last_start = len(tokens) - len(needle)
    for start in range(last_start + 1):
        if tokens[start] == first and tokens[start : start + len(needle)] == needle:
            return start
    return -1


def encode_text(tokenizer, text: str, add_special_tokens: bool = False) -> list[int]:
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=add_special_tokens)]


def parse_int_list(raw: str, column_name: str) -> list[int]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{column_name} must be a JSON list")
    return [int(x) for x in value]


def mask_from_csv(row: dict[str, str], tokens: list[int]) -> list[int] | None:
    raw = row.get("mask", "")
    if not raw:
        return None
    mask = parse_int_list(raw, "mask")
    if len(mask) != len(tokens):
        return None
    return [1 if x else 0 for x in mask]


def mask_from_masked_text(tokenizer, row: dict[str, str], tokens: list[int]) -> list[int] | None:
    masked_text = row.get("masked_text", "")
    if not masked_text:
        return None
    masked_tokens = encode_text(tokenizer, masked_text)
    start = find_subsequence(tokens, masked_tokens)
    if start < 0:
        return None
    mask = [0] * len(tokens)
    for idx in range(start, start + len(masked_tokens)):
        mask[idx] = 1
    return mask


def mask_from_assistant_span(tokenizer, text: str, token_count: int) -> list[int] | None:
    marker = "<|im_start|>assistant\n"
    start_char = text.find(marker)
    if start_char < 0:
        return None
    prefix = text[: start_char + len(marker)]
    start = len(encode_text(tokenizer, prefix))
    if start > token_count:
        return None
    return [0] * start + [1] * (token_count - start)


def build_mask(tokenizer, row: dict[str, str], tokens: list[int]) -> list[int]:
    for builder in (
        lambda: mask_from_csv(row, tokens),
        lambda: mask_from_masked_text(tokenizer, row, tokens),
        lambda: mask_from_assistant_span(tokenizer, row.get("text", ""), len(tokens)),
    ):
        mask = builder()
        if mask is not None:
            return mask
    raise ValueError(f"Could not recover loss mask for {row.get('problem_id', '<missing-id>')}")


def retokenize_csv(input_csv: Path, output_root: Path, model_path: Path, trust_remote_code: bool) -> None:
    from transformers import AutoTokenizer

    input_csv = input_csv.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    token_root = output_root / "tokens"
    logprobs_dir = output_root / "logprobs"
    token_root.mkdir(parents=True, exist_ok=True)
    logprobs_dir.mkdir(parents=True, exist_ok=True)
    index_path = logprobs_dir / "index.jsonl"

    count = 0
    with input_csv.open("r", encoding="utf-8", newline="") as f, index_path.open("w", encoding="utf-8") as index_file:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = row.get("problem_id", "").strip() or f"row_{count:06d}"
            text = row.get("text", "")
            if not text:
                user_prompt = row.get("user_prompt", "")
                assistant_text = row.get("assistant_text") or row.get("masked_text", "")
                text = (
                    "<|im_start|>system\n<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant_text}"
                )

            tokens = encode_text(tokenizer, text)
            mask = build_mask(tokenizer, {**row, "text": text}, tokens)
            if len(tokens) != len(mask):
                raise ValueError(f"tokens/mask length mismatch for {problem_id}")

            out_dir = token_root / problem_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "synthetic.json").write_text(
                json.dumps({"tokens": tokens, "mask": mask}, ensure_ascii=False),
                encoding="utf-8",
            )
            index_file.write(
                json.dumps(
                    {
                        "epoch": int(row["epoch"]) if row.get("epoch") else 0,
                        "step": int(row["step"]) if row.get("step") else 0,
                        "problem_id": problem_id,
                        "segment": row.get("segment") or "synthetic.jsonl",
                        "category": row.get("category", ""),
                        "num_loss_tokens": sum(mask),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    print(f"Retokenized rows: {count}")
    print(f"Tokenized dataset root: {output_root}")


def ensure_tokenized_dataset(args: argparse.Namespace) -> None:
    token_root = args.dataset_root / "tokens"
    if args.retokenize or not token_root.is_dir() or not any(token_root.glob("*/synthetic.json")):
        print("Tokenized dataset missing or --retokenize set; retokenizing from CSV.")
        retokenize_csv(args.input_csv, args.dataset_root, args.model_path, args.trust_remote_code)
    else:
        print(f"Using existing tokenized dataset: {args.dataset_root}")


def iter_token_files(dataset_root: Path) -> Iterable[Path]:
    token_root = dataset_root / "tokens"
    if not token_root.is_dir():
        raise FileNotFoundError(f"Token root not found: {token_root}")
    index_path = dataset_root / "logprobs" / "index.jsonl"
    paths: list[Path] = []
    seen: set[Path] = set()
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                problem_id = str(row.get("problem_id", ""))
                if not problem_id:
                    continue
                path = token_root / problem_id / "synthetic.json"
                if path.exists() and path not in seen:
                    paths.append(path)
                    seen.add(path)
    paths.extend(path for path in sorted(token_root.glob("*/synthetic.json")) if path not in seen)
    if not paths:
        raise FileNotFoundError(f"No token files found under {token_root}")
    return paths


@dataclass
class DatasetStats:
    rows: int
    skipped_too_long: int
    max_seen_length: int


class TokenMaskDataset:
    def __init__(self, dataset_root: Path, max_length: int, skip_too_long: bool):
        import torch

        self.rows: list[dict[str, torch.Tensor]] = []
        skipped_too_long = 0
        max_seen_length = 0
        for path in iter_token_files(dataset_root):
            data = json.loads(path.read_text(encoding="utf-8"))
            tokens = [int(x) for x in data["tokens"]]
            mask = [1 if int(x) else 0 for x in data["mask"]]
            if len(tokens) != len(mask):
                raise ValueError(f"tokens/mask length mismatch in {path}")
            max_seen_length = max(max_seen_length, len(tokens))
            if len(tokens) > max_length:
                if skip_too_long:
                    skipped_too_long += 1
                    continue
                tokens = tokens[:max_length]
                mask = mask[:max_length]
            labels = [tok if keep else -100 for tok, keep in zip(tokens, mask)]
            self.rows.append(
                {
                    "input_ids": torch.tensor(tokens, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        self.stats = DatasetStats(len(self.rows), skipped_too_long, max_seen_length)
        if not self.rows:
            raise ValueError("No usable rows after length filtering")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class DataCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import torch

        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        input_ids = []
        labels = []
        attention_mask = []
        for row in features:
            ids = row["input_ids"]
            labs = row["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones((len(ids),), dtype=torch.long),
                        torch.zeros((pad_len,), dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


def supported_kwargs(cls, kwargs: dict) -> dict:
    params = set(inspect.signature(cls).parameters)
    return {key: value for key, value in kwargs.items() if key in params}


def clone_lora_config(reference_config, args: argparse.Namespace):
    from peft import LoraConfig

    alpha = args.patch_alpha or int(getattr(reference_config, "lora_alpha", 16))
    target_modules = config_target_modules(reference_config, args.target_modules)
    kwargs = {
        "r": args.patch_rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "lora_dropout": args.patch_dropout,
        "fan_in_fan_out": getattr(reference_config, "fan_in_fan_out", False),
        "bias": getattr(reference_config, "bias", "none"),
        "use_rslora": getattr(reference_config, "use_rslora", False),
        "modules_to_save": getattr(reference_config, "modules_to_save", None),
        "init_lora_weights": True,
        "rank_pattern": getattr(reference_config, "rank_pattern", {}),
        "alpha_pattern": getattr(reference_config, "alpha_pattern", {}),
        "use_dora": getattr(reference_config, "use_dora", False),
        "task_type": getattr(reference_config, "task_type", "CAUSAL_LM"),
    }
    return LoraConfig(**supported_kwargs(LoraConfig, kwargs))


def config_target_modules(existing_config, override: str) -> list[str]:
    if override.strip():
        return [x.strip() for x in override.split(",") if x.strip()]
    target_modules = getattr(existing_config, "target_modules", None)
    if target_modules:
        return sorted(target_modules) if isinstance(target_modules, set) else list(target_modules)
    return NEMOTRON_TARGET_MODULES


def set_adapters(model, adapter_names: list[str] | str) -> None:
    try:
        model.set_adapter(adapter_names)
        return
    except Exception as exc:
        last_exc = exc
    try:
        model.base_model.set_adapter(adapter_names)
        return
    except Exception:
        raise RuntimeError(
            "This PEFT version could not activate stacked adapters. "
            "Upgrade peft or use a version supporting set_adapter(['existing', 'patch'])."
        ) from last_exc


def freeze_all_except_patch(model, patch_name: str) -> tuple[int, int]:
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        is_patch_param = f".{patch_name}." in name or name.endswith(f".{patch_name}.weight")
        param.requires_grad = is_patch_param
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        examples = [name for name, _ in list(model.named_parameters())[:20]]
        raise RuntimeError(f"No trainable patch parameters found for {patch_name}. Sample names: {examples}")
    return trainable, total


def get_patch_name_after_unsloth(model, requested_name: str) -> str:
    if hasattr(model, "peft_config"):
        if requested_name in model.peft_config:
            return requested_name
        if "default" in model.peft_config:
            return "default"
        if len(model.peft_config) == 1:
            return next(iter(model.peft_config))
    return requested_name


def maybe_for_training(fast_language_model, model):
    if hasattr(fast_language_model, "for_training"):
        return fast_language_model.for_training(model)
    return model


def load_unsloth_model_and_adapters(args: argparse.Namespace):
    import torch
    from peft import PeftConfig
    from unsloth import FastLanguageModel

    existing_config = PeftConfig.from_pretrained(str(args.existing_adapter))
    target_modules = config_target_modules(existing_config, args.target_modules)
    patch_alpha = args.patch_alpha or int(getattr(existing_config, "lora_alpha", 16))

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use either --load-in-4bit or --load-in-8bit, not both.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model_path),
        max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False,
        trust_remote_code=args.trust_remote_code,
        unsloth_force_compile=args.unsloth_force_compile,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16 if args.bf16 else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_kwargs = {
        "r": args.patch_rank,
        "lora_alpha": patch_alpha,
        "lora_dropout": args.patch_dropout,
        "target_modules": target_modules,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": args.seed,
    }
    if "adapter_name" in inspect.signature(FastLanguageModel.get_peft_model).parameters:
        peft_kwargs["adapter_name"] = args.patch_name

    print("Creating trainable patch adapter B with Unsloth.")
    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    patch_adapter_name = get_patch_name_after_unsloth(model, args.patch_name)

    print(f"Loading existing adapter A frozen: {args.existing_adapter}")
    model.load_adapter(str(args.existing_adapter), adapter_name=args.existing_name, is_trainable=False)
    set_adapters(model, [args.existing_name, patch_adapter_name])
    model = maybe_for_training(FastLanguageModel, model)
    trainable, total = freeze_all_except_patch(model, patch_adapter_name)
    model.print_trainable_parameters()
    print(f"Trainable patch params: {trainable:,} / visible params: {total:,}")
    return model, tokenizer, patch_adapter_name


def load_peft_model_and_adapters(args: argparse.Namespace):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = PeftModel.from_pretrained(
        model,
        str(args.existing_adapter),
        adapter_name=args.existing_name,
        is_trainable=False,
    )
    patch_config = clone_lora_config(model.peft_config[args.existing_name], args)
    model.add_adapter(args.patch_name, patch_config)
    set_adapters(model, [args.existing_name, args.patch_name])
    trainable, total = freeze_all_except_patch(model, args.patch_name)
    model.print_trainable_parameters()
    print(f"Trainable patch params: {trainable:,} / visible params: {total:,}")
    return model, tokenizer, args.patch_name


def save_selected_adapter(model, output_dir: Path, adapter_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(output_dir), selected_adapters=[adapter_name])
    except TypeError:
        set_adapters(model, adapter_name)
        model.save_pretrained(str(output_dir))


def copy_tokenizer_files(model_path: Path, output_dir: Path) -> None:
    for name in [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]:
        src = model_path / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def adapter_rank(adapter_dir: Path) -> int:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {adapter_config_path}")
    config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    ranks = []
    if "r" in config:
        ranks.append(int(config["r"]))
    rank_pattern = config.get("rank_pattern") or {}
    if isinstance(rank_pattern, dict):
        ranks.extend(int(value) for value in rank_pattern.values())
    return max(ranks) if ranks else 0


def check_adapter(adapter_dir: Path) -> dict[str, object]:
    rank = adapter_rank(adapter_dir)
    max_lora_rank = COMPETITION_EVAL_CONFIG["max_lora_rank"]
    ok = 0 < rank <= max_lora_rank
    result = {
        "adapter_dir": str(adapter_dir),
        "adapter_rank": rank,
        "max_lora_rank": max_lora_rank,
        "rank_ok": ok,
        "eval_config": COMPETITION_EVAL_CONFIG,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not ok:
        raise ValueError(f"Adapter rank {rank} exceeds max_lora_rank {max_lora_rank}")
    return result


def export_lambda_adapters(model, args: argparse.Namespace, lambda_values: list[float], patch_adapter_name: str) -> None:
    output_dir = args.output_dir.resolve()
    patch_dir = output_dir / "patch_adapter_uncompressed"
    save_selected_adapter(model, patch_dir, patch_adapter_name)
    copy_tokenizer_files(args.model_path, patch_dir)

    manifest = {
        "existing_adapter": str(args.existing_adapter),
        "patch_adapter_uncompressed": str(patch_dir),
        "lambda_values": lambda_values,
        "merge_mode": args.merge_mode,
        "svd_rank": args.svd_rank if args.merge_mode == "svd" else None,
        "competition_eval_config": COMPETITION_EVAL_CONFIG,
        "submission_rank_safe": args.merge_mode == "svd"
        and 0 < args.svd_rank <= COMPETITION_EVAL_CONFIG["max_lora_rank"],
        "outputs": {},
        "checks": {},
        "notes": [
            "Adapter A was active and frozen during patch training.",
            "Patch adapter B was the only trainable adapter.",
            "Each lambda adapter is checked against max_lora_rank=32.",
        ],
    }

    for value in lambda_values:
        adapter_name = lambda_dir_name(value)
        kwargs = {
            "adapters": [args.existing_name, patch_adapter_name],
            "weights": [1.0, value],
            "adapter_name": adapter_name,
            "combination_type": args.merge_mode,
        }
        if args.merge_mode == "svd":
            kwargs["svd_rank"] = args.svd_rank
        model.add_weighted_adapter(**kwargs)
        set_adapters(model, adapter_name)
        lambda_dir = output_dir / adapter_name
        save_selected_adapter(model, lambda_dir, adapter_name)
        copy_tokenizer_files(args.model_path, lambda_dir)
        check = check_adapter(lambda_dir)
        manifest["outputs"][str(value)] = str(lambda_dir)
        manifest["checks"][str(value)] = check
        print(f"Wrote checked lambda adapter {value}: {lambda_dir}")

    (output_dir / "rank32_pipeline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_patch(args: argparse.Namespace):
    import torch
    from transformers import Trainer, TrainingArguments

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    ensure_tokenized_dataset(args)
    dataset = TokenMaskDataset(args.dataset_root.resolve(), args.max_length, args.skip_too_long)
    print(f"Dataset rows: {dataset.stats.rows}")
    print(f"Skipped too-long rows: {dataset.stats.skipped_too_long}")
    print(f"Max seen length: {dataset.stats.max_seen_length}")

    if args.backend == "unsloth":
        model, tokenizer, patch_adapter_name = load_unsloth_model_and_adapters(args)
    else:
        model, tokenizer, patch_adapter_name = load_peft_model_and_adapters(args)

    checkpoint_dir = args.output_dir / "trainer_checkpoints"
    if args.overwrite_output_dir and checkpoint_dir.exists():
        print(f"Clearing trainer checkpoint dir because --overwrite-output-dir is set: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    training_kwargs = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "optim": "adamw_8bit" if args.backend == "unsloth" or args.load_in_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
        "gradient_checkpointing": False if args.backend == "unsloth" else args.gradient_checkpointing,
    }
    training_args = TrainingArguments(**supported_kwargs(TrainingArguments, training_kwargs))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    return model, patch_adapter_name


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    normalize_paths(args)

    if args.check_adapter_dir is not None:
        check_adapter(args.check_adapter_dir)
        return

    validate_required_paths(args)
    if args.existing_adapter is None:
        raise ValueError("--existing-adapter is required unless --check-adapter-dir is used.")
    validate_submission_merge_args(args)
    maybe_fallback_backend(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lambda_values = parse_lambda_values(args.lambda_values)
    model, patch_adapter_name = train_patch(args)
    export_lambda_adapters(model, args, lambda_values, patch_adapter_name)
    print(f"Done. Submission-safe outputs are under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
