#!/usr/bin/env python3
# fast_ortholora_nemotron.py
# ------------------------------------------------------------
# Fast Ortho-LoRA training loop for Nemotron-style SFT.
#
# No Trainer.
# No SFTTrainer.
# Uses:
#   - Unsloth FastLanguageModel
#   - pre-tokenized token/mask training
#   - Cut Cross Entropy, no logits materialization
#   - Nemotron-H Mamba fast-path patch
#   - task-wise Ortho-LoRA gradient projection
#   - LoRA adapter checkpoint saving
#
# Recommended usage:
#   1) group tasks into 2 groups first, e.g. weak/rest
#   2) use tokenized data if possible
#   3) use active_tasks_per_step=2
# ------------------------------------------------------------

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


# ============================================================
# Config / utilities
# ============================================================

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "in_proj",
    "out_proj",
    "lm_head",
]

DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.environ.get(
    "ORTHOLORA_OUTPUT_DIR",
    "/home/ubuntu/Experiment_Output",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def parse_target_modules(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return DEFAULT_TARGET_MODULES
    return [x.strip() for x in text.split(",") if x.strip()]


def is_lora_param(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name or "lora_A" in name or "lora_B" in name


def is_trainable_lora_param(name: str, p: torch.nn.Parameter) -> bool:
    return p.requires_grad and p.grad is not None and is_lora_param(name)


def find_base_causal_lm(model):
    """
    Unwrap PEFT/Unsloth layers until the actual causal LM object.
    For Nemotron-H in the uploaded notebook, the resulting object has:
      - .backbone
      - .lm_head
    """
    base = model
    seen = set()
    while hasattr(base, "model") and id(base) not in seen:
        seen.add(id(base))
        base = base.model
    return base


# ============================================================
# Data
# ============================================================

def read_task_map_csv(path: str, id_col: str, task_col: str) -> Dict[str, str]:
    if not path:
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if id_col in row and task_col in row:
                out[str(row[id_col])] = str(row[task_col])
    return out


def read_task_map_json(path: str) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): str(v) for k, v in obj.items()}


def infer_task(
    rec: Dict[str, Any],
    problem_id: str,
    task_field: str,
    task_map: Dict[str, str],
    default_task: str,
) -> str:
    if task_field in rec and rec[task_field] not in (None, ""):
        return str(rec[task_field])
    if problem_id in task_map:
        return str(task_map[problem_id])
    if default_task:
        return default_task
    raise ValueError(
        f"Cannot infer task for problem_id={problem_id}. "
        f"Provide `{task_field}` in data, --task_map_csv, --task_map_json, or --default_task."
    )


def truncate_tokens_and_mask(tokens: List[int], mask: List[float], max_seq_len: int) -> Tuple[List[int], List[float]]:
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
        mask = mask[:max_seq_len]
    return tokens, mask


def make_example_from_tokens(
    problem_id: str,
    task: str,
    tokens: List[int],
    mask: List[float],
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    tokens, mask = truncate_tokens_and_mask(tokens, mask, max_seq_len)

    if len(tokens) < 2:
        return None
    if len(mask) != len(tokens):
        raise ValueError(f"mask length != tokens length for {problem_id}: {len(mask)} != {len(tokens)}")

    # The model predicts tokens[1:] from tokens[:-1].
    inputs = tokens[:-1]
    targets = tokens[1:]
    weights = [float(x) for x in mask[1:]]

    if not any(w > 0 for w in weights):
        return None

    return {
        "problem_id": problem_id,
        "task": task,
        "tokens": inputs,
        "targets": targets,
        "weights": weights,
    }


def load_tokenized_jsonl(
    path: str,
    task_field: str,
    task_map: Dict[str, str],
    default_task: str,
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    examples = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            problem_id = str(rec.get("problem_id", rec.get("id", line_idx)))
            task = infer_task(rec, problem_id, task_field, task_map, default_task)
            ex = make_example_from_tokens(
                problem_id=problem_id,
                task=task,
                tokens=list(map(int, rec["tokens"])),
                mask=[float(x) for x in rec["mask"]],
                max_seq_len=max_seq_len,
            )
            if ex is None:
                skipped += 1
                continue
            if "category" in rec and rec["category"] not in (None, ""):
                ex["category"] = str(rec["category"])
            examples.append(ex)

    log(f"[{now()}] Loaded tokenized jsonl: {len(examples)} examples, skipped={skipped}")
    return examples


def load_tokenized_corpus_dir(
    corpus_path: str,
    order_path: str,
    task_field: str,
    task_map: Dict[str, str],
    default_task: str,
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    """
    Supports the notebook-style layout:
      CORPUS_PATH/<problem_id>/synthetic.json
    where synthetic.json contains:
      {"tokens": [...], "mask": [...]}

    If order_path is provided, it replays problem_id order from index.jsonl.
    Otherwise it scans subdirectories.
    """
    corpus_path = str(corpus_path)
    problem_ids: List[str] = []

    if order_path:
        seen = set()
        with open(order_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("epoch", 0) != 0:
                    continue
                pid = str(rec["problem_id"])
                if pid in seen:
                    continue
                seen.add(pid)
                problem_ids.append(pid)
        log(f"[{now()}] Loaded {len(problem_ids)} problem ids from order_path={order_path}")
    else:
        for p in sorted(Path(corpus_path).iterdir()):
            if p.is_dir() and (p / "synthetic.json").exists():
                problem_ids.append(p.name)
        log(f"[{now()}] Scanned {len(problem_ids)} problem dirs from corpus_path={corpus_path}")

    examples = []
    skipped = 0

    for pid in problem_ids:
        seg_path = os.path.join(corpus_path, pid, "synthetic.json")
        if not os.path.isfile(seg_path):
            skipped += 1
            continue

        with open(seg_path, "r", encoding="utf-8") as f:
            rec = json.load(f)

        task = infer_task(rec, pid, task_field, task_map, default_task)
        ex = make_example_from_tokens(
            problem_id=pid,
            task=task,
            tokens=list(map(int, rec["tokens"])),
            mask=[float(x) for x in rec["mask"]],
            max_seq_len=max_seq_len,
        )
        if ex is None:
            skipped += 1
            continue
        if "category" in rec and rec["category"] not in (None, ""):
            ex["category"] = str(rec["category"])
        examples.append(ex)

    log(f"[{now()}] Loaded tokenized corpus dir: {len(examples)} examples, skipped={skipped}")
    return examples


def tokenize_prompt_response_jsonl(
    path: str,
    tokenizer,
    task_field: str,
    task_map: Dict[str, str],
    default_task: str,
    max_seq_len: int,
    prompt_field: str,
    response_field: str,
    text_field: str,
) -> List[Dict[str, Any]]:
    """
    Slower than pre-tokenized corpus, but tokenization is still done only once
    before the training loop.
    """
    examples = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            problem_id = str(rec.get("problem_id", rec.get("id", line_idx)))
            task = infer_task(rec, problem_id, task_field, task_map, default_task)

            if prompt_field in rec and response_field in rec:
                prompt = str(rec[prompt_field])
                response = str(rec[response_field])
                if tokenizer.eos_token and not response.endswith(tokenizer.eos_token):
                    response = response + tokenizer.eos_token

                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                tokens = prompt_ids + response_ids
                mask = [0.0] * len(prompt_ids) + [1.0] * len(response_ids)

            elif text_field in rec:
                text = str(rec[text_field])
                if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
                    text = text + tokenizer.eos_token
                tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
                mask = [1.0] * len(tokens)

            else:
                raise ValueError(
                    f"Each jsonl record must have ({prompt_field}, {response_field}) "
                    f"or {text_field}."
                )

            ex = make_example_from_tokens(
                problem_id=problem_id,
                task=task,
                tokens=tokens,
                mask=mask,
                max_seq_len=max_seq_len,
            )
            if ex is None:
                skipped += 1
                continue
            if "category" in rec and rec["category"] not in (None, ""):
                ex["category"] = str(rec["category"])
            examples.append(ex)

    log(f"[{now()}] Tokenized prompt/response jsonl: {len(examples)} examples, skipped={skipped}")
    return examples


def build_task_buckets(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        buckets.setdefault(ex["task"], []).append(ex)

    if len(buckets) < 2:
        raise ValueError(
            f"Need at least 2 task groups for Ortho-LoRA, got {list(buckets)}. "
            f"Use weak/rest grouping if detailed categories are expensive."
        )

    total_tokens = sum(len(e["tokens"]) for e in examples)
    total_weight = sum(sum(e["weights"]) for e in examples)

    log(f"[{now()}] Dataset summary: examples={len(examples)}, tokens={total_tokens:,}, unmasked={total_weight:,.0f}")
    for task, xs in sorted(buckets.items()):
        toks = sum(len(e["tokens"]) for e in xs)
        wts = sum(sum(e["weights"]) for e in xs)
        log(f"  - {task}: examples={len(xs):,}, tokens={toks:,}, unmasked={wts:,.0f}")

    return buckets


def write_dataset_summary(examples: List[Dict[str, Any]], output_dir: str) -> None:
    task_counts = Counter(str(e["task"]) for e in examples)
    category_counts = Counter(str(e.get("category", "unknown")) for e in examples)
    task_category_counts: Dict[str, Counter] = defaultdict(Counter)
    task_lengths: Dict[str, List[int]] = defaultdict(list)

    for ex in examples:
        task = str(ex["task"])
        category = str(ex.get("category", "unknown"))
        task_category_counts[task][category] += 1
        task_lengths[task].append(len(ex["tokens"]))

    summary = {
        "num_examples": len(examples),
        "task_counts": dict(sorted(task_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "task_category_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_category_counts.items())
        },
        "task_token_lengths": {
            task: {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(sum(lengths) / len(lengths), 2),
            }
            for task, lengths in sorted(task_lengths.items())
            if lengths
        },
        "sample_problem_ids": {
            task: [str(e["problem_id"]) for e in examples if str(e["task"]) == task][:5]
            for task in sorted(task_counts)
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dataset_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[{now()}] Wrote dataset summary to {path}")


def make_balanced_indices(task_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[int]]:
    indices = {}
    for task, xs in task_buckets.items():
        ids = list(range(len(xs)))
        random.shuffle(ids)
        indices[task] = ids
    return indices


class TaskSampler:
    """
    Fast cyclic sampler. Avoids DataLoader overhead and keeps examples in memory.
    """

    def __init__(self, task_buckets: Dict[str, List[Dict[str, Any]]], shuffle: bool = True):
        self.task_buckets = task_buckets
        self.shuffle = shuffle
        self.ptr: Dict[str, int] = {t: 0 for t in task_buckets}
        self.indices = make_balanced_indices(task_buckets)

    def sample(self, task: str, n: int) -> List[Dict[str, Any]]:
        xs = self.task_buckets[task]
        ids = self.indices[task]
        out = []

        while len(out) < n:
            if self.ptr[task] >= len(ids):
                self.ptr[task] = 0
                if self.shuffle:
                    random.shuffle(ids)

            out.append(xs[ids[self.ptr[task]]])
            self.ptr[task] += 1

        return out


def choose_active_tasks(
    all_tasks: List[str],
    active_tasks_per_step: int,
    step: int,
    mode: str,
) -> List[str]:
    if active_tasks_per_step <= 0 or active_tasks_per_step >= len(all_tasks):
        return list(all_tasks)

    if mode == "random":
        return random.sample(all_tasks, active_tasks_per_step)

    # round-robin window
    out = []
    for i in range(active_tasks_per_step):
        out.append(all_tasks[(step + i) % len(all_tasks)])
    return out


# ============================================================
# Batching
# ============================================================

def make_padded_batch(examples: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    n = len(examples)
    max_len = max(len(e["tokens"]) for e in examples)

    input_ids = torch.zeros(n, max_len, dtype=torch.long, device=device)
    targets = torch.zeros(n, max_len, dtype=torch.long, device=device)
    weights = torch.zeros(n, max_len, dtype=torch.float32, device=device)
    attention_mask = torch.zeros(n, max_len, dtype=torch.long, device=device)

    for i, e in enumerate(examples):
        seq_len = len(e["tokens"])
        # Creating CPU tensors then copying is often okay; direct device tensor creation from list also works.
        input_ids[i, :seq_len] = torch.tensor(e["tokens"], dtype=torch.long, device=device)
        targets[i, :seq_len] = torch.tensor(e["targets"], dtype=torch.long, device=device)
        weights[i, :seq_len] = torch.tensor(e["weights"], dtype=torch.float32, device=device)
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": targets,
        "weights": weights,
        "attention_mask": attention_mask,
    }


# ============================================================
# Model speed patches
# ============================================================

def load_fast_model(args):
    from unsloth import FastLanguageModel

    log(f"[{now()}] Loading model with Unsloth: {args.model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
    )

    return model, tokenizer


def attach_lora(model, args):
    from unsloth import FastLanguageModel

    target_modules = parse_target_modules(args.target_modules)
    log(f"[{now()}] Attaching LoRA: r={args.lora_r}, alpha={args.lora_alpha}, targets={target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=args.seed,
    )

    FastLanguageModel.for_training(model)
    return model


def patch_nemotron_fast_path() -> None:
    """
    Notebook speed trick:
    force modeling_nemotron_h.is_fast_path_available = True after checking kernels.
    """
    try:
        import causal_conv1d
        import mamba_ssm
        from causal_conv1d import causal_conv1d_fn

        cc = torch.cuda.get_device_capability(0)
        log(f"[{now()}] GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
        log(f"[{now()}] torch={torch.__version__}, cuda={torch.version.cuda}")
        log(f"[{now()}] mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}")

        x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
        causal_conv1d_fn(x, w, None, activation="silu")
        log(f"[{now()}] causal_conv1d CUDA kernel: OK")
    except Exception as e:
        log(f"[{now()}] WARNING: could not verify causal_conv1d/mamba kernels: {repr(e)}")

    nemotron_mod = None
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            nemotron_mod = mod
            break

    if nemotron_mod is None:
        log(f"[{now()}] WARNING: modeling_nemotron_h module not found. Skipping fast-path patch.")
        return

    log(f"[{now()}] is_fast_path_available was: {nemotron_mod.is_fast_path_available}")
    nemotron_mod.is_fast_path_available = True
    log(f"[{now()}] Patched is_fast_path_available = True")


def maybe_add_lm_head_lora(model, args) -> None:
    """
    The uploaded notebook manually adds lm_head LoRA because Unsloth may drop it for MoE.
    """
    if not args.add_lm_head_lora:
        return

    try:
        from peft import LoraConfig
        from peft.tuners.lora import Linear as LoraLinear
    except Exception as e:
        log(f"[{now()}] WARNING: cannot import PEFT LoraLinear: {repr(e)}")
        return

    causal_lm = find_base_causal_lm(model)
    if not hasattr(causal_lm, "lm_head"):
        log(f"[{now()}] WARNING: base model has no lm_head. Skipping manual lm_head LoRA.")
        return

    lm_head = causal_lm.lm_head
    if isinstance(lm_head, LoraLinear):
        log(f"[{now()}] lm_head already has LoRA")
        return

    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    try:
        model.base_model._create_and_replace(
            cfg,
            "default",
            target=lm_head,
            target_name="lm_head",
            parent=causal_lm,
        )
        log(f"[{now()}] Manually added LoRA to lm_head")
    except Exception as e:
        log(f"[{now()}] WARNING: failed to manually add lm_head LoRA: {repr(e)}")


def cast_lora_to_fp32(model) -> None:
    for name, p in model.named_parameters():
        if ".lora_" in name or "lora_A" in name or "lora_B" in name:
            p.data = p.data.to(torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"[{now()}] Model params: trainable={trainable:,} / total={total:,} ({100 * trainable / total:.4f}%)")
    log(f"[{now()}] LoRA params cast to fp32")


def patch_forward_with_cut_cross_entropy(model) -> bool:
    """
    Replace CausalLM forward so it does not materialize [B, T, vocab] logits.
    This mirrors the uploaded notebook:
      hidden_states = backbone(...)
      lm_weight = base_lm_head + LoRA_B @ LoRA_A
      per_token_ce = linear_cross_entropy(hidden_states, lm_weight, labels, reduction="none")
      model._cached_per_token_ce = per_token_ce

    Returns True if patch is active.
    """
    try:
        from cut_cross_entropy import linear_cross_entropy
    except Exception as e:
        log(f"[{now()}] WARNING: cut_cross_entropy import failed: {repr(e)}")
        return False

    base = find_base_causal_lm(model)

    if not hasattr(base, "backbone") or not hasattr(base, "lm_head"):
        log(f"[{now()}] WARNING: model has no .backbone/.lm_head. CCE patch skipped.")
        return False

    def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        backbone_out = base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("position_ids", "past_key_values", "use_cache")
            },
        )
        hidden_states = backbone_out[0]

        lm_head = base.lm_head

        # This path assumes lm_head is PEFT LoRA-wrapped.
        if hasattr(lm_head, "base_layer") and hasattr(lm_head, "lora_A") and hasattr(lm_head, "lora_B"):
            base_w = lm_head.base_layer.weight
            lora_A = lm_head.lora_A["default"].weight
            lora_B = lm_head.lora_B["default"].weight
            scaling = lm_head.scaling["default"]
            lm_weight = base_w + scaling * (lora_B @ lora_A)
        else:
            lm_weight = lm_head.weight

        if labels is not None:
            per_token_ce = linear_cross_entropy(
                hidden_states,
                lm_weight,
                labels,
                reduction="none",
            )
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None

        model._cached_per_token_ce = per_token_ce
        return loss

    base.forward = _patched_causal_forward
    log(f"[{now()}] Patched CausalLM.forward with Cut Cross Entropy")
    return True


def maybe_load_adapter_weights(model, adapter_path: str) -> None:
    if not adapter_path:
        return

    log(f"[{now()}] Loading adapter weights from {adapter_path}")

    from peft import load_peft_weights

    adapter_weights = load_peft_weights(adapter_path)
    model_sd = model.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    loaded = 0

    for ak, av in adapter_weights.items():
        candidates = [
            ak,
            ak.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"),
            ak.replace(".backbone.lm_head.", ".lm_head."),
            ak.replace(".backbone.lm_head.", ".lm_head.")
              .replace(".lora_A.weight", ".lora_A.default.weight")
              .replace(".lora_B.weight", ".lora_B.default.weight"),
        ]

        for ck in candidates:
            if ck in model_sd:
                new_sd[ck] = av
                loaded += 1
                break

    model.load_state_dict(new_sd, strict=False)

    if loaded != len(adapter_weights):
        raise RuntimeError(f"Not all adapter weights loaded: {loaded}/{len(adapter_weights)}")

    log(f"[{now()}] Loaded {loaded}/{len(adapter_weights)} adapter tensors")


def maybe_freeze_non_in_proj_lora(model, in_proj_only: bool) -> None:
    if not in_proj_only:
        return

    for name, p in model.named_parameters():
        if p.requires_grad and ".in_proj." not in name:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[{now()}] IN_PROJ_ONLY=True. Trainable params now: {trainable:,}")


# ============================================================
# MoE expert LoRA tying
# ============================================================

def build_moe_tie_param_names(model) -> List[str]:
    """
    Notebook convention:
      gate_up_proj / up_proj / w1 / gate_proj -> tie A
      down_proj / w2                         -> tie B

    This keeps 128 expert slices identical.
    """
    tied_names: List[str] = []
    w1_proj_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
    w2_proj_names = ("down_proj", ".w2.")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".experts." not in name or ".lora_" not in name:
            continue

        is_w1 = any(s in name for s in w1_proj_names)
        is_w2 = any(s in name for s in w2_proj_names)
        is_A = ".lora_A." in name
        is_B = ".lora_B." in name

        should_tie = (is_w1 and is_A) or (is_w2 and is_B)
        if not should_tie:
            continue

        if p.dim() < 2 or p.shape[0] <= 1:
            continue

        tied_names.append(name)

    return tied_names


@torch.no_grad()
def tie_param_init(model, tied_names: List[str]) -> None:
    if not tied_names:
        return
    name_to_param = dict(model.named_parameters())
    for name in tied_names:
        p = name_to_param[name]
        mean = p.data.mean(dim=0, keepdim=True)
        p.data.copy_(mean.expand_as(p.data))


@torch.no_grad()
def tie_current_grads(model, tied_names: List[str]) -> None:
    """
    Sum, not mean, matching the uploaded notebook.
    """
    if not tied_names:
        return
    name_to_param = dict(model.named_parameters())
    for name in tied_names:
        p = name_to_param[name]
        if p.grad is None:
            continue
        grad_sum = p.grad.sum(dim=0, keepdim=True)
        p.grad.copy_(grad_sum.expand_as(p.grad))


# ============================================================
# Ortho-LoRA gradient projection
# ============================================================

@torch.no_grad()
def capture_lora_grads(model) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if is_trainable_lora_param(name, p):
            grads[name] = p.grad.detach().clone()
    if not grads:
        raise RuntimeError("No LoRA gradients captured. Check target_modules and requires_grad flags.")
    return grads


@torch.no_grad()
def orthogonal_project_grads(
    task_grads: Dict[str, Dict[str, torch.Tensor]],
    eps: float,
    random_projection_order: bool,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
    """
    PCGrad/Ortho-LoRA:
      if dot(g_i, g_j) < 0:
          g_i = g_i - dot(g_i, g_j) / ||g_j||^2 * g_j

    Applied per LoRA tensor. Since PEFT stores lora_A and lora_B separately,
    this naturally decouples A-space and B-space gradients.
    """
    tasks = list(task_grads.keys())
    projected = {
        t: {name: g.clone() for name, g in grads.items()}
        for t, grads in task_grads.items()
    }

    names = sorted({name for grads in task_grads.values() for name in grads.keys()})

    n_pairs = 0
    n_conflicts = 0
    cosine_sum = 0.0

    for name in names:
        active = [t for t in tasks if name in projected[t]]
        if random_projection_order:
            random.shuffle(active)

        for ti in active:
            others = [t for t in active if t != ti]
            if random_projection_order:
                random.shuffle(others)

            for tj in others:
                gi = projected[ti][name]
                gj = projected[tj][name]

                gi_f = gi.float()
                gj_f = gj.float()

                dot = torch.sum(gi_f * gj_f)
                ni = torch.sum(gi_f * gi_f).clamp_min(eps)
                nj = torch.sum(gj_f * gj_f).clamp_min(eps)
                cos = dot / torch.sqrt(ni * nj)

                n_pairs += 1
                cosine_sum += float(cos.detach().cpu())

                if dot < 0:
                    n_conflicts += 1
                    corrected = gi_f - (dot / nj) * gj_f
                    projected[ti][name] = corrected.to(dtype=gi.dtype, device=gi.device)

    stats = {
        "pairs": float(n_pairs),
        "conflicts": float(n_conflicts),
        "conflict_rate": float(n_conflicts / n_pairs) if n_pairs else 0.0,
        "mean_cosine": float(cosine_sum / n_pairs) if n_pairs else 0.0,
    }
    return projected, stats


@torch.no_grad()
def apply_projected_grads(
    model,
    projected: Dict[str, Dict[str, torch.Tensor]],
    average_task_grad: bool,
) -> None:
    for p in model.parameters():
        p.grad = None

    task_names = list(projected.keys())
    denom = max(1, len(task_names)) if average_task_grad else 1

    name_to_param = dict(model.named_parameters())
    names = sorted({name for grads in projected.values() for name in grads.keys()})

    for name in names:
        if name not in name_to_param:
            continue

        p = name_to_param[name]
        if not p.requires_grad:
            continue

        g_sum = None
        for task in task_names:
            if name not in projected[task]:
                continue
            g = projected[task][name].to(device=p.device)
            g_sum = g if g_sum is None else g_sum + g

        if g_sum is None:
            continue

        p.grad = (g_sum / denom).to(dtype=p.dtype, device=p.device)


# ============================================================
# Training
# ============================================================

def run_task_backward(
    model,
    examples: List[Dict[str, Any]],
    micro_batch_size: int,
    device: torch.device,
    use_cce: bool,
) -> Tuple[float, float]:
    """
    Backward one task batch with micro-batch accumulation.
    Returns (loss_sum, weight_sum).
    """
    n = len(examples)
    n_accum = math.ceil(n / micro_batch_size)
    total_loss_sum = 0.0
    total_weight_sum = 0.0

    for mb_start in range(0, n, micro_batch_size):
        mb = examples[mb_start : mb_start + micro_batch_size]
        batch = make_padded_batch(mb, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            if use_cce:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                per_token_ce = model._cached_per_token_ce
                weighted_loss = per_token_ce * batch["weights"]
                weight_sum = batch["weights"].sum()
                loss_sum = weighted_loss.sum()
                loss = loss_sum / weight_sum if weight_sum > 0 else loss_sum * 0.0
            else:
                # Fallback path materializes logits. Slower.
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = out.loss
                weight_sum = batch["weights"].sum()
                loss_sum = loss.detach() * weight_sum

        (loss / n_accum).backward()

        total_loss_sum += float(loss_sum.detach().cpu())
        total_weight_sum += float(weight_sum.detach().cpu())

        del batch, loss

    return total_loss_sum, total_weight_sum


def save_adapter(model, tokenizer, save_dir: str, zip_submission: bool) -> None:
    from safetensors.torch import load_file, save_file

    os.makedirs(save_dir, exist_ok=True)

    for fname in os.listdir(save_dir):
        if fname.startswith("adapter"):
            os.remove(os.path.join(save_dir, fname))

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Nemotron competition key compatibility from uploaded notebook.
    st_path = os.path.join(save_dir, "adapter_model.safetensors")
    if os.path.exists(st_path):
        tensors = load_file(st_path)
        renamed = {
            k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
            for k, v in tensors.items()
        }
        save_file(renamed, st_path)

    if zip_submission:
        zip_path = os.path.join(save_dir, "submission.zip")
        adapter_files = [f for f in os.listdir(save_dir) if f.startswith("adapter")]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in adapter_files:
                zf.write(os.path.join(save_dir, fname), fname)
        log(f"[{now()}] Wrote {zip_path}")

    log(f"[{now()}] Saved adapter to {save_dir}")


def train(args) -> None:
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this fast script.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")

    # ------------------------------------------------------------
    # Load model first when prompt/response data needs tokenizer.
    # ------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = load_fast_model(args)
    model = attach_lora(model, args)

    patch_nemotron_fast_path()
    maybe_add_lm_head_lora(model, args)
    cast_lora_to_fp32(model)
    maybe_load_adapter_weights(model, args.resume_adapter)
    maybe_freeze_non_in_proj_lora(model, args.in_proj_only)

    use_cce = False
    if args.use_cce:
        use_cce = patch_forward_with_cut_cross_entropy(model)

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    task_map: Dict[str, str] = {}
    task_map.update(read_task_map_json(args.task_map_json))
    task_map.update(read_task_map_csv(args.task_map_csv, args.task_map_id_col, args.task_map_task_col))

    if args.tokenized_jsonl:
        examples = load_tokenized_jsonl(
            path=args.tokenized_jsonl,
            task_field=args.task_field,
            task_map=task_map,
            default_task=args.default_task,
            max_seq_len=args.max_seq_len,
        )
    elif args.corpus_path:
        examples = load_tokenized_corpus_dir(
            corpus_path=args.corpus_path,
            order_path=args.order_path,
            task_field=args.task_field,
            task_map=task_map,
            default_task=args.default_task,
            max_seq_len=args.max_seq_len,
        )
    elif args.train_jsonl:
        examples = tokenize_prompt_response_jsonl(
            path=args.train_jsonl,
            tokenizer=tokenizer,
            task_field=args.task_field,
            task_map=task_map,
            default_task=args.default_task,
            max_seq_len=args.max_seq_len,
            prompt_field=args.prompt_field,
            response_field=args.response_field,
            text_field=args.text_field,
        )
    else:
        raise ValueError("Provide one of --tokenized_jsonl, --corpus_path, or --train_jsonl")

    os.makedirs(args.output_dir, exist_ok=True)
    write_dataset_summary(examples, args.output_dir)

    buckets = build_task_buckets(examples)
    all_tasks = sorted(buckets.keys())
    sampler = TaskSampler(buckets, shuffle=args.shuffle_dataset)

    tied_names: List[str] = []
    if args.moe_tie_weights:
        tied_names = build_moe_tie_param_names(model)
        log(f"[{now()}] MoE tied LoRA tensors: {len(tied_names)}")
        if tied_names:
            log(f"[{now()}] Example tied tensor: {tied_names[0]}")
            tie_param_init(model, tied_names)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    model.train()
    gc.collect()
    torch.cuda.empty_cache()

    log(
        f"[{now()}] Training start: steps={args.num_steps}, "
        f"task_batch_size={args.task_batch_size}, micro_batch_size={args.micro_batch_size}, "
        f"active_tasks_per_step={args.active_tasks_per_step}, use_cce={use_cce}"
    )

    train_log: List[str] = []
    start_time = time.time()

    for step in range(args.num_steps):
        step_t0 = time.time()

        active_tasks = choose_active_tasks(
            all_tasks=all_tasks,
            active_tasks_per_step=args.active_tasks_per_step,
            step=step,
            mode=args.task_sampling,
        )

        task_grads: Dict[str, Dict[str, torch.Tensor]] = {}
        task_loss_mean: Dict[str, float] = {}

        # --------------------------------------------------------
        # Ortho-LoRA: task-wise gradient calculation.
        # This is the unavoidable cost. Keep active_tasks small.
        # --------------------------------------------------------
        for task in active_tasks:
            optimizer.zero_grad(set_to_none=True)

            task_examples = sampler.sample(task, args.task_batch_size)
            loss_sum, weight_sum = run_task_backward(
                model=model,
                examples=task_examples,
                micro_batch_size=args.micro_batch_size,
                device=device,
                use_cce=use_cce,
            )

            if tied_names:
                tie_current_grads(model, tied_names)

            if args.task_weights_json:
                # optional multiplicative task weight
                # Applied after backward by scaling captured gradients.
                pass

            grads = capture_lora_grads(model)

            task_grads[task] = grads
            task_loss_mean[task] = loss_sum / weight_sum if weight_sum > 0 else 0.0

        # Optional task weights.
        if args.task_weights_json:
            with open(args.task_weights_json, "r", encoding="utf-8") as f:
                task_weights = json.load(f)
            for task, grads in task_grads.items():
                w = float(task_weights.get(task, 1.0))
                if w != 1.0:
                    for name in grads:
                        grads[name].mul_(w)

        projected, stats = orthogonal_project_grads(
            task_grads,
            eps=args.projection_eps,
            random_projection_order=args.random_projection_order,
        )

        apply_projected_grads(
            model=model,
            projected=projected,
            average_task_grad=args.average_task_grad,
        )

        if tied_names:
            tie_current_grads(model, tied_names)

        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm,
            )
        else:
            grad_norm = torch.tensor(0.0)

        # Fast linear decay like uploaded notebook.
        if args.lr_schedule == "linear_decay":
            lr = args.learning_rate * (1.0 - step / max(1, args.num_steps))
        else:
            lr = args.learning_rate

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Keep tied params numerically tied after optimizer step.
        if tied_names:
            tie_param_init(model, tied_names)

        if (step + 1) % args.logging_steps == 0 or step == 0:
            loss_msg = " | ".join(f"{t}={task_loss_mean[t]:.5f}" for t in active_tasks)
            elapsed = time.time() - start_time
            step_wall = time.time() - step_t0
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9

            msg = (
                f"[{now()}] step={step + 1}/{args.num_steps} "
                f"loss[{loss_msg}] "
                f"conflict_rate={stats['conflict_rate']:.4f} "
                f"mean_cos={stats['mean_cosine']:.4f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"lr={lr:.2e} "
                f"wall={step_wall:.1f}s "
                f"mem={mem_gb:.1f}GB peak={peak_gb:.1f}GB elapsed={elapsed/60:.1f}m"
            )
            log(msg)
            train_log.append(msg)

        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step + 1}")
            save_adapter(model, tokenizer, ckpt_dir, zip_submission=False)

    final_dir = os.path.join(args.output_dir, "final_adapter")
    save_adapter(model, tokenizer, final_dir, zip_submission=args.zip_submission)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_log) + "\n")

    log(f"[{now()}] Training complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume_adapter", type=str, default="")

    # Data input modes
    p.add_argument("--tokenized_jsonl", type=str, default="")
    p.add_argument("--corpus_path", type=str, default="")
    p.add_argument("--order_path", type=str, default="")
    p.add_argument("--train_jsonl", type=str, default="")

    # Data fields
    p.add_argument("--task_field", type=str, default="task")
    p.add_argument("--prompt_field", type=str, default="prompt")
    p.add_argument("--response_field", type=str, default="response")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--default_task", type=str, default="")

    # task mapping if tokenized corpus lacks task field
    p.add_argument("--task_map_csv", type=str, default="")
    p.add_argument("--task_map_json", type=str, default="")
    p.add_argument("--task_map_id_col", type=str, default="id")
    p.add_argument("--task_map_task_col", type=str, default="task")
    p.add_argument("--task_weights_json", type=str, default="")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--target_modules", type=str, default=",".join(DEFAULT_TARGET_MODULES))
    p.add_argument("--add_lm_head_lora", type=str2bool, default=True)
    p.add_argument("--in_proj_only", type=str2bool, default=False)
    p.add_argument("--moe_tie_weights", type=str2bool, default=True)

    # Speed
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--attn_implementation", type=str, default="eager")
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    p.add_argument("--use_cce", type=str2bool, default=True)

    # Training
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--task_batch_size", type=int, default=16)
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--active_tasks_per_step", type=int, default=2)
    p.add_argument("--task_sampling", type=str, default="round_robin", choices=["round_robin", "random"])
    p.add_argument("--shuffle_dataset", type=str2bool, default=False)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_schedule", type=str, default="linear_decay", choices=["linear_decay", "constant"])
    p.add_argument("--max_grad_norm", type=float, default=1e9)

    # Ortho-LoRA
    p.add_argument("--projection_eps", type=float, default=1e-12)
    p.add_argument("--average_task_grad", type=str2bool, default=True)
    p.add_argument("--random_projection_order", type=str2bool, default=True)

    # Logging/checkpointing
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=env_int("SAVE_STEPS", 100))
    p.add_argument("--zip_submission", type=str2bool, default=True)

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
