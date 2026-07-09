#!/usr/bin/env python3
"""Shared utilities for narrow post-SFT tuning.

The goal of these scripts is not to relearn the whole benchmark. They start
from the existing submission_1 LoRA adapter and apply a small, guarded update
focused on the observed equation_numeric symbolic branch-map failures.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_MODEL_PATH = (
    "/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
DATASET_ROOT = "/home/ubuntu/dataset/merged_sft_dataset"
SUBMISSION_1_ROOT = "/home/ubuntu/FinetunedAdapter(Authrozied)/submission_1"
PREVIOUS_EVAL_JSONL = (
    "/home/ubuntu/evaluator/results/"
    "authorized_submission_1_vllm_60pc_20260517_170748/debug_predictions.jsonl"
)
BIT_REPLAY_ANCHOR_IDS = ("hk_3302f383", "hk_c095f799-p0")

COMPETITION_MAX_LORA_RANK = 32
COMPETITION_MAX_TOKENS = 7680
COMPETITION_TOP_P = 1.0
COMPETITION_TEMPERATURE = 0.0
COMPETITION_MAX_NUM_SEQS = 64
COMPETITION_GPU_MEMORY_UTILIZATION = 0.85
COMPETITION_MAX_MODEL_LEN = 8192

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

SYSTEM_PROMPT = "Solve the problem step by step. Put the final answer inside \\boxed{answer}."
COMPETITION_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{% if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(msg, flush=True)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value in (None, "") else str2bool(value)


def env_path(name: str, default: str | Path) -> Path:
    value = os.environ.get(name)
    return Path(default if value in (None, "") else value).expanduser()


def set_basic_env() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_final_answer(text: str | None) -> str:
    r"""Competition-style final answer extraction."""
    if text is None:
        return "NOT_FOUND"

    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for i, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return (non_empty[-1] if non_empty else matches[-1].strip())

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            return found[-1].strip()

    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    if found:
        return found[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def is_numeric_answer(answer: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(answer).strip()))


def discover_adapter(root: str | Path) -> Path:
    root = Path(root).expanduser()
    if (root / "adapter_config.json").exists():
        return root
    adapters = sorted(root.glob("**/adapter_config.json"))
    adapters = [p.parent for p in adapters if (p.parent / "adapter_model.safetensors").exists()]
    if not adapters:
        raise FileNotFoundError(f"No adapter_config.json + adapter_model.safetensors found under {root}")
    return adapters[0]


def load_token_dataset(dataset_root: str | Path) -> list[dict[str, Any]]:
    dataset_root = Path(dataset_root).expanduser()
    token_dir = dataset_root / "tokens" if (dataset_root / "tokens").is_dir() else dataset_root
    index_path = token_dir.parent / "logprobs" / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index: {index_path}")

    rows: list[dict[str, Any]] = []
    for meta in load_jsonl(index_path):
        pid = str(meta["problem_id"])
        token_path = token_dir / pid / "synthetic.json"
        if not token_path.exists():
            log(f"Skipping {pid}: missing {token_path}")
            continue
        token_row = json.loads(token_path.read_text(encoding="utf-8"))
        category = str(meta.get("category", "unknown"))
        rows.append(
            {
                "problem_id": pid,
                "category": category,
                "task": "bit" if category == "bit_manipulation" else "rest",
                "tokens": token_row["tokens"],
                "mask": token_row["mask"],
            }
        )
    return rows


def split_prompt_response(tokens: list[int], mask: list[float]) -> tuple[list[int], list[int]]:
    split = next((i for i, value in enumerate(mask) if float(value) > 0), len(mask))
    return tokens[:split], tokens[split:]


def extract_user_prompt(decoded_prompt: str) -> str:
    matches = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", decoded_prompt, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return re.sub(r"<\|[^>]+?\|>", "", decoded_prompt).strip()


def annotate_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        tokens = [int(x) for x in row["tokens"]]
        mask = [float(x) for x in row["mask"]]
        prompt_tokens, response_tokens = split_prompt_response(tokens, mask)
        decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=False)
        answer = extract_final_answer(decoded_response).strip()
        pid = str(row["problem_id"])
        out = dict(row)
        out.update(
            {
                "prompt": extract_user_prompt(decoded_prompt),
                "answer": answer,
                "is_numeric_answer": is_numeric_answer(answer),
                "is_symbolic_answer": not is_numeric_answer(answer),
                "is_my_equation": pid.startswith("my_equation_numeric_"),
            }
        )
        annotated.append(out)
    return annotated


def previous_eval_records(path: str | Path = PREVIOUS_EVAL_JSONL) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        pid = str(row.get("problem_id", ""))
        if pid:
            records[pid] = row
    return records


def previous_wrong_equation_ids(path: str | Path = PREVIOUS_EVAL_JSONL) -> set[str]:
    wrong: set[str] = set()
    for row in previous_eval_records(path).values():
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 0:
            wrong.add(str(row.get("problem_id", "")))
    return wrong


def previous_correct_ids_by_category(path: str | Path = PREVIOUS_EVAL_JSONL) -> dict[str, set[str]]:
    correct: dict[str, set[str]] = defaultdict(set)
    for row in previous_eval_records(path).values():
        if int(row.get("exact_match", 0)) == 1:
            correct[str(row.get("category", "unknown"))].add(str(row.get("problem_id", "")))
    return correct


def sample_group_with_forced_ids(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    forced_ids: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    by_id = {str(row.get("problem_id", "")): row for row in rows}
    forced = [dict(by_id[pid]) for pid in forced_ids if pid in by_id]
    if len(forced) >= n:
        return forced[:n]
    forced_set = {str(row.get("problem_id", "")) for row in forced}
    pool = [row for row in rows if str(row.get("problem_id", "")) not in forced_set]
    sampled = forced + sample_group(pool, n - len(forced), rng, replace=False)
    rng.shuffle(sampled)
    return sampled


def sample_group(rows: list[dict[str, Any]], n: int, rng: random.Random, replace: bool = False) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    if replace:
        return [dict(rng.choice(rows)) for _ in range(n)]
    if n <= len(rows):
        return [dict(r) for r in rng.sample(rows, n)]
    out = [dict(r) for r in rows]
    out.extend(dict(rng.choice(rows)) for _ in range(n - len(rows)))
    return out


def stratified_replay_sample(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("category", "unknown"))].append(row)
    if not buckets or n <= 0:
        return []

    categories = sorted(buckets)
    base = n // len(categories)
    rem = n % len(categories)
    sampled: list[dict[str, Any]] = []
    for i, category in enumerate(categories):
        take = base + (1 if i < rem else 0)
        sampled.extend(sample_group(buckets[category], take, rng, replace=False))
    rng.shuffle(sampled)
    return sampled


@dataclass
class MixConfig:
    total_examples: int = 900
    hard_ratio: float = 0.50
    equation_replay_ratio: float = 0.30
    bit_replay_ratio: float = 0.20
    other_replay_ratio: float = 0.00
    seed: int = 42
    include_previous_wrong: bool = True
    correct_equation_replay_only: bool = True
    bit_anchor_ids: tuple[str, ...] = BIT_REPLAY_ANCHOR_IDS


def build_guarded_mix(
    annotated_rows: list[dict[str, Any]],
    cfg: MixConfig,
) -> list[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    eval_records = previous_eval_records() if cfg.include_previous_wrong else {}
    wrong_ids = {
        pid
        for pid, row in eval_records.items()
        if row.get("category") == "equation_numeric" and int(row.get("exact_match", 0)) == 0
    }
    correct_by_category = previous_correct_ids_by_category() if cfg.include_previous_wrong else {}

    hard: list[dict[str, Any]] = []
    eq_replay: list[dict[str, Any]] = []
    bit_replay: list[dict[str, Any]] = []
    other_replay: list[dict[str, Any]] = []

    for row in annotated_rows:
        pid = str(row["problem_id"])
        category = str(row["category"])
        is_hard_symbolic = (
            category == "equation_numeric"
            and bool(row.get("is_my_equation"))
            and bool(row.get("is_symbolic_answer"))
        )
        is_previous_wrong = category == "equation_numeric" and pid in wrong_ids

        if is_hard_symbolic or is_previous_wrong:
            hard.append(row)
        elif category == "equation_numeric":
            if not cfg.correct_equation_replay_only or not eval_records or pid in correct_by_category.get(category, set()):
                eq_replay.append(row)
        elif category == "bit_manipulation":
            bit_replay.append(row)
        else:
            other_replay.append(row)

    total = cfg.total_examples
    if total <= 0:
        selected = [dict(row) for row in hard + eq_replay + bit_replay + other_replay]
    else:
        n_hard = int(round(total * cfg.hard_ratio))
        n_eq = int(round(total * cfg.equation_replay_ratio))
        n_bit = int(round(total * cfg.bit_replay_ratio))
        n_other = max(0, total - n_hard - n_eq - n_bit)
        selected = []
        selected.extend(sample_group(hard, n_hard, rng, replace=False))
        selected.extend(sample_group(eq_replay, n_eq, rng, replace=False))
        selected.extend(sample_group_with_forced_ids(bit_replay, n_bit, rng, cfg.bit_anchor_ids))
        if n_other:
            selected.extend(stratified_replay_sample(other_replay, n_other, rng))

    for row in selected:
        category = str(row["category"])
        pid = str(row["problem_id"])
        if category == "equation_numeric" and (pid in wrong_ids or row.get("is_my_equation") and row.get("is_symbolic_answer")):
            row["mix_group"] = "targeted_equation_fix"
        elif category == "equation_numeric":
            row["mix_group"] = "equation_replay_correct"
        elif category == "bit_manipulation" and pid in set(cfg.bit_anchor_ids):
            row["mix_group"] = "bit_replay_anchor"
        elif category == "bit_manipulation":
            row["mix_group"] = "bit_replay"
        else:
            row["mix_group"] = "other_category_replay"
    rng.shuffle(selected)
    return selected


def competition_prompt(tokenizer: Any, prompt: str, system_prompt: str | None = None) -> str:
    user_content = str(prompt) + COMPETITION_SUFFIX
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception:
        prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
        return f"{prefix}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def make_sft_example(row: dict[str, Any], max_seq_len: int) -> dict[str, Any] | None:
    tokens = [int(x) for x in row["tokens"]]
    mask = [float(x) for x in row["mask"]]
    if len(tokens) != len(mask):
        raise ValueError(f"tokens/mask mismatch: {row.get('problem_id')}")
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
        mask = mask[:max_seq_len]
    if len(tokens) < 2:
        return None
    inputs = tokens[:-1]
    targets = tokens[1:]
    weights = [float(x) for x in mask[1:]]
    if not any(w > 0 for w in weights):
        return None
    return {
        "problem_id": row["problem_id"],
        "category": row["category"],
        "mix_group": row.get("mix_group", ""),
        "tokens": inputs,
        "targets": targets,
        "weights": weights,
        "length": len(inputs),
        "supervised": int(sum(1 for w in weights if w > 0)),
    }


def save_mix_report(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": len(rows),
        "by_category": Counter(str(row.get("category", "unknown")) for row in rows),
        "by_mix_group": Counter(str(row.get("mix_group", "unknown")) for row in rows),
        "numeric_vs_symbolic": Counter(
            "numeric" if row.get("is_numeric_answer") else "symbolic" for row in rows
        ),
    }
    with path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            if isinstance(value, Counter):
                f.write(f"{key}:\n")
                for name, count in sorted(value.items()):
                    f.write(f"  {name}: {count}\n")
            else:
                f.write(f"{key}: {value}\n")


def zip_adapter(adapter_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    adapter_dir = Path(adapter_dir)
    if zip_path is None:
        zip_path = adapter_dir / "submission.zip"
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in adapter_dir.rglob("*"):
            if path.is_file() and path.name != zip_path.name:
                zf.write(path, path.relative_to(adapter_dir))
    return zip_path


def reset_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_training_stack():
    """Import Unsloth first, then the rest of the training stack."""
    set_basic_env()
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch
    import numpy as np
    from cut_cross_entropy import linear_cross_entropy
    from peft import LoraConfig, load_peft_weights
    from peft.tuners.lora import Linear as LoraLinear
    from safetensors.torch import load_file, save_file

    return {
        "FastLanguageModel": FastLanguageModel,
        "torch": torch,
        "np": np,
        "linear_cross_entropy": linear_cross_entropy,
        "LoraConfig": LoraConfig,
        "load_peft_weights": load_peft_weights,
        "LoraLinear": LoraLinear,
        "load_file": load_file,
        "save_file": save_file,
    }


def find_base_causal_lm(model: Any) -> Any:
    base = model
    seen = set()
    while hasattr(base, "model") and id(base) not in seen:
        seen.add(id(base))
        base = base.model
    return base


def patch_fast_path_flag() -> None:
    import sys

    for name, module in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(module, "is_fast_path_available"):
            log(f"is_fast_path_available was: {module.is_fast_path_available}")
            module.is_fast_path_available = True  # type: ignore[attr-defined]
            log("Patched is_fast_path_available = True")
            return
    log("WARNING: could not find modeling_nemotron_h.is_fast_path_available to patch")


def patch_nemotron_moe_dtype() -> None:
    import sys
    import torch

    patched = 0
    for name, module in list(sys.modules.items()):
        if "modeling_nemotron_h" not in name:
            continue
        moe_cls = getattr(module, "NemotronHMOE", None)
        if moe_cls is None or getattr(moe_cls, "_additional_tuning_dtype_patch", False):
            continue

        def _patched_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
            final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
            expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
            expert_mask = expert_mask.permute(2, 0, 1)

            for expert_idx in range(len(self.experts)):
                expert = self.experts[expert_idx]
                mask = expert_mask[expert_idx]
                token_indices, weight_indices = torch.where(mask)

                if token_indices.numel() > 0:
                    expert_weights = topk_weights[token_indices, weight_indices]
                    expert_input = hidden_states[token_indices]
                    expert_output = expert(expert_input)
                    weighted_output = expert_output * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))
                else:
                    dummy_out = expert(torch.zeros_like(hidden_states[:1]))
                    final_hidden_states = final_hidden_states + dummy_out.to(final_hidden_states.dtype).sum() * 0.0

            return final_hidden_states.type(hidden_states.dtype)

        moe_cls.moe = _patched_moe
        moe_cls._additional_tuning_dtype_patch = True
        patched += 1

    if patched:
        log(f"Patched NemotronHMOE.moe dtype handling in {patched} module(s)")
    else:
        log("WARNING: could not find NemotronHMOE to patch dtype handling")


def ensure_lm_head_lora(model: Any, stack: dict[str, Any], rank: int, alpha: int) -> None:
    base = find_base_causal_lm(model)
    lm_head = base.lm_head
    if isinstance(lm_head, stack["LoraLinear"]):
        log("lm_head already has LoRA")
        return
    cfg = stack["LoraConfig"](r=rank, lora_alpha=alpha, lora_dropout=0.0)
    model.base_model._create_and_replace(
        cfg,
        "default",
        target=lm_head,
        target_name="lm_head",
        parent=base,
    )
    log("Manually added LoRA to lm_head")


def load_initial_adapter(model: Any, adapter_dir: str | Path, stack: dict[str, Any]) -> None:
    adapter_dir = Path(adapter_dir)
    log(f"Loading initial adapter: {adapter_dir}")
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        # Keep adapter tensors on CPU while the 30B base model is already on GPU.
        # peft.load_peft_weights may default to CUDA here and can OOM with a
        # 3GB rank-32 adapter plus vLLM/training remnants in memory.
        adapter_weights = stack["load_file"](safetensors_path, device="cpu")
    else:
        adapter_weights = stack["load_peft_weights"](str(adapter_dir), device="cpu")
    model_sd = model.state_dict()
    new_sd: dict[str, Any] = {}
    loaded = 0
    ignored = 0

    for ak, av in adapter_weights.items():
        if ak.endswith(".base_layer.weight"):
            ignored += 1
            continue
        candidates = [
            ak,
            ak.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"),
            ak.replace("base_model.model.model.", "base_model.model.backbone."),
            ak.replace("base_model.model.model.", "base_model.model.backbone.")
            .replace(".lora_A.weight", ".lora_A.default.weight")
            .replace(".lora_B.weight", ".lora_B.default.weight"),
            ak.replace(".backbone.lm_head.", ".lm_head.")
            .replace(".lora_A.weight", ".lora_A.default.weight")
            .replace(".lora_B.weight", ".lora_B.default.weight"),
        ]
        for key in candidates:
            if key in model_sd:
                new_sd[key] = av
                loaded += 1
                break

    missing = len(adapter_weights) - ignored - loaded
    if missing > 0:
        log(f"WARNING: adapter tensors loaded={loaded}, ignored={ignored}, missing={missing}")
    model.load_state_dict(new_sd, strict=False)
    log(f"Loaded {loaded} adapter tensors (ignored={ignored})")


def patch_cce_forward(model: Any, stack: dict[str, Any]) -> Any:
    base = find_base_causal_lm(model)
    if hasattr(base, "_additional_tuning_original_forward"):
        original_forward = base._additional_tuning_original_forward
        if getattr(base, "_additional_tuning_cce_installed", False):
            return original_forward
    else:
        original_forward = base.forward
        base._additional_tuning_original_forward = original_forward
    backbone = getattr(base, "backbone", None) or getattr(base, "model", None)
    if backbone is None:
        raise AttributeError(f"Could not find backbone/model on {type(base).__name__}")
    linear_cross_entropy = stack["linear_cross_entropy"]

    def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        backbone_out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k in ("position_ids", "past_key_values", "use_cache")},
        )
        hidden_states = backbone_out[0]
        lm_head = base.lm_head
        base_w = lm_head.base_layer.weight
        lora_A = lm_head.lora_A["default"].weight
        lora_B = lm_head.lora_B["default"].weight
        scaling = lm_head.scaling["default"]
        lm_weight = base_w + scaling * lora_B @ lora_A
        if labels is not None:
            per_token_ce = linear_cross_entropy(hidden_states, lm_weight, labels, reduction="none")
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None
        model._cached_per_token_ce = per_token_ce
        return loss

    base.forward = _patched_causal_forward
    base._additional_tuning_cce_installed = True
    log("Patched CausalLM.forward with cut-cross-entropy")
    return original_forward


def restore_forward(model: Any) -> None:
    base = find_base_causal_lm(model)
    original = getattr(base, "_additional_tuning_original_forward", None)
    if original is not None:
        base.forward = original
        base._additional_tuning_cce_installed = False


def save_adapter(model: Any, tokenizer: Any, output_dir: str | Path, stack: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    st_path = output_dir / "adapter_model.safetensors"
    if st_path.exists():
        tensors = stack["load_file"](st_path)
        renamed = {
            k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
            for k, v in tensors.items()
        }
        stack["save_file"](renamed, st_path)
    log(f"Saved adapter: {output_dir}")


def load_unsloth_lora_model(
    model_path: str | Path,
    initial_adapter: str | Path,
    *,
    max_seq_len: int,
    lora_rank: int,
    lora_alpha: int,
    target_modules: list[str],
    load_in_4bit: bool,
    gradient_checkpointing: str | bool,
    trust_remote_code: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    stack = import_training_stack()
    torch = stack["torch"]
    FastLanguageModel = stack["FastLanguageModel"]

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_seq_len,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=trust_remote_code,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    tokenizer.chat_template = CHATML_TEMPLATE

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=gradient_checkpointing,
        random_state=42,
    )
    FastLanguageModel.for_training(model)
    patch_fast_path_flag()
    patch_nemotron_moe_dtype()
    ensure_lm_head_lora(model, stack, lora_rank, lora_alpha)
    load_initial_adapter(model, initial_adapter, stack)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,}")
    return model, tokenizer, stack


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
