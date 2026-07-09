#!/usr/bin/env python3
"""Static verifier for the RSP train-only implementation.

This verifier intentionally does not execute GPU training.  It checks that the
training entrypoint matches the RSP method contract before GPU time is used.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
    "lm_head",
}
EXPECTED_RANK = 32
EXPECTED_ALPHA = 32
EXPECTED_DROPOUT = 0.0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_module(path: Path, errors: list[str]) -> ast.Module | None:
    try:
        return ast.parse(read_text(path), filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path}: syntax error: {exc}")
        return None


def literal_assignments(module: ast.Module) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return values


def has_class(module: ast.Module, class_name: str) -> bool:
    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(module))


def has_function(module: ast.Module, function_name: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in ast.walk(module))


def adapter_zip_summary(path: Path, errors: list[str]) -> dict[str, Any]:
    summary = {
        "adapter_zip_checked": False,
        "adapter_config_valid": False,
        "safetensors_header_valid": False,
        "lora_A_tensors": 0,
        "lora_B_tensors": 0,
    }
    if not path.exists():
        errors.append(f"missing adapter zip: {path}")
        return summary
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "adapter_config.json" not in names:
            errors.append("adapter zip missing adapter_config.json")
            return summary
        if "adapter_model.safetensors" not in names:
            errors.append("adapter zip missing adapter_model.safetensors")
            return summary
        summary["adapter_zip_checked"] = True
        config = json.loads(archive.read("adapter_config.json").decode("utf-8"))
        config_errors: list[str] = []
        if config.get("r") != EXPECTED_RANK:
            config_errors.append(f"rank expected {EXPECTED_RANK}, found {config.get('r')!r}")
        if config.get("lora_alpha") != EXPECTED_ALPHA:
            config_errors.append(f"alpha expected {EXPECTED_ALPHA}, found {config.get('lora_alpha')!r}")
        if float(config.get("lora_dropout", -1.0)) != EXPECTED_DROPOUT:
            config_errors.append(f"dropout expected {EXPECTED_DROPOUT}, found {config.get('lora_dropout')!r}")
        if set(map(str, config.get("target_modules", []))) != EXPECTED_TARGET_MODULES:
            config_errors.append("target_modules mismatch")
        errors.extend(config_errors)
        summary["adapter_config_valid"] = not config_errors
        raw = archive.read("adapter_model.safetensors")
        if len(raw) < 8:
            errors.append("safetensors file is too short")
            return summary
        header_len = int.from_bytes(raw[:8], "little")
        if header_len <= 0 or 8 + header_len > len(raw):
            errors.append("safetensors header length invalid")
            return summary
        header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
        tensor_errors: list[str] = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = meta.get("shape") if isinstance(meta, dict) else None
            if ".lora_A." in name and name.endswith(".weight"):
                summary["lora_A_tensors"] += 1
                if not isinstance(shape, list) or len(shape) != 2 or shape[0] != EXPECTED_RANK:
                    tensor_errors.append(f"{name} LoRA A rank mismatch: {shape}")
            if ".lora_B." in name and name.endswith(".weight"):
                summary["lora_B_tensors"] += 1
                if not isinstance(shape, list) or len(shape) != 2 or shape[1] != EXPECTED_RANK:
                    tensor_errors.append(f"{name} LoRA B rank mismatch: {shape}")
        if summary["lora_A_tensors"] == 0 or summary["lora_A_tensors"] != summary["lora_B_tensors"]:
            tensor_errors.append("LoRA A/B tensor counts must be nonzero and equal")
        errors.extend(tensor_errors)
        summary["safetensors_header_valid"] = not tensor_errors
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", type=Path, default=Path("rsp_train_huikang_compatible.py"))
    parser.add_argument("--dataset-verification", type=Path, default=Path("data/rsp_dataset/rsp_verification.json"))
    parser.add_argument("--adapter-zip", type=Path, help="Optional produced adapter zip to validate after GPU training")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    module = parse_module(args.train_script, errors)
    source = read_text(args.train_script) if args.train_script.exists() else ""
    assignments = literal_assignments(module) if module else {}

    if assignments.get("SUBMISSION_ALLOWED") is not False:
        errors.append("SUBMISSION_ALLOWED must be a literal False")
    if assignments.get("EVALUATION_ALLOWED") is not False:
        errors.append("EVALUATION_ALLOWED must be a literal False")
    if set(assignments.get("HUIKANG_TARGET_MODULES", [])) != EXPECTED_TARGET_MODULES:
        errors.append("HUIKANG_TARGET_MODULES must match the locked huikang-compatible module set")
    if module and not has_class(module, "WeightedSFTTrainer"):
        errors.append("missing WeightedSFTTrainer")
    if module and not has_function(module, "average_logprob"):
        errors.append("missing average_logprob preference scoring function")
    for required in [
        "rsp_decision_preferences.jsonl",
        "simpo_beta",
        "simpo_gamma",
        "preference_learning_rate",
        "preference_gradient_accumulation_steps",
        "F.logsigmoid",
        "chosen - rejected",
        "verify_rsp_dataset.py",
        "write_submission_zip",
    ]:
        if required not in source:
            errors.append(f"train script missing required token: {required}")
    forbidden_patterns = [
        r"kaggle\s+competitions\s+submit",
        r"competitions\.submit",
        r"submission_allowed\s*=\s*True",
        r"EVALUATION_ALLOWED\s*=\s*True",
        r"build_eval_kernel",
        r"verify_eval_output",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, source, flags=re.IGNORECASE):
            errors.append(f"forbidden train/eval/submission pattern present: {pattern}")

    dataset_valid = None
    dataset_counts: dict[str, Any] = {}
    if not args.dataset_verification.exists():
        errors.append(f"missing dataset verification file: {args.dataset_verification}")
    else:
        data = json.loads(args.dataset_verification.read_text(encoding="utf-8"))
        dataset_valid = data.get("rsp_dataset_valid")
        dataset_counts = data.get("counts", {})
        if dataset_valid is not True:
            errors.append("RSP dataset verification must be valid before training")
        if data.get("gpu_execution_allowed") is not False:
            errors.append("dataset verifier must remain fail-closed for GPU execution")
        if dataset_counts.get("anchor_sft", 0) < 7000:
            errors.append("anchor_sft count below RSP gate")
        if dataset_counts.get("decision_sft", 0) < 1000:
            errors.append("decision_sft count below RSP gate")
        if dataset_counts.get("decision_preferences", 0) < 1000:
            errors.append("decision_preferences count below RSP gate")

    adapter_summary = None
    if args.adapter_zip:
        adapter_errors: list[str] = []
        adapter_summary = adapter_zip_summary(args.adapter_zip, adapter_errors)
        errors.extend(adapter_errors)

    result = {
        "rsp_train_shell_valid": not errors,
        "current_minimum_goal_achieved": "no",
        "submission_allowed": False,
        "gpu_execution_allowed": not errors,
        "dataset_valid": dataset_valid,
        "dataset_counts": dataset_counts,
        "adapter_summary": adapter_summary,
        "errors": len(errors),
        "first_errors": errors[:50],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["rsp_train_shell_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
