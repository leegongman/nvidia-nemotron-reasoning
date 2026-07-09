#!/usr/bin/env python3
"""Export rank-32 lambda adapters from a completed patch-training checkpoint.

This avoids PEFT's add_weighted_adapter path, which currently cannot combine
LoRA adapters that were trained through nn.Parameter targets on this model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


DEFAULT_CHECKPOINT = Path("/lambdalora/output/trainer_checkpoints/checkpoint-654")
DEFAULT_OUTPUT_DIR = Path("/lambdalora/output")
DEFAULT_MODEL_PATH = Path(
    "/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
DEFAULT_LAMBDAS = "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50"
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export lambda adapters from a saved patch checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lambda-values", default=DEFAULT_LAMBDAS)
    parser.add_argument("--svd-rank", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_lambda_values(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one lambda value is required")
    return values


def lambda_dir_name(value: float) -> str:
    normalized = f"{value:.4f}".rstrip("0").rstrip(".")
    return "lambda_" + normalized.replace("-", "m").replace(".", "p")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_sidecar_files(source_dirs: list[Path], output_dir: Path) -> None:
    for name in [
        "README.md",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]:
        for source_dir in source_dirs:
            source = source_dir / name
            if source.exists():
                shutil.copy2(source, output_dir / name)
                break


def pattern_value(config: dict, prefix: str, rank: int, key: str, default: int) -> int:
    pattern = config.get(key) or {}
    for pattern_key, value in pattern.items():
        if prefix.endswith(pattern_key) or f".{pattern_key}." in prefix or f".{pattern_key}.lora_" in prefix:
            return int(value)
    return int(config.get("lora_alpha" if key == "alpha_pattern" else "r", default))


def lora_scale(config: dict, prefix: str, rank: int) -> float:
    alpha = pattern_value(config, prefix, rank, "alpha_pattern", rank)
    configured_rank = pattern_value(config, prefix, rank, "rank_pattern", rank)
    if configured_rank != rank:
        configured_rank = rank
    return float(alpha) / float(configured_rank)


def lora_prefixes(keys: list[str]) -> list[str]:
    prefixes = []
    keyset = set(keys)
    for key in keys:
        if not key.endswith(LORA_A_SUFFIX):
            continue
        prefix = key[: -len(LORA_A_SUFFIX)]
        if prefix + LORA_B_SUFFIX not in keyset:
            raise KeyError(f"Missing matching lora_B for {key}")
        prefixes.append(prefix)
    return sorted(prefixes)


def low_rank_svd_product(
    existing_a: torch.Tensor,
    existing_b: torch.Tensor,
    existing_scale: float,
    patch_a: torch.Tensor,
    patch_b: torch.Tensor,
    patch_scale: float,
    lambda_value: float,
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.cat(
        [
            existing_b.float() * existing_scale,
            patch_b.float() * (patch_scale * lambda_value),
        ],
        dim=1,
    )
    right = torch.cat([existing_a.float(), patch_a.float()], dim=0)

    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)

    rank = min(target_rank, singular_values.numel())
    new_b = (q_left @ u_core[:, :rank]) * singular_values[:rank].unsqueeze(0)
    new_a = vh_core[:rank, :] @ q_right.T
    return new_a.contiguous(), new_b.contiguous()


def export_one_lambda(
    lambda_value: float,
    checkpoint_dir: Path,
    output_dir: Path,
    model_path: Path,
    target_rank: int,
    overwrite: bool,
) -> dict[str, object]:
    existing_dir = checkpoint_dir / "existing"
    existing_weights = existing_dir / "adapter_model.safetensors"
    patch_weights = checkpoint_dir / "adapter_model.safetensors"
    existing_config = load_json(existing_dir / "adapter_config.json")
    patch_config = load_json(checkpoint_dir / "adapter_config.json")

    adapter_name = lambda_dir_name(lambda_value)
    lambda_dir = output_dir / adapter_name
    if lambda_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{lambda_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(lambda_dir)
    lambda_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(existing_weights, framework="pt", device="cpu") as existing_file, safe_open(
        patch_weights, framework="pt", device="cpu"
    ) as patch_file:
        existing_keys = list(existing_file.keys())
        patch_keys = list(patch_file.keys())
        if set(existing_keys) != set(patch_keys):
            missing_from_patch = sorted(set(existing_keys) - set(patch_keys))[:10]
            missing_from_existing = sorted(set(patch_keys) - set(existing_keys))[:10]
            raise ValueError(
                "Existing and patch adapter keys differ: "
                f"missing_from_patch={missing_from_patch}, missing_from_existing={missing_from_existing}"
            )

        prefixes = lora_prefixes(existing_keys)
        tensors: dict[str, torch.Tensor] = {}
        for index, prefix in enumerate(prefixes, start=1):
            a_key = prefix + LORA_A_SUFFIX
            b_key = prefix + LORA_B_SUFFIX
            existing_a = existing_file.get_tensor(a_key)
            existing_b = existing_file.get_tensor(b_key)
            patch_a = patch_file.get_tensor(a_key)
            patch_b = patch_file.get_tensor(b_key)
            if existing_a.shape[1] != patch_a.shape[1] or existing_b.shape[0] != patch_b.shape[0]:
                raise ValueError(f"Shape mismatch for {prefix}")

            new_a, new_b = low_rank_svd_product(
                existing_a,
                existing_b,
                lora_scale(existing_config, prefix, existing_a.shape[0]),
                patch_a,
                patch_b,
                lora_scale(patch_config, prefix, patch_a.shape[0]),
                lambda_value,
                target_rank,
            )
            tensors[a_key] = new_a
            tensors[b_key] = new_b
            if index % 500 == 0 or index == len(prefixes):
                print(f"{adapter_name}: merged {index}/{len(prefixes)} LoRA modules", flush=True)

        for key in existing_keys:
            if key.endswith(LORA_A_SUFFIX) or key.endswith(LORA_B_SUFFIX):
                continue
            tensors[key] = existing_file.get_tensor(key)

    config = dict(existing_config)
    config["r"] = target_rank
    config["lora_alpha"] = target_rank
    config["rank_pattern"] = {key: target_rank for key in (config.get("rank_pattern") or {})}
    config["alpha_pattern"] = {key: target_rank for key in (config.get("alpha_pattern") or {})}
    config["base_model_name_or_path"] = str(model_path)
    config.pop("target_parameters", None)

    save_file(tensors, lambda_dir / "adapter_model.safetensors")
    (lambda_dir / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_sidecar_files([checkpoint_dir, existing_dir, output_dir / "patch_adapter_uncompressed", model_path], lambda_dir)

    return {
        "lambda": lambda_value,
        "adapter_dir": str(lambda_dir),
        "rank": target_rank,
        "num_tensors": len(tensors),
        "num_lora_modules": len(prefixes),
    }


def main() -> None:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    lambda_values = parse_lambda_values(args.lambda_values)

    results = []
    for value in lambda_values:
        result = export_one_lambda(value, checkpoint_dir, output_dir, model_path, args.svd_rank, args.overwrite)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        results.append(result)

    manifest = {
        "checkpoint_dir": str(checkpoint_dir),
        "merge_formula": "existing_adapter + lambda * patch_adapter, truncated with low-rank SVD",
        "svd_rank": args.svd_rank,
        "outputs": results,
    }
    (output_dir / "rank32_lambda_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
