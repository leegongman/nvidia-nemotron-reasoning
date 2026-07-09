#!/usr/bin/env python3
"""Merge an existing residual LoRA into submission_1 with module-wise scales.

This is the fastest/cheapest sweep path: no training, only recompress

    Delta_final ~= Delta_submission_1 + scale(module) * Delta_residual

into one rank-32 adapter.  It is useful when a global residual scale improves
some equation_numeric behavior but damages fragile bit_manipulation cases.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from additional_tuning_common import (
    COMPETITION_MAX_LORA_RANK,
    SUBMISSION_1_ROOT,
    discover_adapter,
    env_bool,
    env_float,
    env_int,
    env_str,
    log,
    reset_dir,
    str2bool,
    zip_adapter,
)
from sft_residual_lora_svd import (
    compressed_sum_pair,
    copy_adapter_sidecars,
    load_lora_scale,
    normalize_adapter_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module-wise residual LoRA scale merge.")
    parser.add_argument("--initial-adapter", default=env_str("INITIAL_ADAPTER", str(discover_adapter(SUBMISSION_1_ROOT))))
    parser.add_argument("--residual-adapter", default=env_str("RESIDUAL_ADAPTER", "/home/ubuntu/additonal_tuning/outputs/residual_lora_svd_v1/residual_adapter"))
    parser.add_argument("--output-dir", default=env_str("MODULEWISE_OUTPUT_DIR", "/home/ubuntu/additonal_tuning/outputs/residual_modulewise"))
    parser.add_argument("--target-rank", type=int, default=env_int("TARGET_RANK", COMPETITION_MAX_LORA_RANK))
    parser.add_argument("--default-scale", type=float, default=env_float("MODULEWISE_DEFAULT_SCALE", 0.0))
    parser.add_argument("--module-scales", default=env_str("MODULEWISE_SCALES", "in_proj=0.06,out_proj=0.06"))
    parser.add_argument("--zip-submission", type=str2bool, default=env_bool("ZIP_SUBMISSION", True))
    return parser.parse_args()


def parse_scale_map(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --module-scales item: {item!r}; expected name=value")
        name, raw = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Bad --module-scales item: {item!r}; empty name")
        out[name] = float(raw.strip())
    return out


def scale_for_key(key: str, scale_map: dict[str, float], default: float) -> tuple[float, str]:
    # Longest match wins, so users can override a broad module with a specific
    # layer substring if needed.
    best_name = ""
    best_scale = default
    for name, scale in scale_map.items():
        if f".{name}." in key or name in key:
            if len(name) > len(best_name):
                best_name = name
                best_scale = scale
    return best_scale, best_name or "__default__"


def main() -> None:
    args = parse_args()
    if args.target_rank != COMPETITION_MAX_LORA_RANK:
        raise ValueError("target_rank must remain 32 for competition compatibility.")
    if args.default_scale < 0:
        raise ValueError("--default-scale must be non-negative")
    scale_map = parse_scale_map(args.module_scales)
    for name, scale in scale_map.items():
        if scale < 0:
            raise ValueError(f"Scale for {name} must be non-negative")

    import torch
    from safetensors.torch import load_file, save_file

    initial_adapter = Path(args.initial_adapter)
    residual_adapter = Path(args.residual_adapter)
    output_dir = reset_dir(args.output_dir)
    copy_adapter_sidecars(initial_adapter, output_dir)

    old = load_file(initial_adapter / "adapter_model.safetensors", device="cpu")
    res_raw = load_file(residual_adapter / "adapter_model.safetensors", device="cpu")
    res = {normalize_adapter_key(k): v for k, v in res_raw.items()}
    old_scale = load_lora_scale(initial_adapter / "adapter_config.json", args.target_rank)
    res_base_scale = load_lora_scale(residual_adapter / "adapter_config.json", 1)

    merged = {k: v.detach().cpu().float() for k, v in old.items()}
    merged_pairs = 0
    skipped_pairs = 0
    scale_counts: dict[str, int] = {}
    effective_scales: dict[str, float] = {}

    for a_key, res_a in sorted(res.items()):
        if not a_key.endswith(".lora_A.weight"):
            continue
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if a_key not in old or b_key not in old or b_key not in res:
            log(f"WARNING: skipping unmatched residual tensor pair: {a_key}")
            skipped_pairs += 1
            continue
        user_scale, matched_name = scale_for_key(a_key, scale_map, args.default_scale)
        scale_counts[matched_name] = scale_counts.get(matched_name, 0) + 1
        effective_scales[matched_name] = user_scale * res_base_scale
        if user_scale == 0:
            skipped_pairs += 1
            continue
        new_a, new_b = compressed_sum_pair(
            torch,
            old[a_key],
            old[b_key],
            res_a,
            res[b_key],
            old_scale=old_scale,
            res_scale=user_scale * res_base_scale,
            target_rank=args.target_rank,
        )
        merged[a_key] = new_a
        merged[b_key] = new_b
        merged_pairs += 1

    cfg_path = output_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["r"] = args.target_rank
    cfg["lora_alpha"] = args.target_rank
    cfg["inference_mode"] = True
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    save_file(merged, output_dir / "adapter_model.safetensors")
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "modulewise_residual_lora_svd_recompress",
        "initial_adapter": str(initial_adapter),
        "residual_adapter": str(residual_adapter),
        "target_rank": args.target_rank,
        "old_scale": old_scale,
        "residual_base_scale": res_base_scale,
        "default_user_scale": args.default_scale,
        "module_user_scales": scale_map,
        "effective_scales": effective_scales,
        "scale_counts": scale_counts,
        "merged_pairs": merged_pairs,
        "skipped_pairs": skipped_pairs,
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Merged module-wise adapter: {output_dir} (merged_pairs={merged_pairs}, skipped_pairs={skipped_pairs})")
    if args.zip_submission:
        log(f"Submission zip: {zip_adapter(output_dir)}")


if __name__ == "__main__":
    main()
