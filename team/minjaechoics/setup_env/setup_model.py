#!/usr/bin/env python3
"""Download model artifacts needed by train_sft.py.

This script is intentionally separate from training so a fresh server can warm
the Hugging Face cache before running train_sft.py. It downloads repository
snapshots only; it does not load the 30B model into GPU/CPU memory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download Hugging Face models used by train_sft.py."
    )
    parser.add_argument(
        "--base-model-id",
        default=os.environ.get("BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID),
        help=f"Base policy model to download. Default: {DEFAULT_BASE_MODEL_ID}",
    )
    parser.add_argument(
        "--judge-model-id",
        default=os.environ.get("JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL_ID),
        help=f"Optional judge model to download with --include-judge. Default: {DEFAULT_JUDGE_MODEL_ID}",
    )
    parser.add_argument(
        "--include-judge",
        action="store_true",
        default=os.environ.get("DOWNLOAD_JUDGE", "").lower() in {"1", "true", "yes", "y"},
        help="Also download the judge model repo.",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("MODEL_REVISION"),
        help="Model revision/branch/tag/commit to download. Default: Hugging Face default branch.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Explicit Hugging Face hub cache directory. Default: library default, "
            "which honors HF_HOME/HF_HUB_CACHE."
        ),
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Optional local directory to materialize the base model snapshot.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token for gated/private models. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--verify-config",
        action="store_true",
        help="After download, verify AutoConfig/AutoTokenizer can read the cached base model.",
    )
    return parser.parse_args()


def download_snapshot(
    model_id: str,
    *,
    revision: str | None,
    cache_dir: Path | None,
    token: str | None,
    local_dir: Path | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install your training dependencies first, "
            "for example: pip install huggingface_hub transformers"
        ) from exc

    print(f"\nDownloading {model_id}")
    if revision:
        print(f"revision = {revision}")
    if cache_dir:
        print(f"cache_dir = {cache_dir}")
    if local_dir:
        print(f"local_dir = {local_dir}")

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir) if local_dir else None,
        token=token,
    )
    resolved = Path(path).expanduser().resolve()
    print(f"ready: {resolved}")
    return resolved


def verify_config(model_path: Path) -> None:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed, so --verify-config cannot run."
        ) from exc

    print(f"\nVerifying Transformers config/tokenizer from {model_path}")
    AutoConfig.from_pretrained(model_path)
    AutoTokenizer.from_pretrained(model_path)
    print("verification ok")


def main() -> None:
    args = parse_args()

    base_path = download_snapshot(
        args.base_model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
        local_dir=args.local_dir,
    )

    if args.verify_config:
        verify_config(base_path)

    if args.include_judge:
        download_snapshot(
            args.judge_model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            token=args.token,
        )

    print("\nModel setup complete.")


if __name__ == "__main__":
    main()
