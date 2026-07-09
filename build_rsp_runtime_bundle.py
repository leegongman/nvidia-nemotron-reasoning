#!/usr/bin/env python3
"""Build the RSP Kaggle runtime input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


CANDIDATE = "rsp-rule-selection-post-training"
ROOT_FILES = [
    "rsp_design.md",
    "rsp_schema.json",
    "build_rsp_dataset.py",
    "verify_rsp_dataset.py",
    "rsp_train_huikang_compatible.py",
    "verify_rsp_train_shell.py",
    "build_rsp_train_kernel.py",
]
DATASET_FILES = [
    "rsp_anchor_sft.jsonl",
    "rsp_decision_sft.jsonl",
    "rsp_decision_preferences.jsonl",
    "rsp_manifest.json",
    "rsp_verification.json",
    "rsp_train_shell_verification.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(dst), "source": str(src), "sha256": sha256(dst), "bytes": dst.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/rsp_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rsp_runtime_bundle"))
    args = parser.parse_args()

    output = args.output_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for name in ROOT_FILES:
        files[name] = copy_file(Path(name), output / name)
    for name in DATASET_FILES:
        files[f"rsp_dataset/{name}"] = copy_file(args.dataset_dir / name, output / "rsp_dataset" / name)

    manifest = {
        "schema": "rsp-runtime-bundle-v1",
        "candidate_id": CANDIDATE,
        "current_minimum_goal_achieved": "no",
        "submission_allowed": False,
        "gpu_execution_allowed": True,
        "purpose": "Kaggle train-only input bundle for RSP adapter training",
        "required_train_output": [
            "/kaggle/working/submission.zip",
            "/kaggle/working/rsp_training_audit.json",
            "/kaggle/working/rsp_post_training_adapter_gate.json",
        ],
        "files": files,
        "train_notebook": "outputs/rsp_train_kernel/rsp_train.ipynb",
    }
    (output / "runtime_bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "rsp-runtime-inputs",
                "id": "leegongman/rsp-runtime-inputs",
                "licenses": [{"name": "CC0-1.0"}],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rsp_runtime_bundle_built": True, "output_dir": str(output), "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
