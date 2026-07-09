#!/usr/bin/env python3
"""Build RSP Vast.ai PRO 6000 train payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


ROOT_FILES = [
    "rsp_train_huikang_compatible.py",
    "verify_rsp_train_shell.py",
    "verify_rsp_dataset.py",
    "rsp_schema.json",
    "rsp_design.md",
    "rsp_run_train_pro6000.sh",
]
DATASET_FILES = [
    "rsp_anchor_sft.jsonl",
    "rsp_decision_sft.jsonl",
    "rsp_decision_preferences.jsonl",
    "rsp_manifest.json",
    "rsp_verification.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": dst.as_posix(),
        "source": src.as_posix(),
        "bytes": dst.stat().st_size,
        "sha256": sha256(dst),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/rsp_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rsp_vast_payload/payload"))
    parser.add_argument("--archive", type=Path, default=Path("outputs/rsp_vast_payload/rsp_pro6000_payload.tar.gz"))
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}
    for name in ROOT_FILES:
        files[name] = copy_file(Path(name), output / name)
    for name in DATASET_FILES:
        files[f"rsp_dataset/{name}"] = copy_file(args.dataset_dir / name, output / "rsp_dataset" / name)

    manifest = {
        "schema": "rsp-vast-pro6000-payload-v1",
        "candidate_id": "rsp-rule-selection-post-training",
        "current_minimum_goal_achieved": "no",
        "submission_allowed": False,
        "eval_allowed": False,
        "target_runtime_hours": 3,
        "default_train_settings": {
            "enable_4bit": True,
            "sft_epochs": 1.0,
            "preference_epochs": 0.35,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "preference_batch_size": 2,
            "preference_gradient_accumulation_steps": 8,
            "effective_batch_size": 16,
            "max_seq_length": 8192,
        },
        "expected_outputs": [
            "/workspace/rsp_vast/output/submission.zip",
            "/workspace/rsp_vast/output/rsp_training_audit.json",
            "/workspace/rsp_vast/output/rsp_post_training_adapter_gate.json",
        ],
        "run_command": "bash /workspace/rsp_vast/payload/rsp_run_train_pro6000.sh",
        "files": files,
    }
    (output / "payload_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(output, arcname="payload")
    print(
        json.dumps(
            {
                "rsp_vast_payload_built": True,
                "archive": str(args.archive),
                "archive_sha256": sha256(args.archive),
                "target_runtime_hours": 3,
                "files": len(files) + 1,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
