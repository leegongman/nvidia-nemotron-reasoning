#!/usr/bin/env python3
"""Build a Kaggle train-only notebook for RSP.

The generated notebook assumes the RSP runtime bundle is attached as a Kaggle
input dataset.  It verifies data and training shell before loading the 30B
model, then runs train-only adapter generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATE = "rsp-rule-selection-post-training"
RUNTIME_FILES = [
    "rsp_train_huikang_compatible.py",
    "verify_rsp_train_shell.py",
    "verify_rsp_dataset.py",
    "rsp_schema.json",
]


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def embedded_files_cell() -> str:
    files = {name: Path(name).read_text(encoding="utf-8") for name in RUNTIME_FILES}
    return (
        "from pathlib import Path\nimport json\n"
        "RUNTIME_DIR = Path('/kaggle/working/rsp_runtime')\n"
        "RUNTIME_DIR.mkdir(parents=True, exist_ok=True)\n"
        f"EMBEDDED_FILES = {json.dumps(files, ensure_ascii=False)!r}\n"
        "EMBEDDED_FILES = json.loads(EMBEDDED_FILES)\n"
        "for name, content in EMBEDDED_FILES.items():\n"
        "    (RUNTIME_DIR / name).write_text(content, encoding='utf-8')\n"
        "print('embedded runtime files:', sorted(EMBEDDED_FILES))\n"
    )


def resolver_cell() -> str:
    return (
        "from pathlib import Path\nimport zipfile\n"
        "def resolve_one(label, candidates):\n"
        "    found = {path.resolve() for path in candidates if path.exists()}\n"
        "    if len(found) != 1:\n"
        "        raise RuntimeError(f'{label}: expected exactly one path, found {sorted(map(str, found))}')\n"
        "    print(label, '=>', next(iter(found)))\n"
        "    return next(iter(found))\n"
        "dataset_dirs = [\n"
        "    Path('/kaggle/input/rsp-runtime-inputs/rsp_dataset'),\n"
        "    Path('/kaggle/input/rsp-runtime-inputs/rsp_dataset'),\n"
        "    *Path('/kaggle/input').glob('**/rsp_dataset'),\n"
        "]\n"
        "dataset_dirs = [path for path in dataset_dirs if path.exists()]\n"
        "if len(set(path.resolve() for path in dataset_dirs)) == 1:\n"
        "    DATASET_DIR = dataset_dirs[0].resolve()\n"
        "else:\n"
        "    dataset_zip = resolve_one('RSP dataset zip', [\n"
        "        Path('/kaggle/input/rsp-runtime-inputs/rsp_dataset.zip'),\n"
        "        *Path('/kaggle/input').glob('**/rsp_dataset.zip'),\n"
        "    ])\n"
        "    extract_root = Path('/kaggle/working/rsp_runtime_input_unzipped')\n"
        "    extract_root.mkdir(parents=True, exist_ok=True)\n"
        "    with zipfile.ZipFile(dataset_zip) as archive:\n"
        "        archive.extractall(extract_root)\n"
        "    candidates = [extract_root / 'rsp_dataset', extract_root]\n"
        "    DATASET_DIR = resolve_one('unzipped RSP dataset dir', [path for path in candidates if (path / 'rsp_anchor_sft.jsonl').is_file()])\n"
        "def resolve_model_path():\n"
        "    roots = []\n"
        "    for candidate in [\n"
        "        Path('/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1'),\n"
        "        Path('/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1'),\n"
        "        Path('/kaggle/input/nemotron-3-nano-30b-a3b-bf16'),\n"
        "        *Path('/kaggle/input').glob('**/config.json'),\n"
        "    ]:\n"
        "        root = candidate.parent if candidate.name == 'config.json' else candidate\n"
        "        if (root / 'config.json').is_file() and (root / 'tokenizer_config.json').is_file():\n"
        "            roots.append(root.resolve())\n"
        "    roots = sorted(set(roots), key=lambda path: len(str(path)))\n"
        "    if len(roots) == 1:\n"
        "        return str(roots[0])\n"
        "    if len(roots) > 1:\n"
        "        raise RuntimeError(f'multiple model roots found: {roots}')\n"
        "    import kagglehub\n"
        "    return kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')\n"
    )


def dependency_cell() -> str:
    return (
        "import subprocess, sys\n"
        "PACKAGE_DIR = resolve_one('Nemotron package dir', [\n"
        "    Path('/kaggle/input/datasets/mayukh18/nemotron-packages/packages'),\n"
        "    Path('/kaggle/input/nemotron-packages/packages'),\n"
        "    *Path('/kaggle/input').glob('**/nemotron-packages/packages'),\n"
        "])\n"
        "CAUSAL_WHEEL = resolve_one('causal-conv wheel', [\n"
        "    Path('/kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'),\n"
        "    Path('/kaggle/input/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'),\n"
        "    *Path('/kaggle/input').glob('**/causal_conv1d-*.whl'),\n"
        "])\n"
        "MAMBA_WHEEL = resolve_one('mamba-ssm wheel', [\n"
        "    Path('/kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'),\n"
        "    Path('/kaggle/input/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'),\n"
        "    *Path('/kaggle/input').glob('**/mamba_ssm-*.whl'),\n"
        "])\n"
        "install = subprocess.run([\n"
        "    sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(PACKAGE_DIR),\n"
        "    str(CAUSAL_WHEEL), str(MAMBA_WHEEL)\n"
        "], text=True, capture_output=True)\n"
        "print(install.stdout)\n"
        "assert install.returncode == 0, install.stderr\n"
        "import mamba_ssm\n"
        "gpu_info = subprocess.run(['nvidia-smi'], text=True, capture_output=True)\n"
        "print(gpu_info.stdout)\n"
        "assert gpu_info.returncode == 0, gpu_info.stderr\n"
        "print('mamba_ssm import ok')\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/rsp_train_kernel/rsp_train.ipynb"))
    parser.add_argument("--enable-4bit", action="store_true")
    args = parser.parse_args()

    train_flags = " --enable-4bit" if args.enable_4bit else ""
    notebook = {
        "cells": [
            markdown(
                f"# {CANDIDATE} train-only\n\n"
                "현재 최소 목표 달성: 아니요\n\n"
                "This notebook trains only. It does not run evaluation or external submission."
            ),
            code(
                "CANDIDATE = 'rsp-rule-selection-post-training'\n"
                "CURRENT_MINIMUM_GOAL_ACHIEVED = 'no'\n"
                "SUBMISSION_ALLOWED = False\n"
                "EVALUATION_ALLOWED = False\n"
                "assert SUBMISSION_ALLOWED is False and EVALUATION_ALLOWED is False\n"
            ),
            code(embedded_files_cell()),
            code(resolver_cell()),
            code(dependency_cell()),
            code(
                "import subprocess, sys\n"
                "verify_data = subprocess.run([\n"
                "    sys.executable, str(RUNTIME_DIR / 'verify_rsp_dataset.py'),\n"
                "    '--dataset-dir', str(DATASET_DIR),\n"
                "    '--json-output', '/kaggle/working/rsp_dataset_verification.json',\n"
                "], cwd=RUNTIME_DIR, text=True, capture_output=True)\n"
                "print(verify_data.stdout)\n"
                "assert verify_data.returncode == 0, verify_data.stderr\n"
                "verify_shell = subprocess.run([\n"
                "    sys.executable, str(RUNTIME_DIR / 'verify_rsp_train_shell.py'),\n"
                "    '--train-script', str(RUNTIME_DIR / 'rsp_train_huikang_compatible.py'),\n"
                "    '--dataset-verification', '/kaggle/working/rsp_dataset_verification.json',\n"
                "    '--json-output', '/kaggle/working/rsp_train_shell_verification.json',\n"
                "], cwd=RUNTIME_DIR, text=True, capture_output=True)\n"
                "print(verify_shell.stdout)\n"
                "assert verify_shell.returncode == 0, verify_shell.stderr\n"
            ),
            code(
                "MODEL_PATH = resolve_model_path()\n"
                "print('model path:', MODEL_PATH)\n"
                "import subprocess, sys\n"
                "cmd = [\n"
                "    sys.executable, str(RUNTIME_DIR / 'rsp_train_huikang_compatible.py'),\n"
                "    '--dataset-dir', str(DATASET_DIR),\n"
                "    '--model', MODEL_PATH,\n"
                "    '--output-dir', '/kaggle/working/rsp_adapter',\n"
                "    '--submission-zip', '/kaggle/working/submission.zip',\n"
                "    '--audit-json', '/kaggle/working/rsp_training_audit.json',\n"
                "    '--max-seq-length', '8192',\n"
                "    '--lora-rank', '32',\n"
                "    '--lora-alpha', '32',\n"
                "    '--lora-dropout', '0.0',\n"
                "    '--sft-learning-rate', '1.6e-4',\n"
                "    '--preference-learning-rate', '3.5e-5',\n"
                "    '--sft-epochs', '1.0',\n"
                "    '--preference-epochs', '0.35',\n"
                "    '--per-device-train-batch-size', '1',\n"
                "    '--gradient-accumulation-steps', '16',\n"
                "    '--preference-batch-size', '1',\n"
                "    '--preference-gradient-accumulation-steps', '16',\n"
                "    '--warmup-steps', '20',\n"
                "]\n"
                f"cmd += {train_flags.split()!r}\n"
                "train = subprocess.run(cmd, cwd=RUNTIME_DIR, text=True)\n"
                "assert train.returncode == 0\n"
            ),
            code(
                "from pathlib import Path\nimport subprocess, sys\n"
                "assert Path('/kaggle/working/submission.zip').is_file(), 'missing train output submission.zip'\n"
                "post = subprocess.run([\n"
                "    sys.executable, str(RUNTIME_DIR / 'verify_rsp_train_shell.py'),\n"
                "    '--train-script', str(RUNTIME_DIR / 'rsp_train_huikang_compatible.py'),\n"
                "    '--dataset-verification', '/kaggle/working/rsp_dataset_verification.json',\n"
                "    '--adapter-zip', '/kaggle/working/submission.zip',\n"
                "    '--json-output', '/kaggle/working/rsp_post_training_adapter_gate.json',\n"
                "], cwd=RUNTIME_DIR, text=True, capture_output=True)\n"
                "print(post.stdout)\n"
                "assert post.returncode == 0, post.stderr\n"
                "assert not any(Path('/kaggle/working').rglob('eval_*')), 'train-only notebook must not produce eval outputs'\n"
                "print('RSP train-only artifact ready: /kaggle/working/submission.zip')\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rsp_train_kernel_built": True, "output": str(args.output), "enable_4bit": args.enable_4bit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
