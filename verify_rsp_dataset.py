#!/usr/bin/env python3
"""Verify the RSP dataset artifacts.

This is a static/data verifier only. It does not prove model performance.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DIR = Path("data/rsp_dataset")
BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def boxed_answer(text: str) -> str:
    matches = BOX_RE.findall(str(text))
    return matches[-1].strip() if matches else ""


def split_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:idx])
            start = idx + 1
    args.append(text[start:])
    return [arg.strip() for arg in args if arg.strip()]


def bit_call(name: str, args: list[int]) -> int:
    if name in {"ID", ""} and len(args) == 1:
        return args[0]
    if name == "NOT" and len(args) == 1:
        return 1 - args[0]
    if name == "AND" and len(args) >= 2:
        return int(all(args))
    if name == "OR" and len(args) >= 2:
        return int(any(args))
    if name in {"XOR", "PARITY3"} and len(args) >= 2:
        return sum(args) % 2
    if name == "XNOR" and len(args) >= 2:
        return 1 - (sum(args) % 2)
    if name == "NAND" and len(args) >= 2:
        return 1 - int(all(args))
    if name == "NOR" and len(args) >= 2:
        return 1 - int(any(args))
    if name == "MAJORITY3" and len(args) == 3:
        return int(sum(args) >= 2)
    if name == "NOT_MAJORITY3" and len(args) == 3:
        return 1 - int(sum(args) >= 2)
    if name == "CHOICE" and len(args) == 3:
        return args[1] if args[0] else args[2]
    if name == "NOT_CHOICE" and len(args) == 3:
        return 1 - (args[1] if args[0] else args[2])
    if name == "MAJORITY4" and len(args) == 4:
        return int(sum(args) >= 3)
    if name == "NOT_MAJORITY4" and len(args) == 4:
        return 1 - int(sum(args) >= 3)
    if name == "C0" and not args:
        return 0
    if name == "C1" and not args:
        return 1
    raise ValueError(f"unsupported bit expression {name}/{len(args)}")


def eval_bit_expr(expr: str, bits: str) -> int:
    expr = str(expr).strip().replace(" ", "")
    if expr == "C0":
        return 0
    if expr == "C1":
        return 1
    if re.fullmatch(r"[it][0-7]", expr):
        return int(bits[int(expr[1])])
    if expr.startswith("NOT(") and expr.endswith(")"):
        return 1 - eval_bit_expr(expr[4:-1], bits)
    match = re.fullmatch(r"([A-Z0-9_]+)\((.*)\)", expr)
    if not match:
        raise ValueError(f"bad bit expr {expr!r}")
    name, arg_text = match.groups()
    return bit_call(name, [eval_bit_expr(arg, bits) for arg in split_args(arg_text)])


def bit_expressions(completion: str) -> list[str]:
    return [
        expr.strip()
        for expr in re.findall(r"(?m)^B\d+:\s*SELECT\s+(.+?)\s*$", str(completion))
    ]


def parse_bit_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    examples = re.findall(r"(?m)^([01]{8})\s*->\s*([01]{8})\s*$", str(prompt))
    targets = re.findall(r"Now, determine the output for:\s*([01]{8})", str(prompt))
    if not examples or not targets:
        raise ValueError("bit examples or target missing")
    return examples, targets[-1]


def apply_bit(exprs: list[str], bits: str) -> str:
    return "".join(str(eval_bit_expr(expr, bits)) for expr in exprs)


def verify_bit_completion(row: dict[str, Any], completion_key: str, answer_key: str) -> None:
    examples, target = parse_bit_prompt(str(row["prompt"]))
    exprs = bit_expressions(str(row[completion_key]))
    if len(exprs) != 8:
        raise ValueError("bit completion must contain 8 SELECT decisions")
    for src, expected in examples:
        got = apply_bit(exprs, src)
        if got != expected and completion_key == "completion":
            raise ValueError(f"bit selected gates do not reproduce example {src}: {got}!={expected}")
    got_target = apply_bit(exprs, target)
    if got_target != str(row[answer_key]):
        raise ValueError(f"bit target answer mismatch {got_target}!={row[answer_key]}")
    if boxed_answer(str(row[completion_key])) != str(row[answer_key]):
        raise ValueError("bit boxed answer mismatch")


def terminal_box_ok(row: dict[str, Any], completion_key: str, answer_key: str) -> bool:
    return boxed_answer(str(row.get(completion_key, ""))) == str(row.get(answer_key, ""))


def check_required(row: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if field not in row]


ANCHOR_FIELDS = ["id", "row_type", "domain", "prompt", "completion", "final_answer", "source", "sample_weight"]
DECISION_FIELDS = ANCHOR_FIELDS + ["decision_points"]
PREF_FIELDS = [
    "id",
    "row_type",
    "domain",
    "prompt",
    "chosen",
    "rejected",
    "chosen_answer",
    "rejected_answer",
    "decision_points",
    "negative_type",
    "source",
    "sample_weight",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    anchor_path = args.dataset_dir / "rsp_anchor_sft.jsonl"
    decision_path = args.dataset_dir / "rsp_decision_sft.jsonl"
    pref_path = args.dataset_dir / "rsp_decision_preferences.jsonl"
    manifest_path = args.dataset_dir / "rsp_manifest.json"
    errors: list[str] = []

    for path in [anchor_path, decision_path, pref_path, manifest_path]:
        if not path.exists():
            errors.append(f"missing file: {path}")
    if errors:
        result = {"rsp_dataset_valid": False, "errors": len(errors), "first_errors": errors[:50]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    anchors = read_jsonl(anchor_path)
    decisions = read_jsonl(decision_path)
    prefs = read_jsonl(pref_path)
    ids: set[str] = set()

    for collection_name, rows, fields in [
        ("anchor", anchors, ANCHOR_FIELDS),
        ("decision", decisions, DECISION_FIELDS),
    ]:
        for row in rows:
            missing = check_required(row, fields)
            if missing:
                errors.append(f"{collection_name}:{row.get('id')}: missing {missing}")
                continue
            if row["id"] in ids:
                errors.append(f"duplicate id {row['id']}")
            ids.add(str(row["id"]))
            if not terminal_box_ok(row, "completion", "final_answer"):
                errors.append(f"{collection_name}:{row['id']}: boxed/final mismatch")
            if row["row_type"] not in {"anchor_sft", "decision_sft"}:
                errors.append(f"{collection_name}:{row['id']}: wrong row_type {row['row_type']}")
            if row["domain"] == "bit_manipulation" and row["row_type"] == "decision_sft":
                try:
                    verify_bit_completion(row, "completion", "final_answer")
                except Exception as exc:
                    errors.append(f"decision:{row['id']}: {exc}")

    for row in prefs:
        missing = check_required(row, PREF_FIELDS)
        if missing:
            errors.append(f"preference:{row.get('id')}: missing {missing}")
            continue
        if row["id"] in ids:
            errors.append(f"duplicate id {row['id']}")
        ids.add(str(row["id"]))
        if row.get("row_type") != "decision_preference":
            errors.append(f"preference:{row['id']}: wrong row_type {row.get('row_type')}")
        if str(row["chosen_answer"]) == str(row["rejected_answer"]):
            errors.append(f"preference:{row['id']}: rejected answer equals chosen answer")
        if not terminal_box_ok(row, "chosen", "chosen_answer"):
            errors.append(f"preference:{row['id']}: chosen boxed mismatch")
        if not terminal_box_ok(row, "rejected", "rejected_answer"):
            errors.append(f"preference:{row['id']}: rejected boxed mismatch")
        if row["domain"] == "bit_manipulation":
            try:
                verify_bit_completion(row, "chosen", "chosen_answer")
                verify_bit_completion(row, "rejected", "rejected_answer")
            except Exception as exc:
                errors.append(f"preference:{row['id']}: {exc}")

    anchor_count = len(anchors)
    decision_count = len(decisions)
    preference_count = len(prefs)
    if anchor_count < 7000:
        errors.append(f"anchor rows too small: {anchor_count}")
    if decision_count < 1000:
        errors.append(f"decision rows too small: {decision_count}")
    if preference_count < 1000:
        errors.append(f"preference rows too small: {preference_count}")
    if anchor_count and preference_count / anchor_count > 0.35:
        errors.append(f"preference/anchor ratio too high: {preference_count / anchor_count:.3f}")

    result = {
        "rsp_dataset_valid": not errors,
        "current_minimum_goal_achieved": "no",
        "submission_allowed": False,
        "gpu_execution_allowed": False,
        "counts": {
            "anchor_sft": anchor_count,
            "decision_sft": decision_count,
            "decision_preferences": preference_count,
        },
        "domain_counts": {
            "anchor_sft": dict(Counter(row.get("domain") for row in anchors)),
            "decision_sft": dict(Counter(row.get("domain") for row in decisions)),
            "decision_preferences": dict(Counter(row.get("domain") for row in prefs)),
        },
        "errors": len(errors),
        "first_errors": errors[:50],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["rsp_dataset_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
