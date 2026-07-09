#!/usr/bin/env python3
"""Build the RSP rule-selection dataset.

The builder produces three files:

- rsp_anchor_sft.jsonl
- rsp_decision_sft.jsonl
- rsp_decision_preferences.jsonl

It does not train or evaluate.  It only converts verified solver-trace rows into
RSP input artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ANCHOR = Path("data/source/anchor_sft.jsonl")
DEFAULT_EQUATION = Path("data/source/equation_numeric.jsonl")
DEFAULT_TARGET_REPAIR = Path("data/source/target_repair_rows.jsonl")
DEFAULT_OUTPUT = Path("data/rsp_dataset")


BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def terminal_answer(completion: str) -> str:
    matches = BOX_RE.findall(completion)
    return matches[-1].strip() if matches else ""


def clean_domain(domain: str) -> str:
    if domain == "numeral_system":
        return "numeral"
    if domain == "equation_numeric_symbolic":
        return "equation_numeric"
    return domain


def anchor_row(row: dict[str, Any]) -> dict[str, Any]:
    domain = str(row.get("domain") or row.get("category"))
    return {
        "id": f"rsp_anchor::{row['id']}",
        "row_type": "anchor_sft",
        "domain": clean_domain(domain),
        "source_domain": domain,
        "prompt": row["prompt"],
        "completion": row["completion"],
        "final_answer": str(row["final_answer"]),
        "source": "huikang_style_v2_clean_anchor",
        "source_id": str(row["id"]),
        "sample_weight": 1.0,
    }


def safe_final_answer(answer: Any) -> bool:
    text = str(answer)
    if not text:
        return False
    # Curly braces and backslashes are valid task symbols in some synthetic
    # equation rows, but they are unsafe for the competition's boxed-answer
    # extraction and for simple adapter-training gates.
    return not any(ch in text for ch in "{}\\")


def parse_bit_examples(prompt: str) -> tuple[list[tuple[str, str]], str]:
    examples = re.findall(r"(?m)^([01]{8})\s*->\s*([01]{8})\s*$", prompt)
    targets = re.findall(r"Now, determine the output for:\s*([01]{8})", prompt)
    if not examples or not targets:
        raise ValueError("bit examples or target missing")
    return examples, targets[-1]


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
    expr = expr.strip().replace(" ", "")
    if expr == "C0":
        return 0
    if expr == "C1":
        return 1
    if re.fullmatch(r"i[0-7]", expr):
        return int(bits[int(expr[1])])
    if re.fullmatch(r"t[0-7]", expr):
        return int(bits[int(expr[1])])
    if expr.startswith("NOT(") and expr.endswith(")"):
        return 1 - eval_bit_expr(expr[4:-1], bits)
    match = re.fullmatch(r"([A-Z0-9_]+)\((.*)\)", expr)
    if not match:
        raise ValueError(f"bad bit expr {expr!r}")
    name, arg_text = match.groups()
    args = split_args(arg_text)
    return bit_call(name, [eval_bit_expr(arg, bits) for arg in args])


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


def bit_expr_args(expr: str) -> list[str]:
    return re.findall(r"[it][0-7]", expr)


def bit_wrong_exprs(expr: str) -> list[str]:
    expr = expr.strip().replace(" ", "")
    args = bit_expr_args(expr)
    wrong: list[str] = []
    if expr in {"C0", "C1"}:
        wrong.append("C1" if expr == "C0" else "C0")
    elif expr.startswith("NOT(") and len(args) == 1:
        wrong.append(args[0].replace("t", "i"))
    elif re.fullmatch(r"i[0-7]|t[0-7]", expr):
        wrong.append(f"NOT({expr.replace('t', 'i')})")
        wrong.extend(["C0", "C1"])
    elif len(args) >= 2:
        base_args = [arg.replace("t", "i") for arg in args[:2]]
        for gate in ["XOR", "OR", "AND", "XNOR", "NAND", "NOR"]:
            candidate = f"{gate}({base_args[0]},{base_args[1]})"
            if candidate != expr:
                wrong.append(candidate)
    return list(dict.fromkeys(wrong))


def bit_expressions(completion: str) -> list[str]:
    expressions = re.findall(
        r"(?m)(?:^\s*SELECT:\s*|^B\d+:\s*SELECT\s+)(.+?)\s*$",
        completion,
    )
    return [expr.strip() for expr in expressions]


def apply_bit(expressions: list[str], bits: str) -> str:
    return "".join(str(eval_bit_expr(expr, bits)) for expr in expressions)


def build_bit_decisions(row: dict[str, Any], source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples, target = parse_bit_examples(str(row["prompt"]))
    exprs = bit_expressions(str(row["completion"]))
    if len(exprs) != 8:
        return [], []
    if any(apply_bit(exprs, src) != expected for src, expected in examples):
        return [], []
    answer = apply_bit(exprs, target)
    if answer != str(row["final_answer"]):
        return [], []

    decision_points = [{"bit": idx, "selected": expr} for idx, expr in enumerate(exprs)]
    sft = [{
        "id": f"rsp_decision_sft::{source}::{row['id']}",
        "row_type": "decision_sft",
        "domain": "bit_manipulation",
        "prompt": row["prompt"],
        "completion": compact_bit_completion(exprs, target, answer),
        "final_answer": answer,
        "decision_points": decision_points,
        "source": source,
        "source_id": str(row["id"]),
        "sample_weight": 0.35,
    }]

    prefs: list[dict[str, Any]] = []
    for bit_idx, expr in enumerate(exprs):
        for wrong in bit_wrong_exprs(expr):
            wrong_exprs = list(exprs)
            wrong_exprs[bit_idx] = wrong
            rejected = apply_bit(wrong_exprs, target)
            if rejected == answer:
                continue
            prefs.append({
                "id": f"rsp_pref::{source}::{row['id']}::B{bit_idx}::{len(prefs)}",
                "row_type": "decision_preference",
                "domain": "bit_manipulation",
                "prompt": row["prompt"],
                "chosen": compact_bit_completion(exprs, target, answer),
                "rejected": compact_bit_completion(wrong_exprs, target, rejected),
                "chosen_answer": answer,
                "rejected_answer": rejected,
                "decision_points": [{"bit": bit_idx, "selected": expr, "rejected": wrong}],
                "negative_type": "bit_gate_counterfactual",
                "source": source,
                "source_id": str(row["id"]),
                "sample_weight": 0.20,
            })
            break
    return sft, prefs


def compact_bit_completion(exprs: list[str], target: str, answer: str) -> str:
    lines = ["<think>", "PARSE: 8-bit rule induction.", "SELECT:"]
    for idx, expr in enumerate(exprs):
        lines.append(f"B{idx}: SELECT {expr}")
    lines.append("VERIFY: selected rules reproduce the demonstrations.")
    lines.append(f"ANSWER: applying selected rules to {target} gives {answer}.")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def parse_equation_examples(prompt: str) -> tuple[list[tuple[str, str]], str]:
    examples: list[tuple[str, str]] = []
    target = ""
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Now, determine the result for:"):
            target = line.split(":", 1)[1].strip()
        elif " = " in line and not line.startswith(("In Alice", "Below")):
            left, right = line.split(" = ", 1)
            examples.append((left.strip(), right.strip()))
    if not examples or not target:
        raise ValueError("equation examples or target missing")
    return examples, target


def equation_dsl(completion: str) -> dict[str, Any] | None:
    match = re.search(r"(?m)^(?:program|dsl): (\{.*\})$", completion)
    if not match:
        return None
    return json.loads(match.group(1))


def apply_program(program: dict[str, Any], src: str) -> str | None:
    kind = str(program.get("kind"))
    params = dict(program.get("params", {}))
    try:
        if kind == "take":
            return "".join(src[int(i)] for i in params["indices"])
        if kind == "drop_pos":
            drop = {int(i) for i in params["indices"]}
            return "".join(ch for idx, ch in enumerate(src) if idx not in drop)
        if kind == "take_replace":
            out = []
            for j, idx in enumerate(params["indices"]):
                value = src[int(idx)]
                out.append(dict(params["tables"][j]).get(value, value))
            return "".join(out)
        if kind in {"op_if", "branch"}:
            branch = params["branches"].get(src[int(params["pos"])])
            return None if branch is None else apply_program(branch, src)
        if kind == "take_const":
            out = []
            for atom in params["atoms"]:
                if atom[0] == "src":
                    out.append(src[int(atom[1])])
                elif atom[0] == "const":
                    out.append(str(atom[1]))
                elif atom[0] == "map":
                    value = src[int(atom[1])]
                    out.append(dict(atom[2]).get(value, value))
            return "".join(out)
    except Exception:
        return None
    return None


def equation_wrong_outputs(program: dict[str, Any], target: str) -> list[tuple[str, str]]:
    outputs: list[tuple[str, str]] = []
    kind = str(program.get("kind"))
    params = dict(program.get("params", {}))
    if kind in {"op_if", "branch"}:
        branches = params.get("branches", {})
        selected_key = target[int(params["pos"])]
        for key, branch in branches.items():
            if key == selected_key:
                continue
            out = apply_program(branch, target)
            if out:
                outputs.append((f"sibling_branch_{key}", out))
    for indices in ([0], [1], [2], [3], [4], [0, 1], [1, 2], [2, 3], [3, 4]):
        if max(indices) < len(target):
            outputs.append((f"wrong_take_{'_'.join(map(str, indices))}", "".join(target[i] for i in indices)))
    return list(dict(outputs).items())


def build_equation_decisions(row: dict[str, Any], source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        examples, target = parse_equation_examples(str(row["prompt"]))
        dsl = equation_dsl(str(row["completion"]))
        if dsl is None:
            return [], []
        for src, expected in examples:
            if apply_program(dsl, src) != expected:
                return [], []
        answer = apply_program(dsl, target)
        if not answer or answer != str(row["final_answer"]):
            return [], []
        decision_points = [{
            "program_label": dsl.get("label"),
            "program_kind": dsl.get("kind"),
            "target": target,
        }]
        sft = [{
            "id": f"rsp_decision_sft::{source}::{row['id']}",
            "row_type": "decision_sft",
            "domain": "equation_numeric",
            "prompt": row["prompt"],
            "completion": compact_equation_completion(dsl, target, answer),
            "final_answer": answer,
            "decision_points": decision_points,
            "source": source,
            "source_id": str(row["id"]),
            "sample_weight": 0.30,
        }]
        prefs: list[dict[str, Any]] = []
        for neg_name, wrong_answer in equation_wrong_outputs(dsl, target):
            if wrong_answer == answer:
                continue
            if not safe_final_answer(wrong_answer):
                continue
            prefs.append({
                "id": f"rsp_pref::{source}::{row['id']}::{neg_name}",
                "row_type": "decision_preference",
                "domain": "equation_numeric",
                "prompt": row["prompt"],
                "chosen": compact_equation_completion(dsl, target, answer),
                "rejected": compact_equation_rejected(dsl, target, wrong_answer, neg_name),
                "chosen_answer": answer,
                "rejected_answer": wrong_answer,
                "decision_points": decision_points,
                "negative_type": neg_name,
                "source": source,
                "source_id": str(row["id"]),
                "sample_weight": 0.18,
            })
            break
        return sft, prefs
    except Exception:
        return [], []


def compact_equation_completion(dsl: dict[str, Any], target: str, answer: str) -> str:
    label = dsl.get("label", dsl.get("kind", "program"))
    return "\n".join([
        "<think>",
        "PARSE: symbolic equation transformation.",
        f"SELECT: {label}",
        "VERIFY: selected DSL reproduces the demonstrations.",
        f"ANSWER: applying the selected DSL to {target} gives {answer}.",
        "</think>",
        f"\\boxed{{{answer}}}",
    ])


def compact_equation_rejected(dsl: dict[str, Any], target: str, wrong: str, negative_type: str) -> str:
    return "\n".join([
        "<think>",
        "PARSE: symbolic equation transformation.",
        f"SELECT: {negative_type}",
        f"ANSWER: applying this wrong branch to {target} gives {wrong}.",
        "</think>",
        f"\\boxed{{{wrong}}}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--equation", type=Path, default=DEFAULT_EQUATION)
    parser.add_argument("--target-repair", type=Path, default=DEFAULT_TARGET_REPAIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    anchor_rows = read_jsonl(args.anchor)
    equation_rows = read_jsonl(args.equation)
    target_repair_rows = read_jsonl(args.target_repair) if args.target_repair.exists() else []

    anchors = [anchor_row(row) for row in anchor_rows if safe_final_answer(row.get("final_answer", ""))]
    decisions: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []

    for row in anchor_rows:
        domain = clean_domain(str(row.get("domain", "")))
        if not safe_final_answer(row.get("final_answer", "")):
            continue
        if domain == "bit_manipulation":
            sft, prefs = build_bit_decisions(row, "v2_clean_bit")
            decisions.extend(sft)
            preferences.extend(prefs)
    for row in equation_rows:
        if not safe_final_answer(row.get("final_answer", "")):
            continue
        sft, prefs = build_equation_decisions(row, "v2_coverage_equation")
        decisions.extend(sft)
        preferences.extend(prefs)
    for row in target_repair_rows:
        domain = clean_domain(str(row.get("domain", "")))
        if not safe_final_answer(row.get("final_answer", "")):
            continue
        if domain == "bit_manipulation":
            sft, prefs = build_bit_decisions(row, "target_repair_bit")
            decisions.extend(sft)
            preferences.extend(prefs)
        elif domain == "equation_numeric":
            sft, prefs = build_equation_decisions(row, "target_repair_equation")
            decisions.extend(sft)
            preferences.extend(prefs)

    # Keep the decision-preference objective bounded.  RSP is intended to
    # repair rule-selection boundaries without overwhelming the anchor traces.
    pref_caps = {"bit_manipulation": 1600, "equation_numeric": 900}
    capped_preferences: list[dict[str, Any]] = []
    seen_by_domain = Counter()
    for row in preferences:
        domain = str(row["domain"])
        if seen_by_domain[domain] >= pref_caps.get(domain, 0):
            continue
        capped_preferences.append(row)
        seen_by_domain[domain] += 1
    preferences = capped_preferences

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "rsp_anchor_sft.jsonl", anchors)
    write_jsonl(args.output_dir / "rsp_decision_sft.jsonl", decisions)
    write_jsonl(args.output_dir / "rsp_decision_preferences.jsonl", preferences)

    summary = {
        "schema": "rsp-dataset-manifest-v1",
        "candidate_id": "rsp-rule-selection-post-training",
        "current_minimum_goal_achieved": "no",
        "submission_allowed": False,
        "gpu_execution_allowed": False,
        "files": {
            "anchor_sft": "rsp_anchor_sft.jsonl",
            "decision_sft": "rsp_decision_sft.jsonl",
            "decision_preferences": "rsp_decision_preferences.jsonl",
        },
        "counts": {
            "anchor_sft": len(anchors),
            "decision_sft": len(decisions),
            "decision_preferences": len(preferences),
        },
        "domain_counts": {
            "anchor_sft": dict(Counter(row["domain"] for row in anchors)),
            "decision_sft": dict(Counter(row["domain"] for row in decisions)),
            "decision_preferences": dict(Counter(row["domain"] for row in preferences)),
        },
    }
    (args.output_dir / "rsp_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
