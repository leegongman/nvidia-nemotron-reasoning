# ============================================================
# Nemotron LoRA Target Module Inspector
# ------------------------------------------------------------
# 목적:
#   - Nemotron에서 LoRA를 걸 수 있는 module 이름 전체 확인
#   - PEFT/Unsloth target_modules에 넣을 leaf name 정리
#   - attention / mamba / moe / lm_head 등으로 자동 분류
#   - 추천 target_modules 조합 출력
#   - /home/ubuntu/chatgptresult_outputs/nemotron_analysis/nemotron_lora_module_report.txt 저장
#   - /home/ubuntu/chatgptresult_outputs/nemotron_analysis/nemotron_lora_modules.csv 저장
#
# 사용:
#   1. 이미 model이 로드되어 있으면 그대로 검사
#   2. model이 없으면 아래 MODEL_PATH로 직접 로드
# ============================================================

import os
import re
import csv
import json
import torch
from collections import Counter, defaultdict
from pathlib import Path

# ------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/workspace/.hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848",
)
MAX_SEQ_LENGTH = 8192

LOAD_MODEL_IF_NEEDED = True
USE_EMPTY_MODEL_FOR_ANALYSIS = os.environ.get(
    "USE_EMPTY_MODEL_FOR_ANALYSIS", "true"
).lower() in {"1", "true", "yes", "y", "on"}

REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/home/ubuntu/chatgptresult_outputs/nemotron_analysis"))
REPORT_TXT = str(REPORT_DIR / "nemotron_lora_module_report.txt")
REPORT_CSV = str(REPORT_DIR / "nemotron_lora_modules.csv")

# 너무 긴 full path 출력 제한용
MAX_FULL_PATH_PRINT = 300


# ------------------------------------------------------------
# 1. 모델 로드 또는 기존 model 재사용
# ------------------------------------------------------------

def get_or_load_model():
    global model

    if "model" in globals() and model is not None:
        print("[INFO] Existing `model` found. Using already-loaded model.")
        return model

    if not LOAD_MODEL_IF_NEEDED:
        raise RuntimeError("No existing `model` found and LOAD_MODEL_IF_NEEDED=False.")

    if USE_EMPTY_MODEL_FOR_ANALYSIS:
        print("[INFO] No existing `model` found. Building empty native Transformers model for analysis...")
        print("[INFO] This avoids loading 30B weights and does not require mamba-ssm.")
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        print("[INFO] Empty model structure built.")
        return model

    print("[INFO] No existing `model` found. Loading full model with Unsloth...")
    print("[INFO] Full Unsloth loading requires mamba-ssm and causal-conv1d for this Nemotron remote code.")
    from unsloth import FastLanguageModel

    loaded = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    model = loaded[0]

    print("[INFO] Full model loaded.")
    return model


model = get_or_load_model()


# ------------------------------------------------------------
# 2. 유틸 함수
# ------------------------------------------------------------

def has_2d_weight(module):
    if not hasattr(module, "weight"):
        return False

    weight = getattr(module, "weight")

    if isinstance(weight, torch.nn.Parameter):
        return weight.ndim == 2

    if torch.is_tensor(weight):
        return weight.ndim == 2

    return False


def get_weight_shape(module):
    weight = getattr(module, "weight")
    return tuple(weight.shape)


def get_leaf_name(full_name):
    if full_name == "":
        return ""
    return full_name.split(".")[-1]


def classify_module_name(name, leaf, cls_name):
    lower = name.lower()
    leaf_lower = leaf.lower()

    # lm_head
    if leaf_lower == "lm_head" or lower.endswith(".lm_head"):
        return "lm_head"

    # attention-like
    attention_leafs = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query", "key", "value",
        "query_key_value", "dense",
    }
    if leaf_lower in attention_leafs:
        return "attention"

    if any(x in lower for x in [
        "self_attn", "attention", "attn",
    ]) and any(x in leaf_lower for x in [
        "q_proj", "k_proj", "v_proj", "o_proj", "proj"
    ]):
        return "attention"

    # mamba / ssm / sequence mixer
    if any(x in lower for x in [
        "mamba", "ssm", "mixer", "conv1d", "selective",
    ]):
        return "mamba_or_ssm"

    if leaf_lower in {
        "in_proj", "out_proj", "x_proj", "dt_proj",
        "conv1d", "conv_proj",
    }:
        return "mamba_or_ssm"

    # MoE / expert
    if any(x in lower for x in [
        "expert", "experts", "moe", "router",
    ]):
        return "moe_or_expert"

    # FFN / MLP
    if leaf_lower in {
        "up_proj", "down_proj", "gate_proj", "gate_up_proj",
        "w1", "w2", "w3", "fc1", "fc2",
    }:
        return "ffn_or_mlp"

    if any(x in lower for x in [
        "mlp", "feed_forward", "ffn",
    ]):
        return "ffn_or_mlp"

    # embedding은 보통 LoRA target으로 추천하지 않음
    if any(x in lower for x in [
        "embed", "embedding", "word_embeddings",
    ]):
        return "embedding"

    return "other_linear"


def is_likely_lora_target(category, leaf):
    """
    target_modules 후보로 추천할지 여부.
    embedding류는 제외.
    """
    if category == "embedding":
        return False
    if leaf in {"", "weight"}:
        return False
    return True


def short_shape(shape):
    return "x".join(str(x) for x in shape)


# ------------------------------------------------------------
# 3. 모든 2D weight module 수집
# ------------------------------------------------------------

records = []

for full_name, module in model.named_modules():
    if not has_2d_weight(module):
        continue

    leaf = get_leaf_name(full_name)
    cls_name = module.__class__.__name__
    shape = get_weight_shape(module)
    category = classify_module_name(full_name, leaf, cls_name)

    records.append({
        "full_name": full_name,
        "leaf": leaf,
        "class": cls_name,
        "shape": shape,
        "shape_str": short_shape(shape),
        "category": category,
        "is_likely_lora_target": is_likely_lora_target(category, leaf),
    })


# ------------------------------------------------------------
# 4. 통계 정리
# ------------------------------------------------------------

leaf_counter = Counter(r["leaf"] for r in records)
category_counter = Counter(r["category"] for r in records)

leaf_by_category = defaultdict(Counter)
shape_by_leaf = defaultdict(Counter)
examples_by_leaf = defaultdict(list)
examples_by_category = defaultdict(list)

for r in records:
    leaf_by_category[r["category"]][r["leaf"]] += 1
    shape_by_leaf[r["leaf"]][r["shape_str"]] += 1

    if len(examples_by_leaf[r["leaf"]]) < 5:
        examples_by_leaf[r["leaf"]].append(r["full_name"])

    if len(examples_by_category[r["category"]]) < 10:
        examples_by_category[r["category"]].append(r["full_name"])


candidate_leafs = sorted({
    r["leaf"]
    for r in records
    if r["is_likely_lora_target"]
})

attention_leafs = sorted({
    r["leaf"]
    for r in records
    if r["category"] == "attention" and r["is_likely_lora_target"]
})

mamba_leafs = sorted({
    r["leaf"]
    for r in records
    if r["category"] == "mamba_or_ssm" and r["is_likely_lora_target"]
})

moe_leafs = sorted({
    r["leaf"]
    for r in records
    if r["category"] in {"moe_or_expert", "ffn_or_mlp"} and r["is_likely_lora_target"]
})

lm_head_leafs = sorted({
    r["leaf"]
    for r in records
    if r["category"] == "lm_head" and r["is_likely_lora_target"]
})

broad_leafs = sorted(set(attention_leafs + mamba_leafs + moe_leafs + lm_head_leafs))


# ------------------------------------------------------------
# 5. 출력 문자열 구성
# ------------------------------------------------------------

lines = []

def add(line=""):
    lines.append(str(line))


add("=" * 100)
add("NEMOTRON LoRA TARGET MODULE INSPECTION REPORT")
add("=" * 100)
add(f"MODEL_PATH: {MODEL_PATH}")
add(f"Total 2D-weight modules found: {len(records)}")
add()

add("=" * 100)
add("1. CATEGORY COUNTS")
add("=" * 100)
for cat, cnt in category_counter.most_common():
    add(f"{cat:25s} {cnt:6d}")
add()

add("=" * 100)
add("2. LEAF NAME COUNTS")
add("   These are the names usually used in --target_modules.")
add("=" * 100)
for leaf, cnt in leaf_counter.most_common():
    shapes = ", ".join(f"{s}:{c}" for s, c in shape_by_leaf[leaf].most_common())
    add(f"{leaf:35s} count={cnt:5d} | shapes={shapes}")
add()

add("=" * 100)
add("3. LEAF NAMES BY CATEGORY")
add("=" * 100)
for cat in sorted(leaf_by_category.keys()):
    add(f"\n[{cat}]")
    for leaf, cnt in leaf_by_category[cat].most_common():
        shapes = ", ".join(f"{s}:{c}" for s, c in shape_by_leaf[leaf].most_common())
        add(f"  {leaf:35s} count={cnt:5d} | shapes={shapes}")
add()

add("=" * 100)
add("4. EXAMPLE FULL PATHS BY CATEGORY")
add("=" * 100)
for cat in sorted(examples_by_category.keys()):
    add(f"\n[{cat}]")
    for name in examples_by_category[cat]:
        shown = name if len(name) <= MAX_FULL_PATH_PRINT else name[:MAX_FULL_PATH_PRINT] + " ..."
        add(f"  {shown}")
add()

add("=" * 100)
add("5. FULL MODULE LIST")
add("=" * 100)
for r in records:
    add(
        f"{r['category']:18s} | "
        f"{r['leaf']:30s} | "
        f"{r['class']:30s} | "
        f"{r['shape_str']:18s} | "
        f"{r['full_name']}"
    )
add()

add("=" * 100)
add("6. RECOMMENDED target_modules")
add("=" * 100)

def fmt_modules(xs):
    return ",".join(xs) if xs else "(none found)"

add()
add("[A] Attention only")
add(fmt_modules(attention_leafs))
add()
add("Bash:")
add(f'--target_modules {fmt_modules(attention_leafs)} \\')
add("--add_lm_head_lora false \\")

add()
add("[B] Mamba / SSM / sequence mixing only")
add(fmt_modules(mamba_leafs))
add()
add("Bash:")
add(f'--target_modules {fmt_modules(mamba_leafs)} \\')
add("--add_lm_head_lora false \\")

add()
add("[C] FFN / MoE / expert only")
add(fmt_modules(moe_leafs))
add()
add("Bash:")
add(f'--target_modules {fmt_modules(moe_leafs)} \\')
add("--add_lm_head_lora false \\")

add()
add("[D] Broad LoRA without lm_head")
broad_without_lm = sorted(set(attention_leafs + mamba_leafs + moe_leafs))
add(fmt_modules(broad_without_lm))
add()
add("Bash:")
add(f'--target_modules {fmt_modules(broad_without_lm)} \\')
add("--add_lm_head_lora false \\")

add()
add("[E] Broad LoRA with lm_head")
add(fmt_modules(broad_leafs))
add()
add("Bash:")
add(f'--target_modules {fmt_modules(broad_without_lm)} \\')
add("--add_lm_head_lora true \\")

add()
add("=" * 100)
add("7. PRACTICAL NOTES")
add("=" * 100)
add("- PEFT/Unsloth target_modules usually uses leaf names, not full paths.")
add("- Example: use q_proj, not backbone.layers.0.self_attn.q_proj.")
add("- If --add_lm_head_lora true, lm_head may be added manually even if not in --target_modules.")
add("- If you want to completely exclude lm_head, set --add_lm_head_lora false.")
add("- For fast experiments, try attention-only, mamba-only, moe-only, then broad.")
add("- For weak-vs-strong Ortho-LoRA, changing target_modules changes where conflict projection is applied.")
add()


report = "\n".join(lines)


# ------------------------------------------------------------
# 6. 파일 저장
# ------------------------------------------------------------

Path(REPORT_TXT).parent.mkdir(parents=True, exist_ok=True)
Path(REPORT_CSV).parent.mkdir(parents=True, exist_ok=True)
Path(REPORT_TXT).write_text(report, encoding="utf-8")

with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "leaf",
            "class",
            "shape_str",
            "is_likely_lora_target",
            "full_name",
        ],
    )
    writer.writeheader()
    for r in records:
        writer.writerow({
            "category": r["category"],
            "leaf": r["leaf"],
            "class": r["class"],
            "shape_str": r["shape_str"],
            "is_likely_lora_target": r["is_likely_lora_target"],
            "full_name": r["full_name"],
        })


# ------------------------------------------------------------
# 7. 화면 출력
# ------------------------------------------------------------

print(report)

print("\n" + "=" * 100)
print("SAVED FILES")
print("=" * 100)
print(f"TXT report: {REPORT_TXT}")
print(f"CSV table : {REPORT_CSV}")

print("\n" + "=" * 100)
print("COPY-PASTE CANDIDATES")
print("=" * 100)

print("\n# Attention only")
print(f"--target_modules {fmt_modules(attention_leafs)} \\")
print("--add_lm_head_lora false \\")

print("\n# Mamba / SSM only")
print(f"--target_modules {fmt_modules(mamba_leafs)} \\")
print("--add_lm_head_lora false \\")

print("\n# FFN / MoE only")
print(f"--target_modules {fmt_modules(moe_leafs)} \\")
print("--add_lm_head_lora false \\")

print("\n# Broad without lm_head")
print(f"--target_modules {fmt_modules(broad_without_lm)} \\")
print("--add_lm_head_lora false \\")

print("\n# Broad with lm_head")
print(f"--target_modules {fmt_modules(broad_without_lm)} \\")
print("--add_lm_head_lora true \\")
