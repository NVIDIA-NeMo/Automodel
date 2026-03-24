#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rename LLM finetune YAML configs and add benchmark sections.

Usage:
  python rename_and_benchmark.py           # dry-run (shows what would change)
  python rename_and_benchmark.py --execute # apply changes
"""

import os
import re
import sys
import subprocess
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).parent
SKIP_DIRS = {"llama3", "test_configs", "__pycache__"}

# Feature tokens to recognise from original filenames (order matters for output)
FEATURE_TOKENS = [
    "2nodes", "single_gpu", "custom", "columnmapped",
    "megatron_fsdp", "flashoptim", "deepep", "packed_sequence",
    "spark", "hsdp", "tp2", "fp8", "qat", "muon", "thd", "te", "pp",
]

# Model name -> size in billions for models without an obvious B-suffix
MODEL_SIZES_OVERRIDE = {
    "phi-2": 2.7,
    "phi-4": 14.0,
    "Phi-3-mini-4k-instruct": 3.8,
    "Baichuan2-7B-Chat": 7.0,
    "starcoder2-7b": 7.0,
    "glm-4-9b-chat-hf": 9.0,
    "Falcon3-7B-Instruct": 7.0,
    "c4ai-command-r7b-12-2024": 7.0,
    "gemma-7b": 7.0,
    "gemma-2-9b-it": 9.0,
    "gemma-3-270m": 0.27,
    "functiongemma-270m-it": 0.27,
    "OLMo-2-0425-1B-Instruct": 1.0,
    "Moonlight-16B-A3B": 16.0,
    "Qwen3-30B-A3B": 30.0,
    "Qwen3-30B-A3B-Thinking-2507": 30.0,
    "Qwen1.5-MoE-A2.7B": 14.3,   # ~14B total params
    "Qwen3-0.6B": 0.6,
    "Qwen3-Next-80B-A3B-Instruct": 80.0,
    "Mixtral-8x7B-v0.1": 46.0,
    "Mixtral-8x7B-Instruct-v0.1": 46.0,
    "Mistral-Nemo-Base-2407": 12.0,
    "Nemotron-Flash-1B": 1.0,
    "NVIDIA-Nemotron-Nano-9B-v2": 9.0,
    "GLM-4.5-Air": 30.0,    # Large MoE, conservative estimate
    "GLM-4.7-Flash": 30.0,
    "GLM-4.7": 30.0,
    "GLM-5": 30.0,
    "MiniMax-M2.1": 100.0,   # Large MoE
    "MiniMax-M2.5": 100.0,
    "Step-3.5-Flash": 30.0,
    "deepseek-v32": 671.0,   # DeepSeek-V3.2
    "granite-3.3-2b-instruct": 2.0,
    "gpt-oss-20b": 20.0,
    "gpt-oss-120b": 120.0,
    "Qwen3MoeConfig": 2.0,   # 2-layer proxy (tiny test model)
}

DATASET_PRETTY = {
    "squad": "SQuAD",
    "hellaswag": "HellaSwag",
    "mock": "mock dataset (benchmarking)",
    "natural-instructions": "Natural Instructions",
    "tulu-3-sft-mixture": "Tulu-3 SFT mixture",
    "xlam-function-calling-60k": "xLAM function calling",
    "chat": "chat (local JSONL)",
}

FEATURE_DESC = {
    "fp8": "FP8 quantization",
    "pp": "pipeline parallelism",
    "megatron_fsdp": "Megatron-FSDP",
    "hsdp": "HSDP",
    "tp2": "tensor parallelism (tp=2)",
    "packed_sequence": "packed sequences",
    "flashoptim": "FlashAdamW optimizer",
    "muon": "Muon optimizer",
    "qat": "quantization-aware training",
    "spark": "SPARK loss",
    "deepep": "DeepEP",
    "thd": "THD layout",
    "columnmapped": "column-mapped dataset",
    "2nodes": "2 nodes",
    "custom": "custom checkpoint",
    "single_gpu": "single GPU",
    "te": "TransformerEngine",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model_size_b(model_name: str) -> float:
    if model_name in MODEL_SIZES_OVERRIDE:
        return MODEL_SIZES_OVERRIDE[model_name]
    # Match NNB where NN is not preceded by a capital letter (to skip A3B etc.)
    matches = re.findall(r'(?<![A-Z])(\d+(?:\.\d+)?)B', model_name)
    if matches:
        return max(float(m) for m in matches)
    return 0.0


def get_cadence(model_name: str) -> int:
    size = get_model_size_b(model_name)
    if size == 0.0:
        print(f"    WARNING: unknown size for {model_name!r}, using cadence 7")
        return 7
    return 1 if size < 20.0 else 7


def extract_model_name(config: dict) -> str:
    model = config.get("model", {})
    if not isinstance(model, dict):
        return "unknown"
    # Direct path (most configs)
    path = model.get("pretrained_model_name_or_path", "")
    if path:
        return path.split("/")[-1]
    # Nested under model.config (e.g. deepseek from_config style)
    sub = model.get("config", {})
    if isinstance(sub, dict):
        path = sub.get("pretrained_model_name_or_path", "")
        if path:
            name = path.split("/")[-1]
            if name:
                return name
        # Derive from _target_ class name (e.g. Qwen3MoeConfig)
        target = sub.get("_target_", "")
        if target:
            cls = target.split(".")[-1]
            if cls:
                return cls
    return "unknown"


def get_task(config: dict) -> str:
    has_peft = "peft" in config
    has_quant = "quantization" in config
    if has_peft and has_quant:
        return "qlora"
    return "peft" if has_peft else "sft"


def extract_dataset(config: dict) -> str:
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict):
        return "unknown"
    target = dataset.get("_target_", "")
    if "Mock" in target or "mock" in target:
        return "mock"
    for key in ("dataset_name", "path_or_dataset", "path_or_dataset_id"):
        val = dataset.get(key)
        if val:
            val = str(val)
            if "your/path" in val or val.endswith(".jsonl") or val.endswith(".json"):
                return "chat" if "ChatDataset" in target else "custom"
            return val.split("/")[-1]
    return "unknown"


def has_moe(file_path: Path) -> bool:
    if "moe" in file_path.name.lower():
        return True
    try:
        return "moe" in file_path.read_text(errors="replace").lower()
    except Exception:
        return False


def extract_feature_tokens(stem: str) -> list:
    stem_l = stem.lower()
    found = []
    for token in FEATURE_TOKENS:
        if re.search(r"(?:^|_)" + re.escape(token) + r"(?:_|$)", stem_l):
            found.append(token)
    return found


def generate_description(model_name: str, task: str, dataset: str, features: list) -> str:
    task_str = {"sft": "SFT", "peft": "LoRA (PEFT)", "qlora": "QLoRA"}.get(task, task.upper())
    ds_str = DATASET_PRETTY.get(dataset, dataset)
    desc = f"{model_name} {task_str} fine-tuning on {ds_str}"
    feat_strs = [FEATURE_DESC[f] for f in features if f in FEATURE_DESC]
    if feat_strs:
        desc += ", with " + ", ".join(feat_strs)
    return desc


def format_benchmark_block(description: str, cadence: int, is_moe: bool,
                            existing: dict | None = None) -> str:
    ex = existing or {}
    lines = ["benchmark:"]
    lines.append(f"  description: {description}")
    lines.append(f"  warmup_steps: {ex.get('warmup_steps', 20)}")
    lines.append(f"  max_steps: {ex.get('max_steps', 200)}")
    lines.append(f"  num_nodes: {ex.get('num_nodes', 1)}")
    lines.append(f"  num_gpus: {ex.get('num_gpus', 8)}")
    lines.append(f"  cadence: {cadence}")
    if is_moe:
        lines.append("  fake_balanced_gate: True")
    # Preserve extra existing fields (e.g. peak_tflops, compare, nsys_*)
    skip = {"description", "warmup_steps", "max_steps", "num_nodes", "num_gpus",
            "cadence", "fake_balanced_gate"}
    for k in sorted(set(ex.keys()) - skip):
        v = ex[k]
        if isinstance(v, list):
            lines.append(f"  {k}: {v}")
        elif isinstance(v, bool):
            lines.append(f"  {k}: {v}")
        elif isinstance(v, str):
            lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def update_content(content: str, benchmark_block: str) -> str:
    """Remove any existing benchmark section and insert the new one at the top."""
    lines = content.split("\n")

    # Find and remove existing benchmark: block
    bm_start = bm_end = None
    for i, line in enumerate(lines):
        if re.match(r"^benchmark\s*:", line):
            bm_start = i
        elif bm_start is not None and bm_end is None:
            stripped = line.strip()
            if stripped and not line[0].isspace() and not line.startswith("#"):
                bm_end = i
                break
    if bm_start is not None:
        if bm_end is None:
            bm_end = len(lines)
        # Trim blank lines before/after the block
        while bm_start > 0 and lines[bm_start - 1].strip() == "":
            bm_start -= 1
        while bm_end < len(lines) and lines[bm_end].strip() == "":
            bm_end += 1
        lines = lines[:bm_start] + lines[bm_end:]

    # Find insert position: after all leading comment/blank lines
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith("#") or line.strip() == "":
            insert_pos = i + 1
        else:
            break

    bm_lines = benchmark_block.rstrip("\n").split("\n")
    new_lines = lines[:insert_pos] + [""] + bm_lines + [""] + lines[insert_pos:]
    # Collapse more than two consecutive blank lines
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(new_lines))
    return result


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_directory(parent_dir: Path, execute: bool):
    yaml_files = sorted(parent_dir.glob("*.yaml"))
    if not yaml_files:
        return

    print(f"\n{'='*60}")
    print(f"Directory: {parent_dir.relative_to(BASE_DIR)}/")
    print(f"{'='*60}")

    # Step 1: Parse all files
    file_info = []
    for yaml_path in yaml_files:
        try:
            content = yaml_path.read_text(errors="replace")
            config = yaml.safe_load(content) or {}
            if not isinstance(config, dict):
                print(f"  SKIP {yaml_path.name}: not a mapping")
                continue
        except Exception as e:
            print(f"  ERROR parsing {yaml_path.name}: {e}")
            continue

        model_name = extract_model_name(config)
        task = get_task(config)
        dataset = extract_dataset(config)
        moe = has_moe(yaml_path)
        cadence = get_cadence(model_name)
        existing_bm = config.get("benchmark") or {}

        features = extract_feature_tokens(yaml_path.stem)
        base_name = f"{model_name}_{task}_{dataset}"

        file_info.append({
            "path": yaml_path,
            "content": content,
            "model_name": model_name,
            "task": task,
            "dataset": dataset,
            "moe": moe,
            "cadence": cadence,
            "existing_bm": existing_bm,
            "features": features,
            "base_name": base_name,
        })

    # Step 2: Detect collisions → assign final names
    groups: dict[str, list] = {}
    for fi in file_info:
        groups.setdefault(fi["base_name"], []).append(fi)

    for fi in file_info:
        group = groups[fi["base_name"]]
        if len(group) == 1:
            fi["new_name"] = fi["base_name"] + ".yaml"
        else:
            tokens = fi["features"]
            others_have_tokens = any(g["features"] for g in group if g is not fi)
            if tokens:
                fi["new_name"] = fi["base_name"] + "_" + "_".join(tokens) + ".yaml"
            elif others_have_tokens:
                # This is the plain base version — keep base name
                fi["new_name"] = fi["base_name"] + ".yaml"
            else:
                # All files in group have no tokens; sort and number them
                group_sorted = sorted(group, key=lambda g: g["path"].stem)
                idx = group_sorted.index(fi)
                if idx == 0:
                    fi["new_name"] = fi["base_name"] + ".yaml"
                else:
                    fi["new_name"] = fi["base_name"] + f"_{idx + 1}.yaml"

    # Step 3: Report / apply
    for fi in file_info:
        model_subdir = parent_dir / fi["model_name"]
        new_path = model_subdir / fi["new_name"]
        rel_old = fi["path"].relative_to(BASE_DIR)
        rel_new = new_path.relative_to(BASE_DIR)

        description = generate_description(
            fi["model_name"], fi["task"], fi["dataset"], fi["features"]
        )
        bm_block = format_benchmark_block(
            description, fi["cadence"], fi["moe"], fi["existing_bm"]
        )
        new_content = update_content(fi["content"], bm_block)

        print(f"\n  {rel_old}")
        print(f"  → {rel_new}")
        print(f"     cadence={fi['cadence']} moe={fi['moe']} task={fi['task']} dataset={fi['dataset']}")

        if execute:
            model_subdir.mkdir(exist_ok=True)
            # git mv to new location (preserves git history)
            result = subprocess.run(
                ["git", "mv", str(fi["path"]), str(new_path)],
                capture_output=True, text=True,
                cwd=str(BASE_DIR.parent.parent),
            )
            if result.returncode != 0:
                print(f"     ERROR git mv: {result.stderr.strip()}")
                continue
            # Write updated content (adds benchmark section)
            new_path.write_text(new_content, encoding="utf-8")
            print(f"     ✓ done")


def main():
    execute = "--execute" in sys.argv
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"\n{'#'*60}")
    print(f"# LLM finetune config rename + benchmark  [{mode}]")
    print(f"{'#'*60}")

    if not execute:
        print("\nNo changes will be made. Pass --execute to apply.\n")

    subdirs = sorted(d for d in BASE_DIR.iterdir()
                     if d.is_dir() and d.name not in SKIP_DIRS)

    for subdir in subdirs:
        process_directory(subdir, execute=execute)

    print("\n\nDone.")
    if not execute:
        print("Re-run with --execute to apply changes.")


if __name__ == "__main__":
    main()
