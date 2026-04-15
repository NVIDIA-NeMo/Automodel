#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Collect benchmark results from SLURM logs and recipe configs into a JSON artifact.

Usage:
    python3 collect_benchmark_artifact.py \
        --config examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml \
        --log /path/to/slurm_output.out \
        --output benchmark_results.json
"""

import argparse
import json
import re
import sys

import yaml


def parse_log(log_path: str) -> dict:
    """Parse benchmark summary and metadata from SLURM log."""
    with open(log_path, "r", errors="replace") as f:
        content = f.read()

    results = {}
    metadata = {}

    summary_patterns = {
        "setup_time_seconds": r"Total setup time:\s+([\d.]+)\s+seconds",
        "warmup_time_seconds": r"Total warmup time.*?:\s+([\d.]+)\s+seconds",
        "training_time_seconds": r"Total iteration time.*?:\s+([\d.]+)\s+seconds",
        "avg_iter_time_seconds": r"Average iteration time:\s+([\d.]+)\s+seconds",
        "avg_mfu_percent": r"Average MFU:\s+([\d.]+)%",
    }
    for key, pattern in summary_patterns.items():
        m = re.search(pattern, content)
        if m:
            results[key] = float(m.group(1))

    m = re.search(r"TFLOPs/GPU:\s+([\d.]+)", content)
    if m:
        results["tflops_per_gpu"] = float(m.group(1))

    m = re.search(r"TFLOPS multiplier for PEFT:.*?=\s+([\d.]+)", content)
    if m:
        results["peft_tflops_multiplier"] = float(m.group(1))

    m = re.search(r"Trainable parameters:\s+([\d,]+)", content)
    if m:
        results["trainable_params"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Total parameters:\s+([\d,]+)", content)
    if m:
        results["total_params"] = int(m.group(1).replace(",", ""))

    m = re.search(r"World size:\s+(\d+)", content)
    if m:
        metadata["world_size"] = int(m.group(1))
    m = re.search(r"Model name:\s+(.+)", content)
    if m:
        metadata["model_name"] = m.group(1).strip()
    m = re.search(r"Recipe:\s+(.+)", content)
    if m:
        metadata["recipe"] = m.group(1).strip()
    m = re.search(r"Timestamp:\s+'([^']+)'", content)
    if m:
        metadata["timestamp"] = m.group(1)

    return {"results": results, "metadata": metadata}


def parse_config(config_path: str) -> dict:
    """Extract relevant config fields from recipe YAML."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    config = {}

    benchmark = cfg.get("benchmark", {})
    config["num_nodes"] = benchmark.get("num_nodes")
    config["warmup_steps"] = benchmark.get("warmup_steps")
    config["peak_tflops"] = benchmark.get("peak_tflops")

    step = cfg.get("step_scheduler", {})
    config["global_batch_size"] = step.get("global_batch_size")
    config["local_batch_size"] = step.get("local_batch_size")
    config["max_steps"] = step.get("max_steps")

    dataset = cfg.get("dataset", {})
    config["seq_len"] = dataset.get("seq_len")

    dist = cfg.get("distributed", {})
    config["strategy"] = dist.get("strategy")
    config["tp_size"] = dist.get("tp_size")
    config["cp_size"] = dist.get("cp_size")
    config["pp_size"] = dist.get("pp_size")
    config["ep_size"] = dist.get("ep_size")

    backend = cfg.get("model", {}).get("backend", {})
    config["backend_attn"] = backend.get("attn")
    config["backend_linear"] = backend.get("linear")
    config["backend_experts"] = backend.get("experts")
    config["backend_dispatcher"] = backend.get("dispatcher")
    config["fp8"] = backend.get("te_fp8") is not None

    peft_cfg = cfg.get("peft")
    if peft_cfg:
        config["peft"] = {
            "type": "lora",
            "dim": peft_cfg.get("dim"),
            "alpha": peft_cfg.get("alpha"),
        }
    else:
        config["peft"] = None

    return config


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark artifact from SLURM log and recipe config")
    parser.add_argument("--config", required=True, help="Path to recipe YAML config")
    parser.add_argument("--log", required=True, help="Path to SLURM output log")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    log_data = parse_log(args.log)
    config_data = parse_config(args.config)

    # Merge trainable/total params into peft section if present
    if config_data.get("peft") and "trainable_params" in log_data["results"]:
        config_data["peft"]["trainable_params"] = log_data["results"].pop("trainable_params")
        config_data["peft"]["total_params"] = log_data["results"].pop("total_params", None)
        if "peft_tflops_multiplier" in log_data["results"]:
            config_data["peft"]["tflops_multiplier"] = log_data["results"].pop("peft_tflops_multiplier")

    artifact = {
        "metadata": {
            "config_path": args.config,
            **log_data["metadata"],
        },
        "config": config_data,
        "results": log_data["results"],
    }

    with open(args.output, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Benchmark artifact written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
