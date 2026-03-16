#!/usr/bin/env python3
"""Compare layer-by-layer activations between NeMo AutoModel and HF Transformers.

For each prompt, compares:
  - Per-layer hidden states (cosine similarity and max absolute difference)
  - Final logits (cosine similarity, max abs diff, and top-1 agreement)

HF model is loaded with device_map="auto" in the main process.
NeMo model is loaded via torchrun with the training code path (EP, FSDP, etc.).

Usage:
    python compare_activations.py \
        --config examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml \
        --threshold 0.99 \
        --num-prompts 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Solve for x: 3x + 7 = 22. Show your work step by step.",
    "What is the derivative of f(x) = x^3 * ln(x)?",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain the difference between a stack and a queue with code examples.",
    "Tell me about the history of the Silk Road in three paragraphs.",
    "What are the main differences between classical and operant conditioning?",
    "Summarize the following in exactly two bullet points: Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with many layers.",
    "Translate the following English sentence to French: 'The weather is beautiful today and I would like to go for a walk.'",
    "Describe how the transformer attention mechanism works, including the role of queries, keys, and values.",
    "Write a haiku about a neural network learning to see.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare layer-by-layer activations: NeMo AutoModel vs HF Transformers.",
    )
    parser.add_argument("--config", required=True, help="Training YAML config for NeMo.")
    parser.add_argument("--model-path", default=None, help="Override model path (default: read from config).")
    parser.add_argument("--nproc", type=int, default=8, help="GPUs for NeMo torchrun.")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument(
        "--threshold", type=float, default=0.99, help="Minimum cosine similarity for PASS (default: 0.99)."
    )
    parser.add_argument("--gate-precision", type=str, default=None)
    parser.add_argument("--lm-head-precision", type=str, default=None)
    return parser.parse_known_args()


def _read_model_name_from_config(config_path: str) -> str:
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("model", {}).get("pretrained_model_name_or_path", "")


def run_hf_extraction(model_path: str, prompts: list[str], prompts_file: str | None) -> list[dict]:
    """Run HF activation extraction in a subprocess (to isolate GPU memory)."""
    script = Path(__file__).parent / "extract_hf_activations.py"

    with tempfile.TemporaryDirectory(prefix="hf_act_") as tmpdir:
        output_file = os.path.join(tmpdir, "hf_activations.pt")

        pf = prompts_file
        if pf is None:
            pf = os.path.join(tmpdir, "prompts.json")
            with open(pf, "w") as f:
                json.dump(prompts, f)

        python = os.environ.get("VIRTUAL_ENV", "/opt/venv") + "/bin/python"
        cmd = [
            python,
            str(script),
            "--model",
            model_path,
            "--output-file",
            output_file,
            "--num-prompts",
            str(len(prompts)),
            "--prompts-file",
            pf,
        ]

        logger.info("Step 1: HF Transformers activation extraction ...")
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            logger.error("HF extraction exited with code %d", result.returncode)
            sys.exit(1)

        return torch.load(output_file, map_location="cpu", weights_only=False)


def run_nemo_extraction(
    config: str,
    prompts: list[str],
    nproc: int,
    gate_precision: str | None,
    lm_head_precision: str | None,
    prompts_file: str | None,
    extra_args: list[str],
) -> list[dict]:
    """Run NeMo activation extraction via torchrun."""
    script = Path(__file__).parent / "extract_nemo_activations.py"

    with tempfile.TemporaryDirectory(prefix="nemo_act_") as tmpdir:
        output_file = os.path.join(tmpdir, "nemo_activations.pt")

        pf = prompts_file
        if pf is None:
            pf = os.path.join(tmpdir, "prompts.json")
            with open(pf, "w") as f:
                json.dump(prompts, f)

        venv = os.environ.get("VIRTUAL_ENV", "/opt/venv")
        torchrun_bin = os.path.join(venv, "bin", "torchrun")
        if not os.path.isfile(torchrun_bin):
            torchrun_bin = "torchrun"  # fallback to PATH
        cmd = [
            torchrun_bin,
            "--nproc-per-node",
            str(nproc),
            "--tee",
            "3",
            str(script),
            "--config",
            config,
            "--output-file",
            output_file,
            "--num-prompts",
            str(len(prompts)),
            "--prompts-file",
            pf,
        ]
        if gate_precision:
            cmd += ["--gate-precision", gate_precision]
        if lm_head_precision:
            cmd += ["--lm-head-precision", lm_head_precision]
        if extra_args:
            cmd += extra_args

        logger.info("Step 2: NeMo AutoModel activation extraction (torchrun) ...")
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            logger.error("NeMo torchrun exited with code %d", result.returncode)
            sys.exit(1)

        return torch.load(output_file, map_location="cpu", weights_only=False)


def compare_and_report(
    hf_results: list[dict],
    nemo_results: list[dict],
    threshold: float,
) -> bool:
    """Compare per-layer activations and final logits. Return True if all pass."""
    print()
    print("=" * 90)
    print("  Layer-by-Layer Activation Comparison: NeMo AutoModel vs HF Transformers")
    print("=" * 90)
    print(f"  Threshold (cosine sim): {threshold}")
    print(f"  Prompts: {len(hf_results)}")
    print()

    all_pass = True

    for prompt_idx, (hf, nemo) in enumerate(zip(hf_results, nemo_results)):
        prompt_display = hf["prompt"][:60] + "..." if len(hf["prompt"]) > 60 else hf["prompt"]
        print(f"  Prompt {prompt_idx}: {hf['num_tokens']} tokens | {prompt_display}")

        hf_layers = hf["layer_outputs"]
        nemo_layers = nemo["layer_outputs"]

        # Compare per-layer hidden states
        common_layers = sorted(set(hf_layers.keys()) & set(nemo_layers.keys()))
        layer_sims = []
        layer_diffs = []
        worst_layer = None
        worst_sim = 1.0

        for layer_name in common_layers:
            hf_h = hf_layers[layer_name].float()
            nemo_h = nemo_layers[layer_name].float()
            cos_sim = torch.nn.functional.cosine_similarity(hf_h.unsqueeze(0), nemo_h.unsqueeze(0)).item()
            max_diff = (hf_h - nemo_h).abs().max().item()
            layer_sims.append(cos_sim)
            layer_diffs.append(max_diff)

            if cos_sim < worst_sim:
                worst_sim = cos_sim
                worst_layer = layer_name

        mean_sim = sum(layer_sims) / len(layer_sims) if layer_sims else 0
        mean_diff = sum(layer_diffs) / len(layer_diffs) if layer_diffs else float("inf")
        max_max_diff = max(layer_diffs) if layer_diffs else float("inf")

        # Compare final logits
        hf_logits = hf["logits"].float()
        nemo_logits = nemo["logits"].float()
        logit_cos = torch.nn.functional.cosine_similarity(hf_logits.unsqueeze(0), nemo_logits.unsqueeze(0)).item()
        logit_max_diff = (hf_logits - nemo_logits).abs().max().item()
        top1_agree = hf_logits.argmax().item() == nemo_logits.argmax().item()

        prompt_pass = worst_sim >= threshold and logit_cos >= threshold
        if not prompt_pass:
            all_pass = False

        status = "PASS" if prompt_pass else "FAIL"
        print(f"    [{status}] Layers: mean_cos={mean_sim:.6f}  worst_cos={worst_sim:.6f} ({worst_layer})")
        print(f"           mean_max_diff={mean_diff:.6f}  max_max_diff={max_max_diff:.6f}")
        print(
            f"           Logits: cos={logit_cos:.6f}  max_diff={logit_max_diff:.4f}  top1_agree={'Y' if top1_agree else 'N'}"
        )
        print()

    print("=" * 90)
    if all_pass:
        print(f"  RESULT: PASS — all prompts above threshold {threshold}")
    else:
        print(f"  RESULT: FAIL — some prompts below threshold {threshold}")
    print("=" * 90)

    return all_pass


def main():
    args, extra_args = parse_args()

    model_path = args.model_path or _read_model_name_from_config(args.config)
    if not model_path:
        logger.error("Could not determine model path")
        sys.exit(1)

    # Load prompts
    if args.prompts_file:
        text = Path(args.prompts_file).read_text().strip()
        try:
            prompts = json.loads(text)
        except json.JSONDecodeError:
            prompts = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        prompts = list(DEFAULT_PROMPTS)
    prompts = prompts[: args.num_prompts]
    logger.info("Using %d prompts", len(prompts))

    # Step 1: HF reference
    hf_results = run_hf_extraction(model_path, prompts, args.prompts_file)

    # Step 2: NeMo
    nemo_results = run_nemo_extraction(
        args.config,
        prompts,
        args.nproc,
        args.gate_precision,
        args.lm_head_precision,
        args.prompts_file,
        extra_args,
    )

    # Step 3: Compare
    logger.info("Step 3: Comparing activations ...")
    all_pass = compare_and_report(hf_results, nemo_results, args.threshold)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
