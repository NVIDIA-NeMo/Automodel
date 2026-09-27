#!/usr/bin/env python3
"""Measure Titans fast-weight updates and their effect on real checkpoint logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from nemo_automodel.components.models.titans.layers import NeuralMemory


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    """Summarize one gate tensor in fp32."""
    value = value.detach().float()
    return {
        "min": float(value.min()),
        "mean": float(value.mean()),
        "max": float(value.max()),
        "std": float(value.std()) if value.numel() > 1 else 0.0,
    }


def difference_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int | bool]:
    """Summarize exact and numerical differences between tensors."""
    difference = actual.detach().float() - expected.detach().float()
    return {
        "exact_equal": bool(torch.equal(actual, expected)),
        "changed_values": int(difference.ne(0).sum()),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "l2": float(torch.linalg.vector_norm(difference)),
    }


@torch.no_grad()
def diagnose_memory(memory: NeuralMemory, hidden: torch.Tensor) -> dict[str, Any]:
    """Measure gates, state updates, and output effects for one memory layer."""
    beta = memory.b_proj(hidden).sigmoid()
    keep = memory._decay_gate(memory.a_proj(hidden)).exp()
    eta = memory.m_proj(hidden).sigmoid() if memory.momentum else torch.zeros_like(beta)
    enabled_output, enabled_state = memory(hidden, return_state=True, enable_ttt_updates=True)
    disabled_output, disabled_state = memory(hidden, return_state=True, enable_ttt_updates=False)
    weight_deltas = [
        difference_stats(enabled, disabled)
        for enabled, disabled in zip(enabled_state.weights, disabled_state.weights)
    ]
    momentum_norms = [float(torch.linalg.vector_norm(value.float())) for value in enabled_state.momentum]
    return {
        "input_shape": list(hidden.shape),
        "beta": tensor_stats(beta),
        "forget_keep": tensor_stats(keep),
        "momentum_eta": tensor_stats(eta),
        "output_difference": difference_stats(enabled_output, disabled_output),
        "fast_weight_differences": weight_deltas,
        "fast_momentum_l2": momentum_norms,
    }


def parse_args() -> argparse.Namespace:
    """Parse checkpoint diagnostic arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, -1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run diagnostics on selected layers of one real checkpoint."""
    args = parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    from titans_generate import TitansGenerator

    generator = TitansGenerator(args.checkpoint, args.tokenizer, args.device)
    model = generator.model.eval()
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, args.tokens),
        generator=rng,
        device=args.device,
    )
    layer_count = len(model.model.layers)
    layer_indices = sorted({index % layer_count for index in args.layers})
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def capture(index: int):
        def hook(_module, inputs):
            captured.setdefault(index, inputs[0].detach())

        return hook

    for index in layer_indices:
        layer = model.model.layers[index]
        if not hasattr(layer, "memory"):
            continue
        hooks.append(layer.memory.register_forward_pre_hook(capture(index)))
    with torch.no_grad():
        enabled_logits = model(input_ids, enable_ttt_updates=True).logits
        disabled_logits = model(input_ids, enable_ttt_updates=False).logits
    for hook in hooks:
        hook.remove()

    layers = {}
    for index, hidden in captured.items():
        layers[str(index)] = diagnose_memory(model.model.layers[index].memory, hidden)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture_variant": model.config.architecture_variant,
        "tokens": args.tokens,
        "seed": args.seed,
        "logit_difference": difference_stats(enabled_logits, disabled_logits),
        "layers": layers,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
