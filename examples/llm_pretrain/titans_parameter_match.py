#!/usr/bin/env python3
"""Solve the nearest SwiGLU width for a Titans parameter-matched control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from nemo_automodel.components.models.titans.config import TitansConfig
from nemo_automodel.components.models.titans.model import TitansForCausalLM

_NON_CONFIG_KEYS = {"_target_", "architectures"}


def load_model_config(recipe_path: str | Path) -> dict[str, Any]:
    """Load only TitansConfig fields from a training recipe."""
    recipe = yaml.safe_load(Path(recipe_path).read_text())
    return {key: value for key, value in recipe["model"]["config"].items() if key not in _NON_CONFIG_KEYS}


def parameter_count(config_values: dict[str, Any]) -> int:
    """Count model parameters without allocating storage."""
    with torch.device("meta"):
        model = TitansForCausalLM(TitansConfig(**config_values))
    return sum(parameter.numel() for parameter in model.parameters())


def solve_control_width(
    target_config: dict[str, Any],
    *,
    architecture_variant: str,
    attention_segment_size: int = 512,
) -> dict[str, int | float | str]:
    """Return the closest scalar SwiGLU width and an auditable mismatch bound.

    A shared integer width changes the model by ``3 * hidden_size *
    num_hidden_layers`` parameters at a time, so exact equality is not always
    representable. The selected width minimizes absolute error; callers must
    retain the reported delta instead of describing an inexact match as exact.
    """

    if architecture_variant not in {"local_attention", "full_attention"}:
        raise ValueError("Control architecture must be 'local_attention' or 'full_attention'.")
    target_parameters = parameter_count(target_config)
    base_width = int(target_config["intermediate_size"])
    control = target_config | {
        "architecture_variant": architecture_variant,
        "attention_segment_size": attention_segment_size,
        "num_longterm_memory_tokens": 0,
        "num_persistent_memory_tokens": 0,
        "memory_layer_indices": None,
    }
    base_parameters = parameter_count(control)
    parameters_per_width = 3 * int(control["hidden_size"]) * int(control["num_hidden_layers"])
    ideal_width = base_width + (target_parameters - base_parameters) / parameters_per_width
    candidates = {max(1, int(ideal_width)), max(1, int(ideal_width) + 1)}
    counts = {
        width: parameter_count(control | {"intermediate_size": width})
        for width in candidates
    }
    width = min(candidates, key=lambda candidate: (abs(counts[candidate] - target_parameters), candidate))
    matched_parameters = counts[width]
    delta = matched_parameters - target_parameters
    return {
        "architecture_variant": architecture_variant,
        "intermediate_size": width,
        "target_parameters": target_parameters,
        "control_parameters": matched_parameters,
        "parameter_delta": delta,
        "relative_error": abs(delta) / target_parameters,
        "width_granularity_parameters": parameters_per_width,
    }


def main() -> None:
    """Solve and print the nearest parameter-matched control width."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_recipe", type=Path)
    parser.add_argument(
        "--architecture-variant",
        choices=("local_attention", "full_attention"),
        default="local_attention",
    )
    parser.add_argument("--attention-segment-size", type=int, default=512)
    args = parser.parse_args()
    result = solve_control_width(
        load_model_config(args.target_recipe),
        architecture_variant=args.architecture_variant,
        attention_segment_size=args.attention_segment_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
