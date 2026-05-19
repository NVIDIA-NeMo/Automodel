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

"""Unified CLI entry-point for NeMo AutoModel.

Usage
-----
::

    # Zero-YAML LoRA fine-tuning:
    automodel Qwen/Qwen2.5-1.5B-Instruct openai-chat-data.jsonl
    automodel Qwen/Qwen2.5-VL-3B-Instruct openai-vlm-chat-data.jsonl

    # Explicit recipe config; the CLI handles torchrun internally:
    automodel <config.yaml> [--nproc-per-node N] [--key.subkey=override ...]

    # Also supported — external torchrun launch:
    torchrun --nproc-per-node N -m nemo_automodel.cli.app <config.yaml> [--key.subkey=override ...]

    # Convenience wrapper for development (not installed):
    python app.py <config.yaml> [--nproc-per-node N] [--key.subkey=override ...]

The YAML config must specify which recipe class to instantiate.  All three
forms are accepted::

    recipe: TrainFinetuneRecipeForNextTokenPrediction        # bare class name
    recipe: nemo_automodel.recipes.llm.train_ft.TrainFin...  # fully-qualified
    recipe:
      _target_: nemo_automodel.recipes.llm.train_ft.Trai...  # Hydra-style

For SLURM clusters, use ``sbatch slurm.sub`` directly (see the reference
script at the repo root).  Add a ``skypilot:`` or ``nemo_run:`` section
in the YAML for those launchers.

When launched via ``torchrun``, the CLI detects the existing distributed
environment and runs the recipe in-process on each worker instead of
re-spawning torchrun.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

from nemo_automodel.cli.quickstart import build_lora_config
from nemo_automodel.cli.utils import load_yaml, resolve_recipe_name
from nemo_automodel.components.config.loader import resolve_yaml_env_vars

# When launched via external torchrun each worker imports this module.
# Suppress non-rank-0 CLI output before setup_logging installs RankFilter.
if int(os.environ.get("RANK", "0")) > 0:
    logging.disable(logging.CRITICAL)
else:
    logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="automodel",
        description=(
            "CLI for NeMo AutoModel. Run zero-YAML LoRA fine-tuning with "
            "`automodel <hf-model-id> <openai-chat-data.jsonl>`, or pass a YAML recipe config."
        ),
    )
    parser.add_argument(
        "inputs",
        metavar="<config.yaml | hf-model-id data.jsonl>",
        nargs="+",
        help=(
            "Either a YAML configuration file, or a Hugging Face model id followed by "
            "OpenAI-format chat JSONL data. Multimodal image/video content is routed to the VLM recipe."
        ),
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=int,
        default=None,
        help=(
            "Number of workers per node for local/interactive jobs. "
            "Ignored when a skypilot/nemo_run section is present."
        ),
    )
    return parser


def _looks_like_yaml_config(value: str) -> bool:
    return Path(value).suffix.lower() in {".yaml", ".yml"}


def _infer_local_worker_count(nproc_per_node: int | None) -> int | None:
    if nproc_per_node is not None:
        return nproc_per_node
    try:
        from torch.distributed.run import determine_local_world_size

        return determine_local_world_size(nproc_per_node="gpu")
    except Exception:
        return None


def _write_generated_config(config: dict) -> Path:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="automodel-lora-",
        suffix=".yaml",
        delete=False,
    ) as fp:
        yaml.safe_dump(config, fp, sort_keys=False)
        return Path(fp.name)


def _resolve_invocation(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, dict, bool]:
    inputs = args.inputs
    if len(inputs) == 1:
        config_path = Path(inputs[0]).resolve()
        return config_path, load_yaml(config_path), False

    if len(inputs) == 2:
        if _looks_like_yaml_config(inputs[0]):
            parser.error("YAML config mode accepts exactly one positional argument.")
        model_id, data_path = inputs
        nproc = _infer_local_worker_count(args.nproc_per_node)
        try:
            config = build_lora_config(model_id, data_path, nproc_per_node=nproc)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        config_path = _write_generated_config(config)
        logger.info("Generated LoRA config: %s", config_path)
        return config_path, config, True

    parser.error("Expected either `automodel <config.yaml>` or `automodel <hf-model-id> <openai-chat-data.jsonl>`.")


def main():
    """CLI for running recipes with NeMo-AutoModel.

    Supports interactive (local), SkyPilot, and NeMo-Run launchers.
    For SLURM, use ``sbatch slurm.sub`` directly.

    Returns:
        int: Job's exit code.
    """
    parser = build_parser()
    args, extra = parser.parse_known_args()
    config_path, config, generated_config = _resolve_invocation(args, parser)
    logger.info("Config: %s", config_path)

    recipe_section = config.get("recipe")
    if isinstance(recipe_section, str) and recipe_section.strip():
        raw_target = recipe_section.strip()
    elif isinstance(recipe_section, dict) and "_target_" in recipe_section:
        raw_target = recipe_section["_target_"]
    else:
        logger.error(
            "YAML config must specify a recipe target.\n"
            "Examples:\n"
            "  recipe: TrainFinetuneRecipeForNextTokenPrediction\n"
            "  recipe: nemo_automodel.recipes.llm.train_ft."
            "TrainFinetuneRecipeForNextTokenPrediction\n"
            "  recipe:\n"
            "    _target_: nemo_automodel.recipes.llm.train_ft."
            "TrainFinetuneRecipeForNextTokenPrediction\n\n"
            "See docs/BREAKING_CHANGES.md for the full list of available recipe targets."
        )
        sys.exit(1)

    try:
        recipe_target = resolve_recipe_name(raw_target)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Recipe: %s", recipe_target)

    try:
        if skypilot_config := config.pop("skypilot", None):
            logger.info("Launching job via SkyPilot")
            from nemo_automodel.components.launcher.skypilot.launcher import SkyPilotLauncher

            return SkyPilotLauncher().launch(
                config,
                config_path,
                recipe_target,
                resolve_yaml_env_vars(skypilot_config),
                extra,
            )

        elif nemo_run_config := config.pop("nemo_run", None):
            logger.info("Launching job via NeMo-Run")
            from nemo_automodel.components.launcher.nemo_run.launcher import NemoRunLauncher

            return NemoRunLauncher().launch(config, config_path, recipe_target, nemo_run_config, extra)

        else:
            logger.info("Launching job interactively (local)")
            from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
            from nemo_automodel.components.launcher.interactive import InteractiveLauncher

            cfg = parse_args_and_load_config(str(config_path), argv=extra)
            return InteractiveLauncher().launch(cfg, config_path, recipe_target, args.nproc_per_node, extra)
    finally:
        if generated_config:
            try:
                config_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove generated config: %s", config_path)


if __name__ == "__main__":
    main()
