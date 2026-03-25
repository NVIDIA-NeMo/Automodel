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

    # Recommended — the CLI handles torchrun internally:
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

Launcher selection is automatic based on the presence of ``slurm:``, ``k8s:``,
or ``nemo_run:`` sections in the YAML.

When launched via ``torchrun``, the CLI detects the existing distributed
environment and runs the recipe in-process on each worker instead of
re-spawning torchrun.
"""

import argparse
import logging
import sys
from pathlib import Path

from nemo_automodel.cli.utils import load_yaml, resolve_recipe_name

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="automodel",
        description=("CLI for NeMo AutoModel recipes. The YAML config specifies both the recipe and the launcher."),
    )
    parser.add_argument(
        "config",
        metavar="<config.yaml>",
        type=Path,
        help="Path to YAML configuration file (must specify a recipe target)",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=int,
        default=None,
        help=(
            "Number of workers per node for local/interactive jobs. "
            "Ignored when a slurm/k8s/nemo_run section is present."
        ),
    )
    return parser


def main():
    """CLI for running recipes with NeMo-AutoModel.

    Supports interactive (local), SLURM, Kubernetes, and NeMo-Run launchers.

    Returns:
        int: Job's exit code.
    """
    args, extra = build_parser().parse_known_args()
    config_path = args.config.resolve()
    logger.info("Config: %s", config_path)
    config = load_yaml(config_path)

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
            "See BREAKING_CHANGES.md for the full list of available recipe targets."
        )
        sys.exit(1)

    try:
        recipe_target = resolve_recipe_name(raw_target)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Recipe: %s", recipe_target)

    if slurm_config := config.pop("slurm", None):
        logger.info("Launching job via SLURM")
        from nemo_automodel.components.launcher.slurm.launcher import SlurmLauncher

        return SlurmLauncher().launch(config, config_path, recipe_target, slurm_config, extra)

    elif k8s_config := config.pop("k8s", None):
        logger.info("Launching job via Kubernetes (PyTorchJob)")
        from nemo_automodel.components.launcher.k8s.launcher import K8sLauncher

        return K8sLauncher().launch(config, config_path, recipe_target, k8s_config, extra)

    elif nemo_run_config := config.pop("nemo_run", None):
        logger.info("Launching job via NeMo-Run")
        from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
        from nemo_automodel.components.launcher.nemo_run.launcher import NemoRunLauncher

        cfg = parse_args_and_load_config(str(config_path))
        return NemoRunLauncher().launch(cfg, config_path, recipe_target, nemo_run_config, extra)

    else:
        logger.info("Launching job interactively (local)")
        from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
        from nemo_automodel.components.launcher.interactive import InteractiveLauncher

        cfg = parse_args_and_load_config(str(config_path))
        return InteractiveLauncher().launch(cfg, config_path, recipe_target, args.nproc_per_node, extra)


if __name__ == "__main__":
    main()
