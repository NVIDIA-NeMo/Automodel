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


# %% [markdown]
# # D1 readout adaptation
#
# This experimental workflow freezes the native backbone and adapts additive answer-token rows for D1-3B or the marker head for D1 Omni. It runs a released baseline, supervised readout adaptation and optional utility continuation, with validation-selected checkpoints and a separate heldout report.
#
# The implementation and tutorial follow this repository's Apache 2.0 license. Model loading uses pinned public publisher downloads; downloaded weights and remote publisher components retain their [publisher license](https://www.liquid.ai/lfm-license). No model weights, executed outputs or measured performance are bundled here.
#
# Notebook source/default-path checks are CPU-only; this notebook has not been executed end-to-end. Separate released-checkpoint GPU, publisher and media acceptance is documented in the [D1 acceptance notes](../../tests/functional_tests/decision_models/d1/README.md), with its precision and fixture limitations. It uses text-only fictional observations and makes no image/audio adaptation claim. Checkpoint self-reload, sampled publisher agreement and full numerical equivalence remain separate checks. Set `RUN_TRAINING=True` only in a prepared single-GPU environment.

# %%
from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / "nemo_automodel").is_dir())
TUTORIAL = REPO / "tutorials/decision_models"
sys.path.insert(0, str(TUTORIAL))
from decision_tutorial.data import write_dataset
from decision_tutorial.reporting import summarize_run, write_summary

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("decision_tutorial")
RUN_TRAINING = False  # Set True only in a prepared GPU environment.
ENABLE_UTILITY = False  # Continue from the selected SFT checkpoint when True.
UTILITY_OBJECTIVE = "expected_reward"  # Or "grpo" for sampled native policy continuation.
NPROC = 1  # Choose the GPU processes available in your allocation.
SEED = 7
RUN = TUTORIAL / "runs" / "d1" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA = RUN / "data"
splits = write_dataset(DATA, seed=SEED)
LOGGER.info("Fresh split sizes: %s", {name: len(rows) for name, rows in splits.items()})


def launch(arguments: list[str]) -> None:
    """Show an explicit argv command and execute only when training is enabled."""
    LOGGER.info("%s", shlex.join(arguments))
    if RUN_TRAINING:
        subprocess.run(arguments, cwd=REPO, check=True)


# %%
MODEL = "d1-omni"  # Or "d1-3b".
command = [
    "uv",
    "run",
    "python",
    "examples/decision/d1_posttrain.py",
    "--model",
    MODEL,
    "--data-dir",
    str(DATA),
    "--cache-dir",
    str(RUN / "features"),
    "--seed",
    str(SEED),
]
launch(command + ["--mode", "base", "--output-dir", str(RUN / "base")])
launch(command + ["--mode", "sft", "--input-dir", str(RUN / "base"), "--output-dir", str(RUN / "sft")])
if ENABLE_UTILITY:
    launch(command + ["--mode", "rl", "--input-dir", str(RUN / "sft"), "--output-dir", str(RUN / "rl")])

# %%
if RUN_TRAINING:
    stages = ("base", "sft", "rl") if ENABLE_UTILITY else ("base", "sft")
    rows = summarize_run(RUN, stages=stages)
    write_summary(rows, RUN / "heldout-summary.csv")
    LOGGER.info("Local D1 measurements: %s", rows)
