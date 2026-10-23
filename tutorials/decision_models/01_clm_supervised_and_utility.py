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
# # CLM: released baseline, supervised heads and optional utility continuation
#
# CLM keeps its public Qwen3-8B encoder frozen. The native example caches embeddings, updates only projection heads and logit scale, selects a checkpoint using validation and evaluates the same held-out questions. Prepare the framework's GPU environment and model-hub access before enabling training. Running with the default `RUN_TRAINING=False` only creates smoke data and displays commands.
#
# Supervised option training and contrastive InfoNCE are separate objectives. Notebook 03 demonstrates the latter using the same native frozen-embedding interface.

# %%
from __future__ import annotations

import json
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
RUN = TUTORIAL / "runs" / "clm" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA = RUN / "data"
splits = write_dataset(DATA, seed=SEED)
LOGGER.info("Fresh split sizes: %s", {name: len(rows) for name, rows in splits.items()})


def launch(arguments: list[str]) -> None:
    """Show an explicit argv command and execute only when training is enabled."""
    LOGGER.info("%s", shlex.join(arguments))
    if RUN_TRAINING:
        subprocess.run(arguments, cwd=REPO, check=True)


# %% [markdown]
# ## Released baseline and frozen embedding cache

# %%
CACHE = RUN / "embedding_cache"
command = [
    "uv",
    "run",
    "automodel",
    "examples/decision/clm_soc_triage.yaml",
    "--nproc-per-node",
    str(NPROC),
    f"--data.train={DATA / 'train.jsonl'}",
    f"--data.validation={DATA / 'validation.jsonl'}",
    f"--data.test={DATA / 'test.jsonl'}",
    f"--encode.cache_dir={CACHE}",
    "--train.batch_size=16",
    f"--seed={SEED}",
]
launch(command + ["--mode=eval", f"--output_dir={RUN / 'base'}"])

# %% [markdown]
# ## Supervised head adaptation
#
# The recipe selects by validation accuracy and retains its selected checkpoint. The baseline is measured before optimization. Test results must not be used to tune the configuration.

# %%
launch(command + ["--mode=train", "--loss.mode=sft", f"--output_dir={RUN / 'sft'}"])

# %% [markdown]
# ## Optional exact expected utility
#
# Known per-option utilities allow the expectation to be summed exactly. Continue from the selected SFT head, with a reference KL term and supervised anchor. Utility can improve while accuracy or probability quality worsens; none of these gains is guaranteed.

# %%
if ENABLE_UTILITY:
    launch(
        command
        + [
            "--mode=train",
            "--loss.mode=rl",
            f"--loss.rl_objective={UTILITY_OBJECTIVE}",
            f"--heads.init_checkpoint={RUN / 'sft/heads.pt'}",
            "--train.eval_metric=reward",
            "--optimizer.lr=0.0001",
            f"--output_dir={RUN / 'rl'}",
        ]
    )

# %% [markdown]
# ## Held-out reporting
#
# Report decision accuracy, exact record match, NLL, Brier and utility on the same test population. Keep the split manifest unchanged; the reporting helper checks counts, not record identity. Record seed, data/model revisions, trainable scope and topology. Repeat training seeds for research claims; this small fixture only checks mechanics.

# %%
if RUN_TRAINING:
    stages = ("base", "sft", "rl") if ENABLE_UTILITY else ("base", "sft")
    rows = summarize_run(RUN, stages=stages)
    write_summary(rows, RUN / "heldout-summary.csv")
    LOGGER.info("Held-out measurements: %s", rows)

# %% [markdown]
# ## Inspect locally measured learning curves
#
# These plots show the logged SFT/contrastive objective and saved validation observations. Validation cadence and x-axis units are explicit. A per-update objective can be noisy and main-rank local; it is not a full training-set loss or a held-out loss. Utility continuation has a different objective and should be plotted separately.

# %%
if RUN_TRAINING:
    from nemo_automodel.shared.import_utils import safe_import

    available, plt = safe_import("matplotlib.pyplot")
    if not available:
        raise ModuleNotFoundError("Install optional matplotlib to draw local learning curves")
    history = [json.loads(line) for line in (RUN / "sft/training_log.jsonl").read_text().splitlines()]
    updates = [row for row in history if "loss" in row and "step" in row]
    validations = [row for row in history if "val_accuracy" in row]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot([row["step"] for row in updates], [row["loss"] for row in updates])
    axes[0].set(xlabel="Optimizer step", ylabel="Logged training objective")
    axes[1].plot([row["epoch"] for row in validations], [row["val_accuracy"] for row in validations], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Validation decision accuracy")
    figure.tight_layout()
    figure.savefig(RUN / "learning-curve.png")
