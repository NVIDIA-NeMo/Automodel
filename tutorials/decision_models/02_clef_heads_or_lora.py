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
# # CLEF: frozen-head adaptation or LoRA plus schema head
#
# Select one trainable scope for each fresh experiment: `HEAD_ONLY=True` freezes the backbone and updates the joint schema head; `False` updates backbone LoRA plus that head. The same baseline, supervised and optional utility stages use native framework recipes. A 27B backbone requires suitable GPU memory even when it is frozen. Choose processes/topology for your prepared allocation; smaller smoke data does not reduce model size.
#
# The saved example source contains no measured outcome. Head-only and LoRA experiments have different budgets and should be compared as separate controlled runs.

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
RUN = TUTORIAL / "runs" / "clef" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA = RUN / "data"
splits = write_dataset(DATA, seed=SEED)
LOGGER.info("Fresh split sizes: %s", {name: len(rows) for name, rows in splits.items()})


def launch(arguments: list[str]) -> None:
    """Show an explicit argv command and execute only when training is enabled."""
    LOGGER.info("%s", shlex.join(arguments))
    if RUN_TRAINING:
        subprocess.run(arguments, cwd=REPO, check=True)


# %% [markdown]
# ## Released baseline and validation-selected SFT
#
# The supervised objective combines label-smoothed CE, Brier and ordinal terms where configured. Inspect the YAML/CLI values before training. The native loop selects checkpoints with validation accuracy, including its starting checkpoint.

# %%
HEAD_ONLY = True  # False selects backbone LoRA plus the schema head.
config = "clef_soc_triage_heads.yaml" if HEAD_ONLY else "clef_soc_triage_lora.yaml"
command = [
    "uv",
    "run",
    "automodel",
    f"examples/decision/{config}",
    "--nproc-per-node",
    str(NPROC),
    f"--data.train={DATA / 'train.jsonl'}",
    f"--data.validation={DATA / 'validation.jsonl'}",
    f"--data.test={DATA / 'test.jsonl'}",
    f"--train.global_batch_size={2 * NPROC}",
    f"--seed={SEED}",
]
launch(command + ["--mode=eval", f"--output_dir={RUN / 'base'}"])
launch(command + ["--mode=train", "--loss.mode=sft", f"--output_dir={RUN / 'sft'}"])

# %% [markdown]
# ## Optional utility continuation
#
# Keep the trainable scope consistent when loading the selected SFT stage. Expected utility, its reference KL penalty and supervised anchor are configured independently of the SFT loss. They do not guarantee calibrated probabilities.

# %%
if ENABLE_UTILITY:
    launch(
        command
        + [
            "--mode=train",
            "--loss.mode=rl",
            f"--loss.rl_objective={UTILITY_OBJECTIVE}",
            f"--init_from={RUN / 'sft/LOWEST_VAL'}",
            "--train.eval_metric=reward",
            f"--output_dir={RUN / 'rl'}",
        ]
    )

# %% [markdown]
# ## Local held-out report
#
# Inspect accuracy, exact record match, utility and proper scoring rules together. The fixture is a smoke test; run multiple seeds and use a more representative task before making effectiveness claims.

# %%
if RUN_TRAINING:
    stages = ("base", "sft", "rl") if ENABLE_UTILITY else ("base", "sft")
    rows = summarize_run(RUN, stages=stages)
    write_summary(rows, RUN / "heldout-summary.csv")
    LOGGER.info("Trainable scope: %s; measurements: %s", "head" if HEAD_ONLY else "LoRA + head", rows)

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
    axes[1].plot([row["step"] for row in validations], [row["val_accuracy"] for row in validations], marker="o")
    axes[1].set(xlabel="Step", ylabel="Validation decision accuracy")
    figure.tight_layout()
    figure.savefig(RUN / "learning-curve.png")

# %% [markdown]
# ## Optional local merged export
#
# Native stages retain the selected head and any adapters for reload. Both head-only and LoRA stages can produce a publisher-loadable Hugging Face model plus joint schema head; LoRA adapters are merged when present. This is a new local artifact subject to the model license; no weights are included in this tutorial. Export does not establish publisher or cross-framework numerical equivalence.

# %%
EXPORT_MODEL = False  # A full merged model may require substantial local disk/memory.
if EXPORT_MODEL:
    launch(
        [
            "uv",
            "run",
            "python",
            "examples/decision/export_clef_release.py",
            "--stage",
            str(RUN / "sft/LOWEST_VAL"),
            "--out",
            str(RUN / "release"),
        ]
    )
