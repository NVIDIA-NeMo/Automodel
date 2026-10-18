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
# # Build fictional typed-decision smoke data
#
# This CPU-only notebook creates a new deterministic SOC-style fixture with `choice`, `score` and `noul` questions, exact labels and complete per-option utilities. It makes no model calls. Identifiers use reserved `.example`/`example.com` names and RFC 5737 IP addresses. The policy is a tutorial fixture, not an operational security runbook.
#
# The checked-in `fixtures/smoke/` has 16/8/8 records for schema tests. Training notebooks generate a larger fresh 128/32/32 split. IDs and exact states are disjoint; recurring scenario templates are intentional and do not represent a production generalization test.

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

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("decision_tutorial")
RUN_TRAINING = False  # Set True only in a prepared GPU environment.
ENABLE_UTILITY = False  # Continue from the selected SFT checkpoint when True.
UTILITY_OBJECTIVE = "expected_reward"  # Or "grpo" for sampled native policy continuation.
NPROC = 1  # Choose the GPU processes available in your allocation.
SEED = 7
RUN = TUTORIAL / "runs" / "data" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA = RUN / "data"
splits = write_dataset(DATA, seed=SEED)
LOGGER.info("Fresh split sizes: %s", {name: len(rows) for name, rows in splits.items()})


def launch(arguments: list[str]) -> None:
    """Show an explicit argv command and execute only when training is enabled."""
    LOGGER.info("%s", shlex.join(arguments))
    if RUN_TRAINING:
        subprocess.run(arguments, cwd=REPO, check=True)


# %% [markdown]
# ## Inspect the policy and validate the boundaries
#
# The generator returns the schema, gold answers and utility table together. Labels remain outside `state`. Use training records for fitting, validation records for model selection and a sealed test split for final reporting.

# %%
from decision_tutorial.data import QUESTIONS, validate_splits

validate_splits(splits)
QUESTIONS, splits["train"][0]

# %% [markdown]
# ## Save the experiment identity
#
# The manifest records split sizes, seed and file hashes. A new seed or policy defines a new experiment. Existing directories are rejected so a dataset cannot be replaced silently. Notebook 05 shows an optional Data Designer workflow; no API key is needed here.

# %%
manifest = json.loads((DATA / "manifest.json").read_text())
manifest
