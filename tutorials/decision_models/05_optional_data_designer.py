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
# # Optional fictional prose with NeMo Data Designer
#
# This optional example adds model-generated paraphrases to deterministic source observations. Gold labels and utilities stay fixed by the policy. A writer renders prose, a judge filters unsupported claims, and identifier checks reject non-example addresses. Source-case IDs are assigned to splits before generation; repeated cases stay in their original split and are retained only once.
#
# Install the optional notebook dependency in your prepared environment with `uv pip install data-designer==0.9.4 pandas pyarrow`. Set your own `NVIDIA_API_KEY` for the configured public provider. No key or service is needed unless `RUN_GENERATION=True`. The default notebook performs no network calls.
#
# Inspect the preview before enabling the larger generation. The automated judge and identifier checks do not prove factual accuracy or absence of personal data; review generated text before using or sharing it.

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
RUN = TUTORIAL / "runs" / "generated-prose" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA = RUN / "data"
splits = write_dataset(DATA, seed=SEED)
LOGGER.info("Fresh split sizes: %s", {name: len(rows) for name, rows in splits.items()})


def launch(arguments: list[str]) -> None:
    """Show an explicit argv command and execute only when training is enabled."""
    LOGGER.info("%s", shlex.join(arguments))
    if RUN_TRAINING:
        subprocess.run(arguments, cwd=REPO, check=True)


# %%
import copy
import hashlib
import os

from decision_tutorial.data import validate_splits, validate_summary

RUN_GENERATION = False
REVIEWED_PREVIEW = False  # Set only after inspecting the preview; this is a new experiment.
MODEL_ID = "nvidia/nemotron-3.5-lightning-30b-a3b"
JUDGE_MODEL_ID = MODEL_ID  # A separate compatible judge model can be configured.
NUM_GENERATIONS = 512
source_records = [record for records in splits.values() for record in records]
source_by_event = {str(record["state"]["event_id"]): record for record in source_records}
split_by_id = {record["id"]: split for split, records in splits.items() for record in records}

if RUN_GENERATION:
    from nemo_automodel.shared.import_utils import safe_import

    available_config, dd = safe_import("data_designer.config")
    available_interface, interface = safe_import("data_designer.interface")
    if not available_config or not available_interface:
        raise ModuleNotFoundError("Install the optional data-designer notebook dependency first")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise ValueError("Set NVIDIA_API_KEY for the configured public provider")
    designer = interface.DataDesigner()
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(alias="writer", model=MODEL_ID, provider="nvidia"),
            dd.ModelConfig(alias="judge", model=JUDGE_MODEL_ID, provider="nvidia"),
        ]
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="source_state",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[json.dumps(record["state"], sort_keys=True) for record in source_records]
            ),
        )
    )
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="summary",
            model_alias="writer",
            prompt="""Rewrite the fictional source observation in two sentences without adding evidence.
Use only the identifiers supplied in this JSON. Do not add people, organizations, credentials,
email addresses, domains or IP addresses. Do not recommend an action or state a verdict/severity.
Source: {{ source_state }}
Return only the summary.""",
        )
    )
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="support_check",
            model_alias="judge",
            prompt="""Source fictional observation: {{ source_state }}
Candidate summary: {{ summary }}
Are all factual claims in the summary supported by the source, with no added identifiers,
verdict, severity or response recommendation? Reply only pass or fail.""",
        )
    )
    designer.validate(builder)
    preview = designer.preview(builder, num_records=2)
    preview.display_sample_record()

# %% [markdown]
# ## Generate, filter and freeze new splits
#
# `REVIEWED_PREVIEW=True` enables this cell after preview inspection. Retain rejected raw generation only in your local ignored run directory. The output manifest describes a new dataset; it makes no claim to reproduce another task. Select checkpoints using its validation split and keep its test split sealed.

# %%
if RUN_GENERATION and REVIEWED_PREVIEW:
    result = designer.create(builder, num_records=NUM_GENERATIONS, dataset_name="fictional-decision-prose")
    generated = result.load_dataset()
    destination = RUN / "generated-data"
    if destination.exists():
        raise FileExistsError("Choose a fresh experiment directory")
    accepted = {name: [] for name in splits}
    used_ids = set()
    for row in generated.to_dict(orient="records"):
        if str(row["support_check"]).strip().lower() != "pass" or not isinstance(row["summary"], str):
            continue
        original_state = json.loads(row["source_state"])
        record = copy.deepcopy(source_by_event[original_state["event_id"]])
        if record["id"] in used_ids:
            continue  # Keep a source case in its assigned split, once only.
        try:
            validate_summary(row["summary"])
        except ValueError:
            continue  # Reject added identifier/key patterns rather than hiding them by rewriting.
        record["state"]["generated_summary"] = row["summary"]
        accepted[split_by_id[record["id"]]].append(record)
        used_ids.add(record["id"])
    validate_splits(accepted)
    destination.mkdir()
    generated.to_parquet(RUN / "generated-prose.parquet", index=False)
    hashes = {}
    for split, records in accepted.items():
        path = destination / f"{split}.jsonl"
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "kind": "new-generated-prose-experiment",
        "writer": MODEL_ID,
        "judge": JUDGE_MODEL_ID,
        "seed_source_sha256": json.loads((DATA / "manifest.json").read_text())["sha256"],
        "counts": {name: len(records) for name, records in accepted.items()},
        "sha256": hashes,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Generated split counts: %s", manifest["counts"])
