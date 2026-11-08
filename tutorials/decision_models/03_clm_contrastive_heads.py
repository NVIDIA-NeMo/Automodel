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
# # Frozen CLM features with supervised contrastive head adaptation
#
# This example reuses the native released-baseline embedding cache, then updates only the public CLM projection heads on CPU. No backbone gradient is computed. InfoNCE uses aligned state/positive-action pairs; identical positive action text is masked as an in-batch false negative. Validation decision accuracy selects the checkpoint, with the released head eligible. Test accuracy is measured separately and improvement is not asserted.
#
# Enable `RUN_TRAINING` only after preparing a GPU for the baseline embedding pass. Training the cached projection heads then runs on CPU. This is an alternative to the option-supervised SFT in notebook 01.

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
RUN = TUTORIAL / "runs" / "clm-contrastive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA = RUN / "data"
splits = write_dataset(DATA, seed=SEED)
LOGGER.info("Fresh split sizes: %s", {name: len(rows) for name, rows in splits.items()})


def launch(arguments: list[str]) -> None:
    """Show an explicit argv command and execute only when training is enabled."""
    LOGGER.info("%s", shlex.join(arguments))
    if RUN_TRAINING:
        subprocess.run(arguments, cwd=REPO, check=True)


# %% [markdown]
# ## Native released baseline and immutable local cache

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
# ## Train and validate the released projection heads
#
# The tiny fixture repeats templates by design. Test questions remain outside fitting and selection. A falling contrastive objective alone does not establish better held-out decisions; retain NLL, Brier and utility in the final report.

# %%
if RUN_TRAINING:
    import copy

    import torch

    from nemo_automodel.components.datasets.decision.records import load_records, question_targets
    from nemo_automodel.components.eval.decision.metrics import DecisionEvaluator
    from nemo_automodel.components.loss.decision import group_masked_bidirectional_infonce
    from nemo_automodel.components.models.clm.heads import EmbeddingTable, clm_batch, load_clm_heads, save_clm_heads
    from nemo_automodel.components.models.clm.prompting import clm_pairs

    caches = list(CACHE.glob("embeddings_*.pt"))
    if len(caches) != 1:
        raise ValueError("Expected exactly one cache from this fresh baseline run")
    table = EmbeddingTable.load(caches[0])
    records = {split: load_records(DATA / f"{split}.jsonl") for split in ("train", "validation", "test")}
    head = load_clm_heads(map_location="cpu")

    @torch.no_grad()
    def evaluate_records(items: list[dict[str, object]]) -> dict[str, float]:
        """Score declared options using the native head and evaluator.

        The captured table holds frozen [texts, hidden] embeddings. Native
        batches contain [questions, hidden] states and [options, hidden]
        candidates; question_logits returns ragged [options_i] tensors.
        """
        head.eval()
        states, options, counts, orders = clm_batch(items, table, "cpu")
        logits = head.question_logits(states, options, counts)
        targets = [question_targets(record, order) for record, order in zip(items, orders)]
        probabilities, cursor = [], 0
        for record_targets in targets:
            probabilities.append(
                [logits[cursor + index].float().softmax(-1).tolist() for index in range(len(record_targets))]
            )
            cursor += len(record_targets)
        evaluator = DecisionEvaluator()
        evaluator.add_batch(items, targets, probabilities)
        return evaluator.summary()["overall"]

    pairs = []
    for record in records["train"]:
        for name, (state, keys, actions) in clm_pairs(record).items():
            positive = actions[keys.index(str(record["answers"][name]))]
            pairs.append((state, positive))
    positive_ids = {text: index for index, text in enumerate(sorted({action for _, action in pairs}))}
    best_score = evaluate_records(records["validation"])["accuracy"]
    best_state = copy.deepcopy(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003)
    history = []
    for epoch in range(5):
        order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        losses = []
        head.train()
        for start in range(0, len(order), 32):
            batch = [pairs[index] for index in order[start : start + 32]]
            if len(batch) < 2:
                continue  # InfoNCE needs at least two aligned pairs.
            states = table.lookup([state for state, _ in batch])
            actions = table.lookup([action for _, action in batch])
            groups = torch.tensor([positive_ids[action] for _, action in batch])
            optimizer.zero_grad(set_to_none=True)
            loss = group_masked_bidirectional_infonce(
                head.project_states(states), head.project_actions(actions), head.logit_scale, groups
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_accuracy = evaluate_records(records["validation"])["accuracy"]
        history.append(
            {"epoch": epoch + 1, "objective": sum(losses) / len(losses), "validation_accuracy": validation_accuracy}
        )
        if validation_accuracy > best_score:
            best_score = validation_accuracy
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    selected = RUN / "head-adaptation"
    selected.mkdir()
    save_clm_heads(head, selected / "heads.pt")
    (selected / "metrics_test.json").write_text(json.dumps({"overall": evaluate_records(records["test"])}, indent=2))
    (selected / "history.json").write_text(json.dumps(history, indent=2))
    rows = summarize_run(RUN, stages=("base", "head-adaptation"))
    write_summary(rows, RUN / "heldout-summary.csv")
    LOGGER.info("Contrastive head adaptation: %s", rows)

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
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(epochs, [row["objective"] for row in history])
    axes[0].set(xlabel="Epoch", ylabel="Mean training InfoNCE")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Validation decision accuracy")
    figure.tight_layout()
    figure.savefig(RUN / "learning-curve.png")
