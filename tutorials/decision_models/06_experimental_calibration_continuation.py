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
# # Experimental CLM calibration-aware policy continuation
#
# This is an explicit caller of the framework's experimental `rlcr_loss`, inspired by [JevAny](https://github.com/SimpleJev/JevAny/blob/main/recipes/rlcr.toml). It is separate from the exact expected-utility objective in notebook 01 and is not an established publisher recipe.
#
# The loss samples Gaussian proposals in logit space, rewards correctness minus squared confidence error, subtracts a per-question group baseline, and retains a supervised CE anchor. No sequence rollout or external reward model is used. These tutorial hyperparameters illustrate the interface; they have not been validated as an effective training recipe.
#
# The native released baseline and supervised stage first create a frozen CLM embedding cache and an SFT head. CPU continuation adapts that head only. Validation Brier selects the checkpoint, with the starting SFT head eligible; the heldout report retains accuracy, exact match, NLL, Brier and utility together. Falling training loss does not establish calibration gains.

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
RUN = TUTORIAL / "runs" / "clm-experimental-calibration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA = RUN / "data"
splits = write_dataset(DATA, seed=SEED)
LOGGER.info("Fresh split sizes: %s", {name: len(rows) for name, rows in splits.items()})


def launch(arguments: list[str]) -> None:
    """Show an explicit argv command and execute only when training is enabled."""
    LOGGER.info("%s", shlex.join(arguments))
    if RUN_TRAINING:
        subprocess.run(arguments, cwd=REPO, check=True)


# %% [markdown]
# ## Native released baseline and SFT initialization

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

# %%
launch(command + ["--mode=train", "--loss.mode=sft", f"--output_dir={RUN / 'sft'}"])

# %% [markdown]
# ## Optional experiment: propose logits and retain a CE anchor
#
# `RUN_TRAINING=False` leaves every model operation disabled. When enabled in a prepared environment, the encoder pass uses a GPU and continuation uses CPU tensors. Questions are grouped by option count without padding fake choices. Keep the split manifest fixed and report the selected result even when it does not improve.

# %%
if RUN_TRAINING:
    import copy

    import torch

    from nemo_automodel.components.datasets.decision.records import load_records, question_targets
    from nemo_automodel.components.eval.decision.metrics import DecisionEvaluator
    from nemo_automodel.components.loss.decision import rlcr_loss
    from nemo_automodel.components.models.clm.heads import EmbeddingTable, clm_batch, load_clm_heads, save_clm_heads

    caches = list(CACHE.glob("embeddings_*.pt"))
    if len(caches) != 1:
        raise ValueError("Expected exactly one cache from this fresh baseline run")
    table = EmbeddingTable.load(caches[0])
    records = {split: load_records(DATA / f"{split}.jsonl") for split in ("train", "validation", "test")}
    head = load_clm_heads(RUN / "sft/heads.pt", map_location="cpu")

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

    torch.manual_seed(SEED)
    proposal_generator = torch.Generator().manual_seed(SEED)
    best_brier = evaluate_records(records["validation"])["brier"]
    best_state = copy.deepcopy(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.00001)
    history = []
    for epoch in range(3):
        order = torch.randperm(len(records["train"]), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        epoch_losses = []
        head.train()
        for start in range(0, len(order), 16):
            batch = [records["train"][index] for index in order[start : start + 16]]
            states, options, counts, orders = clm_batch(batch, table, "cpu")
            logits = head.question_logits(states, options, counts)
            targets = [
                target.label for record, names in zip(batch, orders) for target in question_targets(record, names)
            ]
            grouped = {}
            for question_logits, target in zip(logits, targets):
                grouped.setdefault(question_logits.numel(), []).append((question_logits, target))
            optimizer.zero_grad(set_to_none=True)
            terms = []
            for group in grouped.values():
                output = rlcr_loss(
                    torch.stack([values for values, _ in group]),
                    torch.tensor([target for _, target in group]),
                    group_size=8,
                    exploration_std=0.3,
                    policy_weight=1.0,
                    supervised_weight=0.1,
                    generator=proposal_generator,
                )
                terms.append(output.loss * len(group) / len(logits))
            loss = torch.stack(terms).sum()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation = evaluate_records(records["validation"])
        history.append(
            {
                "epoch": epoch + 1,
                "objective": sum(epoch_losses) / len(epoch_losses),
                "validation_accuracy": validation["accuracy"],
                "validation_brier": validation["brier"],
            }
        )
        if validation["brier"] < best_brier:
            best_brier = validation["brier"]
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    selected = RUN / "experimental-rlcr"
    selected.mkdir()
    save_clm_heads(head, selected / "heads.pt")
    (selected / "metrics_test.json").write_text(json.dumps({"overall": evaluate_records(records["test"])}, indent=2))
    (selected / "history.json").write_text(json.dumps(history, indent=2))
    rows = summarize_run(RUN, stages=("base", "sft", "experimental-rlcr"))
    write_summary(rows, RUN / "heldout-summary.csv")
    LOGGER.info("Experimental calibration continuation: %s", rows)
