# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""The validation loss must be the mean over every validation sample.

Each DP rank validates a disjoint shard, and the last batch of a shard is
usually smaller than the rest. The reported ``val_loss`` (which also selects the
best checkpoint) must therefore weight each batch by its size and be reduced
across DP ranks, exactly like ``val_accuracy`` next to it. The tests drive the
real ``_validate_one_epoch`` and compare against ``F.cross_entropy`` over the
whole validation set.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.recipes.llm.train_seq_cls import TrainFinetuneRecipeForSequenceClassification

VOCAB, N_CLASSES, HIDDEN, SEQ = 16, 3, 8, 4


class _TinyClassifier(nn.Module):
    """Smallest model with the sequence-classification forward contract."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_CLASSES, bias=False)

    def forward(self, input_ids, attention_mask=None):
        pooled = self.embed(input_ids).mean(dim=1)
        return SimpleNamespace(logits=self.head(pooled))


def _make_recipe(model, dp_size, allreduce):
    """A recipe instance with only what ``_validate_one_epoch`` touches."""
    recipe = object.__new__(TrainFinetuneRecipeForSequenceClassification)
    recipe.model_parts = [model]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), world_size=dp_size)
    recipe.loss_fn = nn.CrossEntropyLoss()
    recipe.optimizer = [torch.optim.SGD(model.parameters(), lr=0.0)]
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
    recipe._get_dp_group_size = lambda include_cp=False: dp_size
    recipe._dp_allreduce = allreduce
    return recipe


def _val_shards(batch_sizes, dp_size, seed=123):
    """Build one validation set and each DP rank's list of batches over it."""
    total = sum(batch_sizes) * dp_size
    torch.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (total, SEQ))
    labels = torch.randint(0, N_CLASSES, (total,))

    shards, cursor = [], 0
    for _ in range(dp_size):
        rank_batches = []
        for size in batch_sizes:
            sl = slice(cursor, cursor + size)
            rank_batches.append(
                {
                    "input_ids": input_ids[sl],
                    "attention_mask": torch.ones_like(input_ids[sl]),
                    "labels": labels[sl],
                }
            )
            cursor += size
        shards.append(rank_batches)
    return input_ids, labels, shards


def _run(batch_sizes, dp_size, seed=0):
    """Return (val_loss reported by each rank, reference loss over the whole set).

    ``_dp_allreduce`` is simulated in two phases: phase one records what each
    rank feeds into every reduction, phase two replays the true cross-rank sum
    in the same call order.
    """
    torch.manual_seed(seed)
    model = _TinyClassifier()
    input_ids, labels, shards = _val_shards(batch_sizes, dp_size)

    def drive(rank_batches, allreduce):
        recipe = _make_recipe(model, dp_size, allreduce)
        with mock.patch.object(torch.cuda, "max_memory_allocated", lambda: 0):
            return recipe._validate_one_epoch(rank_batches)

    recorded = []
    for rank_batches in shards:
        calls = []
        drive(rank_batches, lambda t, *a, **k: (calls.append(t.detach().clone()), t)[1])
        recorded.append(calls)
    totals = [torch.stack(vals).sum(0) for vals in zip(*recorded)]

    reported = []
    for rank_batches in shards:
        idx = {"i": 0}

        def allreduce(t, *a, **k):
            out = totals[idx["i"]]
            idx["i"] += 1
            return out.clone()

        sample = drive(rank_batches, allreduce)
        reported.append(float(sample.metrics["val_loss"]))

    with torch.no_grad():
        ref = float(F.cross_entropy(model(input_ids).logits, labels, reduction="mean"))
    return reported, ref


# batch sizes per rank, dp_size -- the trailing 1 is the short final batch
CASES = [
    ([3, 1], 1),
    ([4, 4, 2], 1),
    ([2], 2),
    ([3, 1], 2),
    ([3, 1], 4),
]
IDS = [f"batches={b}-dp={d}" for b, d in CASES]


@pytest.mark.parametrize("batch_sizes,dp_size", CASES, ids=IDS)
def test_val_loss_matches_whole_validation_set(batch_sizes, dp_size):
    """Every rank reports the mean over all validation samples on all ranks."""
    reported, ref = _run(batch_sizes, dp_size)

    for rank, got in enumerate(reported):
        assert got == pytest.approx(ref, rel=1e-5), (
            f"batches={batch_sizes}, dp={dp_size}, rank={rank}: val_loss {got:.4f} vs whole-set mean {ref:.4f}"
        )


def test_short_final_batch_differs_from_mean_of_means():
    """Guards the guard: 3 + 1 must be a case where a mean of batch means is wrong.

    If the two conventions coincided, the uneven cases above would pass for the
    wrong reason.
    """
    torch.manual_seed(0)
    model = _TinyClassifier()
    input_ids, labels, shards = _val_shards([3, 1], dp_size=1)

    with torch.no_grad():
        means = [F.cross_entropy(model(b["input_ids"]).logits, b["labels"]) for b in shards[0]]
        mean_of_means = float(torch.stack(means).mean())
        whole = float(F.cross_entropy(model(input_ids).logits, labels, reduction="mean"))

    assert mean_of_means != pytest.approx(whole, rel=1e-3)
