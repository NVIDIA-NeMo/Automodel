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

"""Gradients and reported loss must both equal the global-batch reference.

``CrossEntropyLoss`` reduces with ``mean``, so a per-micro-batch loss is a local
average. Averages cannot be added back together: summing them across
accumulation steps and across DP ranks over-counts by ``grad_acc`` and by
``dp_size`` respectively, and misweights any unevenly sized micro-batch.

Both the gradient and the reported metric are therefore normalized by one
DP-allreduced global label count, the convention ``train_ft.py`` uses with
``num_label_tokens``. ``* dp_size`` is kept: it cancels the DDP/FSDP gradient
average, which only reconstructs the global mean once the denominator is global.

The tests below drive the real ``_run_train_optim_step``, simulate the DP
average, and compare against ``F.cross_entropy(..., reduction="mean")`` over the
whole global batch. Uneven splits such as ``3 + 1`` are covered explicitly:
equal splits are the one case where a mean of means happens to be correct.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.recipes.llm import train_seq_cls as seq_cls_mod
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
    """A recipe instance with only what ``_run_train_optim_step`` touches.

    ``allreduce`` stands in for ``_dp_allreduce``, a SUM across DP ranks. Ranks
    hold different data, so it cannot be faked as ``value * dp_size``; the
    caller supplies a real cross-rank sum.
    """
    recipe = object.__new__(TrainFinetuneRecipeForSequenceClassification)
    recipe.model_parts = [model]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), world_size=dp_size)
    recipe.loss_fn = nn.CrossEntropyLoss()
    recipe.optimizer = [torch.optim.SGD(model.parameters(), lr=0.0)]
    recipe.lr_scheduler = None
    recipe.max_grad_norm = 1.0
    recipe.device_mesh = None
    recipe.timestamp = 0.0
    recipe.mfu_calculator = None
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
    recipe._get_dp_group_size = lambda include_cp=False: dp_size
    recipe._get_cp_group_size = lambda: 1
    recipe._get_pp_rank = lambda: 0
    recipe._dp_allreduce = allreduce
    return recipe


def _global_batch(splits, dp_size, seed=123):
    """Build one global batch and the per-rank micro-batch shards over it."""
    total = sum(splits) * dp_size
    torch.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (total, SEQ))
    labels = torch.randint(0, N_CLASSES, (total,))

    shards, cursor = [], 0
    for _ in range(dp_size):
        rank_batches = []
        for size in splits:
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


def _run(splits, dp_size, seed=0):
    """Return (DDP-averaged grads, reported loss, reference grads, reference loss).

    ``_dp_allreduce`` is simulated in two phases. Every value the recipe reduces
    (label count, token count, summed loss, accuracy) is computed from the
    forward pass alone and never from a previous reduction, so phase one can
    record each rank's inputs with an identity stub and phase two can replay the
    true cross-rank sum in the same call order.

    Gradients are captured at ``clip_grad_norm``, which runs after the
    accumulation loop and before ``optimizer.step()`` zeroes them.
    """
    torch.manual_seed(seed)
    init_state = {k: v.clone() for k, v in _TinyClassifier().state_dict().items()}
    input_ids, labels, shards = _global_batch(splits, dp_size)

    def drive(rank_batches, allreduce):
        model = _TinyClassifier()
        model.load_state_dict(init_state)
        recipe = _make_recipe(model, dp_size, allreduce)
        captured = {}

        def _capture(**kwargs):
            captured["grads"] = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
            return torch.tensor(0.0)

        with (
            mock.patch.object(seq_cls_mod, "clip_grad_norm", _capture),
            mock.patch.object(torch.cuda, "max_memory_allocated", lambda: 0),
        ):
            sample = recipe._run_train_optim_step(rank_batches)
        return captured["grads"], sample

    # Phase 1: record what each rank feeds into every reduction, in order.
    recorded = []
    for rank_batches in shards:
        calls = []
        drive(rank_batches, lambda t, *a, **k: (calls.append(t.detach().clone()), t)[1])
        recorded.append(calls)

    totals = [torch.stack(vals).sum(0) for vals in zip(*recorded)]

    # Phase 2: replay the real cross-rank sum.
    per_rank_grads, reported = [], []
    for rank_batches in shards:
        idx = {"i": 0}

        def allreduce(t, *a, **k):
            out = totals[idx["i"]]
            idx["i"] += 1
            return out.clone()

        grads, sample = drive(rank_batches, allreduce)
        per_rank_grads.append(grads)
        reported.append(float(sample.metrics["loss"]))

    averaged = [torch.stack(g).mean(0) for g in zip(*per_rank_grads)]

    # Reference: the mean loss over the entire global batch.
    ref_model = _TinyClassifier()
    ref_model.load_state_dict(init_state)
    ref_loss = F.cross_entropy(ref_model(input_ids).logits, labels, reduction="mean")
    ref_loss.backward()
    ref_grads = [p.grad.detach().clone() for p in ref_model.parameters() if p.grad is not None]

    # Every rank logs the same global number, so any of them represents the step.
    return averaged, reported[0], ref_grads, float(ref_loss)


# splits per rank, dp_size -- includes the uneven 3 + 1 case at several dp sizes
CASES = [
    ([2], 1),
    ([2, 2], 1),
    ([1, 1, 1, 1], 1),
    ([3, 1], 1),
    ([2], 2),
    ([2, 2], 2),
    ([3, 1], 2),
    ([2, 2], 4),
    ([3, 1], 4),
]
IDS = [f"splits={s}-dp={d}" for s, d in CASES]


@pytest.mark.parametrize("splits,dp_size", CASES, ids=IDS)
def test_gradients_match_global_batch_reference(splits, dp_size):
    """Neither the micro-batch split nor dp_size may change the gradient."""
    got, _, ref, _ = _run(splits, dp_size)

    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        ratio = (g.norm() / r.norm()).item()
        assert torch.allclose(g, r, rtol=1e-5, atol=1e-6), (
            f"splits={splits}, dp={dp_size}: gradient scaled by ~{ratio:.2f}x"
        )


@pytest.mark.parametrize("splits,dp_size", CASES, ids=IDS)
def test_reported_loss_matches_global_batch_reference(splits, dp_size):
    """The logged loss is the global mean, not a mean of per-micro-batch means.

    A mean of means is inflated by dp_size and misweights uneven splits, so the
    metric would disagree with the objective the gradient actually optimizes.
    """
    _, got, _, ref = _run(splits, dp_size)

    assert got == pytest.approx(ref, rel=1e-5), (
        f"splits={splits}, dp={dp_size}: reported {got:.4f} vs global {ref:.4f} (~{got / ref:.2f}x)"
    )


def test_uneven_split_differs_from_mean_of_means():
    """Guards the guard: 3 + 1 must be a case where the two conventions disagree.

    If they happened to coincide, the uneven parametrizations above would pass
    for the wrong reason.
    """
    input_ids, labels, shards = _global_batch([3, 1], dp_size=1)
    model = _TinyClassifier()

    with torch.no_grad():
        means = [F.cross_entropy(model(b["input_ids"]).logits, b["labels"]) for b in shards[0]]
        mean_of_means = float(torch.stack(means).mean())
        global_mean = float(F.cross_entropy(model(input_ids).logits, labels, reduction="mean"))

    assert mean_of_means != pytest.approx(global_mean, rel=1e-3)
