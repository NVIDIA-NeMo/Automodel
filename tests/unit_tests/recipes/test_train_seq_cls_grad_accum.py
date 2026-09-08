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

"""Gradients must not depend on how a global batch is split into micro-batches.

``torch.nn.CrossEntropyLoss`` reduces with ``mean``, and every ``backward()``
adds to ``.grad``. Accumulating N micro-batch means therefore yields N x the
mean over the full accumulated batch unless the count is divided back out, which
silently multiplies the effective learning rate by ``grad_accumulation_steps``.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.llm import train_seq_cls as seq_cls_mod
from nemo_automodel.recipes.llm.train_seq_cls import TrainFinetuneRecipeForSequenceClassification

VOCAB, N_CLASSES, HIDDEN = 16, 3, 8


class _TinyClassifier(nn.Module):
    """Smallest model with the sequence-classification forward contract."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_CLASSES, bias=False)

    def forward(self, input_ids, attention_mask=None):
        pooled = self.embed(input_ids).mean(dim=1)
        return SimpleNamespace(logits=self.head(pooled))


def _make_recipe(model):
    """A recipe instance with only what ``_run_train_optim_step`` touches."""
    recipe = object.__new__(TrainFinetuneRecipeForSequenceClassification)
    recipe.model_parts = [model]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), world_size=1)
    recipe.loss_fn = nn.CrossEntropyLoss()
    recipe.optimizer = [torch.optim.SGD(model.parameters(), lr=0.0)]
    recipe.lr_scheduler = None
    recipe.max_grad_norm = 1.0
    recipe.device_mesh = None
    recipe.timestamp = 0.0
    recipe.mfu_calculator = None
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
    recipe._get_dp_group_size = lambda include_cp=False: 1
    recipe._get_cp_group_size = lambda: 1
    recipe._get_pp_rank = lambda: 0
    recipe._dp_allreduce = lambda t, *a, **k: t
    return recipe


def _grads_for(split, seed=0):
    """Run one optimizer step over ``split`` micro-batches; return the accumulated grads.

    Gradients are captured at ``clip_grad_norm``, which runs after the
    accumulation loop and before ``optimizer.step()`` zeroes them.
    """
    torch.manual_seed(seed)
    model = _TinyClassifier()
    recipe = _make_recipe(model)

    torch.manual_seed(123)
    input_ids = torch.randint(0, VOCAB, (8, 4))
    labels = torch.randint(0, N_CLASSES, (8,))

    batches = [
        {
            "input_ids": input_ids[i : i + 8 // split],
            "attention_mask": torch.ones_like(input_ids[i : i + 8 // split]),
            "labels": labels[i : i + 8 // split],
        }
        for i in range(0, 8, 8 // split)
    ]
    assert len(batches) == split

    captured = {}

    def _capture(**kwargs):
        captured["grads"] = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        return torch.tensor(0.0)

    with (
        mock.patch.object(seq_cls_mod, "clip_grad_norm", _capture),
        mock.patch.object(torch.cuda, "max_memory_allocated", lambda: 0),
    ):
        recipe._run_train_optim_step(batches)

    assert captured["grads"], "no gradients were captured"
    return captured["grads"]


@pytest.mark.parametrize("split", [2, 4, 8])
def test_gradients_are_invariant_to_microbatch_split(split):
    """Splitting one global batch into more micro-batches must not change grads."""
    single = _grads_for(1)
    accumulated = _grads_for(split)

    assert len(single) == len(accumulated)
    for g_one, g_many in zip(single, accumulated):
        ratio = (g_many.norm() / g_one.norm()).item()
        assert torch.allclose(g_many, g_one, rtol=1e-5, atol=1e-6), (
            f"{split} micro-batches scaled the gradient by ~{ratio:.2f}x"
        )
