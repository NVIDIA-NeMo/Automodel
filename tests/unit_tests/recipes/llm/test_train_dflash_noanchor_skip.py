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

"""DFlash training loop skips a no-valid-anchor micro-batch (single-process path).

A micro-batch whose samples are all too short raises ``NoValidAnchorsError`` from
the trainer forward. Single-process (no DDP) the loop must skip it -- count it,
run no optimizer step for it -- and keep training the rest. (The multi-rank
lockstep behavior, which avoids a one-rank-only backward hang, is exercised by
the ``_all_ranks_have_valid`` unit tests and validated on a multi-GPU server.)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.speculative.dflash.core import NoValidAnchorsError
from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe


class _FakeTrainerModule(nn.Module):
    """Forward returns a grad-bearing loss, except on the call indices in
    ``raise_on`` where it raises NoValidAnchorsError (all samples too short)."""

    def __init__(self, raise_on):
        super().__init__()
        self.dummy = nn.Linear(4, 4)
        self._raise_on = set(raise_on)
        self._calls = 0

    def forward(self, **kwargs):
        i = self._calls
        self._calls += 1
        if i in self._raise_on:
            raise NoValidAnchorsError("every sample too short")
        out = self.dummy(torch.randn(1, 4)).sum()
        return SimpleNamespace(loss=out.abs() + 1.0, accuracy=torch.tensor(0.5))


class _FakeTargetWrapper:
    def generate_batch(self, input_ids, attention_mask, loss_mask):
        bs, sl = input_ids.shape
        return SimpleNamespace(input_ids=input_ids, hidden_states=torch.randn(bs, sl, 16), loss_mask=loss_mask)


def _batch():
    return {
        "input_ids": torch.randint(0, 64, (1, 6)),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "loss_mask": torch.ones(1, 6, dtype=torch.long),
    }


def _build_recipe(raise_on, num_batches=3, grad_accum=1):
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    trainer_module = _FakeTrainerModule(raise_on)
    recipe.device = torch.device("cpu")
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    recipe.trainer_module = trainer_module
    recipe.target_wrapper = _FakeTargetWrapper()
    recipe.train_dataloader = [_batch() for _ in range(num_batches)]
    recipe.val_dataloader = None
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.grad_accumulation_steps = grad_accum
    recipe.max_grad_norm = 1.0
    recipe.num_epochs = 1
    recipe.log_every_steps = 1
    recipe.total_optim_steps = -(-num_batches // grad_accum)
    recipe._skipped_micro_batches = 0
    recipe.save_checkpoint_every_epoch = False
    recipe.ckpt_every_steps = None
    recipe.optimizer = torch.optim.AdamW([p for p in trainer_module.parameters() if p.requires_grad], lr=1e-4)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)
    recipe._maybe_save_step_checkpoint = lambda epoch_idx: None
    recipe._maybe_save_final_checkpoint = lambda completed_epochs: False
    return recipe


def test_noanchor_micro_batch_is_skipped_single_process():
    # 3 micro-batches, the middle one (call index 1) has no valid anchors.
    recipe = _build_recipe(raise_on={1}, num_batches=3, grad_accum=1)
    recipe.run_train_validation_loop()
    # The bad micro-batch was skipped, the other two each closed a window.
    assert recipe._skipped_micro_batches == 1
    assert recipe.runtime.global_step == 2


def test_no_skip_when_all_micro_batches_valid():
    recipe = _build_recipe(raise_on=set(), num_batches=3, grad_accum=1)
    recipe.run_train_validation_loop()
    assert recipe._skipped_micro_batches == 0
    assert recipe.runtime.global_step == 3


# ---------------------------------------------------------------------------
# Progress bar: one update per optimizer step, postfix at log points, closed
# ---------------------------------------------------------------------------


class _FakeProgressBar:
    def __init__(self):
        self.n = 0
        self.postfix = None
        self.closed = False

    def update(self, count=1):
        self.n += count

    def set_postfix(self, **kwargs):
        self.postfix = kwargs

    def close(self):
        self.closed = True


def test_progress_bar_advances_per_optim_step(monkeypatch):
    fake = _FakeProgressBar()
    monkeypatch.setattr(TrainDFlashRecipe, "_make_progress_bar", lambda self, **kwargs: fake)
    recipe = _build_recipe(raise_on=set(), num_batches=3, grad_accum=1)
    recipe.run_train_validation_loop()
    assert fake.n == recipe.runtime.global_step == 3
    assert fake.closed
    assert set(fake.postfix) == {"loss", "acc", "lr"}


def test_progress_bar_closed_when_loop_raises(monkeypatch):
    """The bar is closed even when the training loop raises (try/finally)."""
    fake = _FakeProgressBar()
    monkeypatch.setattr(TrainDFlashRecipe, "_make_progress_bar", lambda self, **kwargs: fake)

    def _boom(self, **kwargs):
        raise RuntimeError("boom")

    recipe = _build_recipe(raise_on=set(), num_batches=2, grad_accum=1)
    monkeypatch.setattr(type(recipe.trainer_module), "forward", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        recipe.run_train_validation_loop()
    assert fake.closed
