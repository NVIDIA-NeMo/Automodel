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

"""Coverage tests for the EAGLE-3 trailing grad-accum flush and related
recipe internals: ``_all_reduce_mean``, ``_optim_steps_per_epoch``,
``run_train_validation_loop`` with divisible / non-divisible batch counts,
gradient rescaling, ``_save_checkpoint``, and ``_run_eval``."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nemo_automodel.recipes.llm.train_eagle3 import (
    TrainEagle3Recipe,
    _all_reduce_mean,
    _optim_steps_per_epoch,
)

# ---------------------------------------------------------------------------
# Minimal stand-ins that satisfy the Eagle3 training loop interface
# ---------------------------------------------------------------------------


@dataclass
class _FakeBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    aux_hidden_states: torch.Tensor
    logits: torch.Tensor


class _FakeMetrics:
    def __init__(self):
        self.loss = torch.tensor(1.0, requires_grad=True)
        self.accuracy = torch.tensor(0.5)
        self.valid_tokens = torch.tensor(4)


class _FakeDraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.config = SimpleNamespace(save_pretrained=lambda path: None)


class _FakeTrainerModule(nn.Module):
    """Minimal trainer module whose forward returns fake metrics with a
    gradient-bearing loss so ``backward()`` and ``clip_grad_norm_`` work."""

    def __init__(self):
        super().__init__()
        self.draft_model = _FakeDraftModel()
        self.dummy = nn.Linear(4, 4)
        self.register_buffer("selected_token_ids", torch.arange(8))
        self.register_buffer("selected_token_mask", torch.ones(8, dtype=torch.bool))

    def forward(self, **kwargs):
        out = self.dummy(torch.randn(1, 4)).sum()
        return SimpleNamespace(
            loss=out.abs() + 1.0,
            accuracy=torch.tensor(0.5),
            valid_tokens=torch.tensor(4),
        )


class _FakeTargetWrapper:
    def generate_batch(self, input_ids, attention_mask, loss_mask):
        bs, sl = input_ids.shape
        return _FakeBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=torch.randn(bs, sl, 16),
            logits=torch.randn(bs, sl, 64),
        )


class _KeyedLoader:
    """Wraps a DataLoader to yield dicts with the keys the training loop expects."""

    def __init__(self, loader):
        self._loader = loader
        self.sampler = loader.sampler

    def __iter__(self):
        for ids, attn, lm in self._loader:
            yield {"input_ids": ids, "attention_mask": attn, "loss_mask": lm}

    def __len__(self):
        return len(self._loader)


# ---------------------------------------------------------------------------
# Recipe builder
# ---------------------------------------------------------------------------


def _build_recipe(tmp_path, num_samples=5, grad_accum=3, num_epochs=1, log_every=1):
    """Assemble a TrainEagle3Recipe with fake components -- no GPU, no HF model."""
    sl = 6
    train_data = TensorDataset(
        torch.randint(0, 64, (num_samples, sl)),
        torch.ones(num_samples, sl, dtype=torch.long),
        torch.ones(num_samples, sl, dtype=torch.long),
    )
    train_loader = _KeyedLoader(DataLoader(train_data, batch_size=1))

    trainer_module = _FakeTrainerModule()

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.device = torch.device("cpu")
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    recipe.trainer_module = trainer_module
    recipe.target_wrapper = _FakeTargetWrapper()
    recipe.train_dataloader = train_loader
    recipe.val_dataloader = None
    recipe.output_dir = tmp_path
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.grad_accumulation_steps = grad_accum
    recipe.max_grad_norm = 1.0
    recipe.num_epochs = num_epochs
    recipe.log_every_steps = log_every
    recipe.peak_lr = 1e-4
    recipe.total_optim_steps = num_epochs * _optim_steps_per_epoch(num_samples, grad_accum)
    recipe.warmup_steps = 1
    recipe.min_lr_ratio = 0.1

    recipe.optimizer = torch.optim.AdamW([p for p in trainer_module.parameters() if p.requires_grad], lr=1e-4)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)

    return recipe


# ---------------------------------------------------------------------------
# _all_reduce_mean (non-distributed)
# ---------------------------------------------------------------------------


def test_all_reduce_mean_passthrough():
    t = torch.tensor(3.0)
    assert _all_reduce_mean(t).item() == 3.0


# ---------------------------------------------------------------------------
# Training loop: divisible batch count (no trailing flush)
# ---------------------------------------------------------------------------


def test_no_trailing_flush_when_divisible(tmp_path):
    recipe = _build_recipe(tmp_path, num_samples=6, grad_accum=3)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 2


# ---------------------------------------------------------------------------
# Training loop: non-divisible batch count (trailing flush fires)
# ---------------------------------------------------------------------------


def test_trailing_flush_fires_when_non_divisible(tmp_path):
    recipe = _build_recipe(tmp_path, num_samples=5, grad_accum=3)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 2


def test_trailing_flush_single_micro_batch(tmp_path):
    recipe = _build_recipe(tmp_path, num_samples=4, grad_accum=3)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 2


# ---------------------------------------------------------------------------
# Training loop: entire epoch is one trailing flush
# ---------------------------------------------------------------------------


def test_entire_epoch_is_trailing_flush(tmp_path):
    recipe = _build_recipe(tmp_path, num_samples=2, grad_accum=4)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 1


# ---------------------------------------------------------------------------
# Training loop: grad_accum=1 (never triggers flush)
# ---------------------------------------------------------------------------


def test_grad_accum_one_no_flush(tmp_path):
    recipe = _build_recipe(tmp_path, num_samples=3, grad_accum=1)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 3


# ---------------------------------------------------------------------------
# Multi-epoch
# ---------------------------------------------------------------------------


def test_multi_epoch_with_trailing_flush(tmp_path):
    recipe = _build_recipe(tmp_path, num_samples=5, grad_accum=3, num_epochs=2)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 4


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


def test_checkpoint_save_is_noop_without_checkpointer(tmp_path):
    """When the recipe is built without a checkpointer (e.g. test harness skips
    ``setup()``), the end-of-epoch save call must be a true no-op rather than
    crashing on ``self.checkpointer.config.enabled``."""
    recipe = _build_recipe(tmp_path, num_samples=3, grad_accum=2)
    recipe.run_train_validation_loop()
    # No checkpointer was wired in, so nothing should be written anywhere under tmp_path.
    assert list(tmp_path.glob("epoch_*")) == []
    assert list(tmp_path.glob("checkpoints/*")) == []


# ---------------------------------------------------------------------------
# _run_eval
# ---------------------------------------------------------------------------


def test_run_eval_returns_none_without_val_loader(tmp_path):
    recipe = _build_recipe(tmp_path)
    assert recipe._run_eval() is None


# ---------------------------------------------------------------------------
# _module() unwrapping
# ---------------------------------------------------------------------------


def test_module_returns_unwrapped(tmp_path):
    recipe = _build_recipe(tmp_path)
    assert recipe._module() is recipe.trainer_module


# ---------------------------------------------------------------------------
# Logging path: log_every_steps triggers the info log
# ---------------------------------------------------------------------------


def test_logging_path_is_exercised(tmp_path):
    recipe = _build_recipe(tmp_path, num_samples=4, grad_accum=2, log_every=1)
    recipe.run_train_validation_loop()
    assert recipe.runtime.global_step == 2
