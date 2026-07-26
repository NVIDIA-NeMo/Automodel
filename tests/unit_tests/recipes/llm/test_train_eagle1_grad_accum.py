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

"""Unit tests for EAGLE-1 / EAGLE-2 recipe gradient-accumulation arithmetic.

Mirrors the EAGLE-3 grad-accum coverage. Two invariants are guarded:

1. The LR scheduler is sized against the *ceil* of ``num_batches / accum`` so
   the trailing partial accumulation window the trainer flushes each epoch is
   counted as a real optimizer step (floor would saturate ``progress`` and the
   trailing flushes would train at ``min_lr_ratio``).
2. The trailing flush rescales the accumulated gradient by
   ``grad_accumulation_steps / pending_micro_batches`` so a non-divisible epoch's
   final step lands on the same gradient scale as every full window -- not the
   ``pending / accum`` fraction the per-micro-batch ``loss / accum`` division
   leaves behind.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe

# ---------------------------------------------------------------------------
# Trailing-flush gradient rescale
# ---------------------------------------------------------------------------


class _ConstantGradModule(nn.Module):
    """Trainer-module stand-in whose loss yields a constant unit gradient.

    ``loss = w.sum()`` so ``d loss / d w == ones`` for every micro-batch,
    independent of the (ignored) batch inputs. After the recipe divides the
    loss by ``grad_accumulation_steps`` and accumulates, a full window of
    ``accum`` micro-batches leaves ``w.grad == ones``; a trailing window of
    ``r`` micro-batches leaves ``r / accum * ones`` *before* the flush rescale
    and ``ones`` *after* it.
    """

    def __init__(self, n: int = 4):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))

    def forward(self, **kwargs):
        loss = self.w.sum()
        # Mirror EagleStepMetrics: the recipe loop reads hidden_loss/token_loss
        # for per-step logging in addition to the combined loss.
        return SimpleNamespace(
            loss=loss,
            hidden_loss=loss.detach(),
            token_loss=torch.zeros((), device=self.w.device),
            accuracy=torch.tensor(0.5),
        )


class _FakeTargetWrapper:
    def generate_batch(self, input_ids, attention_mask, loss_mask, **packing_kwargs):
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            input_hidden_states=None,
            target_hidden_states=None,
            target_logits=None,
            position_ids=None,
            seq_lens=None,
            doc_remaining=None,
        )


class _ListLoader:
    """Yields ``num_batches`` keyed dicts; no ``sampler`` attr (set_epoch skipped)."""

    def __init__(self, num_batches: int):
        one = torch.ones(1, 4, dtype=torch.long)
        self._batches = [
            {"input_ids": one.clone(), "attention_mask": one.clone(), "loss_mask": one.clone()}
            for _ in range(num_batches)
        ]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _build_recipe(num_batches: int, grad_accum: int, trainer_module: nn.Module | None = None) -> TrainEagle1Recipe:
    recipe = TrainEagle1Recipe.__new__(TrainEagle1Recipe)
    recipe.device = torch.device("cpu")
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    # Assign trainer_module exactly once: BaseRecipe.__setattr__ tracks model
    # attributes and rejects re-assignment, so a DDP-wrapped module must be
    # passed in here rather than swapped in after construction.
    recipe.trainer_module = trainer_module if trainer_module is not None else _ConstantGradModule()
    recipe.target_wrapper = _FakeTargetWrapper()
    recipe.train_dataloader = _ListLoader(num_batches)
    recipe.val_dataloader = None
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.grad_accumulation_steps = grad_accum
    # Large clip threshold so clip_grad_norm_ never rescales the captured grad.
    recipe.max_grad_norm = 1e9
    recipe.num_epochs = 1
    recipe.log_every_steps = 1
    recipe.total_optim_steps = -(-num_batches // grad_accum)
    recipe.optimizer = torch.optim.SGD(recipe.trainer_module.parameters(), lr=0.0)
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)
    return recipe


@pytest.mark.parametrize(
    "num_batches,accum,expected_steps",
    [
        (4, 3, 2),  # one full window (3) + trailing window of 1
        (5, 3, 2),  # one full window (3) + trailing window of 2
        (2, 4, 1),  # entire epoch is a single trailing window of 2
    ],
)
def test_trailing_flush_rescales_gradient_to_full_window_scale(num_batches, accum, expected_steps):
    """Every optimizer step -- full window and trailing flush alike -- sees a
    unit gradient. Without the trailing rescale the final step would see
    ``pending / accum * ones`` and this assertion would fail."""
    recipe = _build_recipe(num_batches, accum)
    module = recipe.trainer_module

    captured: list[torch.Tensor] = []

    def _pre_step_hook(optimizer, args, kwargs):
        # Fires right before each optimizer step, after the trailing rescale and
        # the (no-op, max_grad_norm=1e9) clip -- so it reflects exactly the
        # gradient the step consumes. A pre-hook avoids overriding ``step`` and
        # the spurious LR-scheduler ordering warning that would cause.
        captured.append(module.w.grad.detach().clone())

    recipe.optimizer.register_step_pre_hook(_pre_step_hook)
    recipe.run_train_validation_loop()

    assert recipe.runtime.global_step == expected_steps
    assert len(captured) == expected_steps
    for grad in captured:
        torch.testing.assert_close(grad, torch.ones_like(grad))


# ---------------------------------------------------------------------------
# Real (single-process, gloo) DDP: the loop enters no_sync() on interior steps
# ---------------------------------------------------------------------------


def test_no_sync_entered_for_interior_microbatches_under_ddp(tmp_path):
    """Wrap the trainer module in a real DDP (1 rank, gloo) and confirm the loop
    enters ``no_sync()`` exactly on the micro-batches that defer their
    all-reduce: interior steps of a window, but neither the window closer nor
    the epoch's final batch. 4 batches with accum=3 => batch 0,1 deferred,
    batch 2 closes the window (sync), batch 3 is the last batch (sync) => 2
    no_sync entries."""
    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this session")

    dist.init_process_group(backend="gloo", init_method=f"file://{tmp_path / 'pg'}", world_size=1, rank=0)
    try:
        ddp = DistributedDataParallel(_ConstantGradModule())  # CPU => no device_ids
        recipe = _build_recipe(num_batches=4, grad_accum=3, trainer_module=ddp)

        no_sync_entries = 0
        real_no_sync = ddp.no_sync

        def _spy_no_sync():
            nonlocal no_sync_entries
            no_sync_entries += 1
            return real_no_sync()

        ddp.no_sync = _spy_no_sync
        recipe.run_train_validation_loop()

        assert no_sync_entries == 2
        assert recipe.runtime.global_step == 2
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Progress bar: one update per optimizer step (incl. trailing flush), closed
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
    monkeypatch.setattr(TrainEagle1Recipe, "_make_progress_bar", lambda self, **kwargs: fake)
    # 4 batches / accum 3 -> one full window + one trailing flush = 2 steps.
    recipe = _build_recipe(num_batches=4, grad_accum=3)
    recipe.run_train_validation_loop()
    assert fake.n == recipe.runtime.global_step == 2
    assert fake.closed
    assert set(fake.postfix) == {"loss", "acc", "lr"}


class _RaisingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4))

    def forward(self, **kwargs):
        raise RuntimeError("boom")


def test_progress_bar_closed_when_loop_raises(monkeypatch):
    """The bar is closed even when the training loop raises (try/finally)."""
    fake = _FakeProgressBar()
    monkeypatch.setattr(TrainEagle1Recipe, "_make_progress_bar", lambda self, **kwargs: fake)
    recipe = _build_recipe(num_batches=2, grad_accum=1, trainer_module=_RaisingModule())
    with pytest.raises(RuntimeError, match="boom"):
        recipe.run_train_validation_loop()
    assert fake.closed
