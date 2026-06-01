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
import torch.nn as nn

from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe, _optim_steps_per_epoch


@pytest.mark.parametrize(
    "num_batches,accum,expected",
    [
        (10, 1, 10),
        (10, 2, 5),
        (10, 3, 4),  # 3 full windows + 1 trailing micro-batch -> 4 steps
        (10, 4, 3),  # 2 full windows + 2 trailing -> 3 steps
        (1, 4, 1),  # entire epoch is one trailing flush
        (4, 4, 1),
        (5, 4, 2),
        (0, 4, 0),  # iterable dataloader / no length
    ],
)
def test_optim_steps_per_epoch_uses_ceil_division(num_batches, accum, expected):
    assert _optim_steps_per_epoch(num_batches, accum) == expected


def test_optim_steps_per_epoch_handles_invalid_inputs():
    assert _optim_steps_per_epoch(0, 1) == 0
    assert _optim_steps_per_epoch(-1, 4) == 0
    assert _optim_steps_per_epoch(10, 0) == 0
    assert _optim_steps_per_epoch(10, -1) == 0


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
        return SimpleNamespace(loss=self.w.sum(), accuracy=torch.tensor(0.5))


class _FakeTargetWrapper:
    def generate_batch(self, input_ids, attention_mask, loss_mask):
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            input_hidden_states=None,
            target_hidden_states=None,
            target_logits=None,
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


def _build_recipe(num_batches: int, grad_accum: int) -> TrainEagle1Recipe:
    recipe = TrainEagle1Recipe.__new__(TrainEagle1Recipe)
    recipe.device = torch.device("cpu")
    recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
    recipe.trainer_module = _ConstantGradModule()
    recipe.target_wrapper = _FakeTargetWrapper()
    recipe.train_dataloader = _ListLoader(num_batches)
    recipe.val_dataloader = None
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.grad_accumulation_steps = grad_accum
    # Large clip threshold so clip_grad_norm_ never rescales the captured grad.
    recipe.max_grad_norm = 1e9
    recipe.num_epochs = 1
    recipe.log_every_steps = 1
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
