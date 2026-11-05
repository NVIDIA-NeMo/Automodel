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

"""The shared draft training loop in ``SpecDecodeRecipeBase``, on a minimal recipe.

The per-algorithm recipe tests exercise the loop through their own hooks; these
pin the loop's own contract: trailing-window gradient rescale, the epoch report
and checkpoint, the single-process sync context, and best-effort teardown.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.speculative_decoding._spec_recipe_base import SpecDecodeRecipeBase


class _ToyRecipe(SpecDecodeRecipeBase):
    """One scalar draft weight ``w`` trained on ``loss = w * x`` for each batch ``x``."""

    def __init__(self, batches: list[float], *, grad_accumulation_steps: int, num_epochs: int = 1):
        self.trainer_module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.trainer_module.weight.fill_(0.0)
        self.optimizer = torch.optim.SGD(self.trainer_module.parameters(), lr=1.0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.train_dataloader = batches
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = 1e9
        self.num_epochs = num_epochs
        self.log_every_steps = 1
        self.total_optim_steps = 10_000
        self.runtime = SimpleNamespace(global_step=0)
        self.dist_env = SimpleNamespace(is_main=True)
        self.save_checkpoint_every_epoch = False
        self.step_grads: list[float] = []
        self.logged_windows: list[list[float]] = []
        self.reports: list[tuple[int, int]] = []
        self.optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: self.step_grads.append(self.trainer_module.weight.grad.item())
        )

    def _make_progress_bar(self, total=None, initial=0):
        return None

    def _forward_micro_batch(self, batch: float):
        """Return metrics whose scalar ``loss`` is ``w * batch``."""
        return SimpleNamespace(loss=self.trainer_module.weight.sum() * batch)

    def _new_train_window(self) -> list[float]:
        return []

    def _accumulate_train_metrics(self, window: list[float], metrics) -> None:
        window.append(metrics.loss.item())

    def _log_train_window(self, window: list[float], epoch_idx: int, grad_norm, pbar) -> None:
        self.logged_windows.append(list(window))

    def _run_eval(self):
        return None

    def _report_epoch(self, epoch_idx: int, completed_steps: int, eval_metrics):
        self.reports.append((epoch_idx, completed_steps))
        return eval_metrics


def test_trailing_window_gradient_matches_a_full_window():
    # 5 batches, accumulation 2: two full windows and a trailing window of one.
    recipe = _ToyRecipe([1.0, 3.0, 2.0, 2.0, 4.0], grad_accumulation_steps=2)
    recipe.run_train_validation_loop()

    # d(w * x / 2)/dw summed over the window: full windows average their two
    # batches; the trailing window's single batch is rescaled by 2 / 1.
    assert recipe.step_grads == pytest.approx([2.0, 2.0, 4.0])
    assert recipe.runtime.global_step == 3
    assert recipe.reports == [(0, 3)]
    # A fresh window per log point; the trailing window is not logged by default.
    # w starts at 0 and is -2 after the first step.
    assert recipe.logged_windows == [[0.0, 0.0], [-4.0, -4.0]]


def test_epoch_checkpoint_saved_even_for_an_empty_epoch():
    recipe = _ToyRecipe([], grad_accumulation_steps=2)
    recipe.save_checkpoint_every_epoch = True
    recipe.save_checkpoint = mock.MagicMock()

    recipe.run_train_validation_loop()

    recipe.save_checkpoint.assert_called_once()
    kwargs = recipe.save_checkpoint.call_args.kwargs
    assert kwargs["epoch"] == 1
    assert kwargs["train_loss"] is None
    assert kwargs["is_final_checkpoint"] is True


def test_single_process_draft_never_enters_no_sync():
    recipe = _ToyRecipe([1.0], grad_accumulation_steps=4)

    ctx = recipe._grad_sync_context(pending_micro_batches=0, batch_idx=0, num_batches=10)

    assert isinstance(ctx, nullcontext)


def test_stop_at_step_budget_ends_the_run_at_a_window_boundary():
    recipe = _ToyRecipe([1.0] * 8, grad_accumulation_steps=2, num_epochs=3)
    recipe.stop_at_step_budget = True
    recipe.total_optim_steps = 3

    recipe.run_train_validation_loop()

    assert recipe.runtime.global_step == 3
    assert recipe.reports == [(0, 3)]


def test_default_teardown_finishes_loggers_even_when_one_fails():
    recipe = _ToyRecipe([1.0], grad_accumulation_steps=1)
    recipe.wandb_run = mock.MagicMock()
    recipe.wandb_run.finish.side_effect = RuntimeError("wandb down")
    recipe.trackio_logger = mock.MagicMock()

    recipe.run_train_validation_loop()

    recipe.wandb_run.finish.assert_called_once()
    recipe.trackio_logger.finish.assert_called_once()


class _SampledLoader:
    """A loader whose sampler records the epochs it was set to."""

    def __init__(self, batches: list[float]):
        self.batches = batches
        self.sampler = SimpleNamespace(epochs=[])
        self.sampler.set_epoch = self.sampler.epochs.append

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def test_sampler_epoch_is_set_every_epoch():
    recipe = _ToyRecipe([], grad_accumulation_steps=1, num_epochs=2)
    recipe.train_dataloader = _SampledLoader([1.0, 1.0])

    recipe.run_train_validation_loop()

    assert recipe.train_dataloader.sampler.epochs == [0, 1]
    assert recipe.runtime.global_step == 4


def test_loader_without_length_flushes_its_trailing_window():
    recipe = _ToyRecipe([], grad_accumulation_steps=2)
    recipe.train_dataloader = (x for x in [1.0, 3.0, 4.0])

    recipe.run_train_validation_loop()

    assert recipe.step_grads == pytest.approx([2.0, 4.0])


def test_base_report_returns_eval_metrics_and_abstract_hooks_raise():
    recipe = _ToyRecipe([], grad_accumulation_steps=1)

    assert SpecDecodeRecipeBase._report_epoch(recipe, 0, 1, {"val_loss": 1.0}) == {"val_loss": 1.0}
    for hook, args in (
        (SpecDecodeRecipeBase._forward_micro_batch, (None,)),
        (SpecDecodeRecipeBase._new_train_window, ()),
        (SpecDecodeRecipeBase._log_train_window, ([], 0, None, None)),
    ):
        with pytest.raises(NotImplementedError):
            hook(recipe, *args)


def test_teardown_runs_when_training_raises():
    recipe = _ToyRecipe([1.0], grad_accumulation_steps=1)
    recipe.trackio_logger = mock.MagicMock()
    recipe._forward_micro_batch = mock.MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        recipe.run_train_validation_loop()

    recipe.trackio_logger.finish.assert_called_once()
