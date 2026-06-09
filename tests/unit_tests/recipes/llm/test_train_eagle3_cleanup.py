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

"""Teardown guarantees for the EAGLE-3 training loop.

``run_train_validation_loop`` must release training resources (disconnect the
remote target, finish the W&B run, destroy the process group) on *any* exit
path, not just normal completion. A mid-training exception that skipped the
remote-target ``close()`` would otherwise leave the long-lived target server
with a stale client-idle state and a half-open NCCL transport.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe


def _make_recipe(**overrides) -> TrainEagle3Recipe:
    """A minimal recipe with just the attributes the train loop reads (no setup)."""
    r = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    r.trainer_module = mock.MagicMock()
    dataloader = mock.MagicMock()
    dataloader.__len__.return_value = 4
    r.train_dataloader = dataloader
    r.num_epochs = 1
    r._resume_epoch = 0
    r.dist_env = SimpleNamespace(is_main=True)
    r.target_wrapper = mock.MagicMock()
    r.wandb_run = mock.MagicMock()
    r.runtime = SimpleNamespace(global_step=0)
    r.grad_accumulation_steps = 1
    r.log_every_steps = 1
    r.total_optim_steps = 4
    r.warmup_steps = 0
    r.peak_lr = 1e-4
    r.min_lr_ratio = 0.1
    r._maybe_save_final_checkpoint = mock.MagicMock()
    for key, value in overrides.items():
        setattr(r, key, value)
    return r


def test_run_loop_finalizes_on_exception():
    """A mid-training exception still releases the target/W&B (and re-raises)."""
    recipe = _make_recipe()
    recipe._train_epochs = mock.MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        recipe.run_train_validation_loop()
    recipe.target_wrapper.close.assert_called_once()
    recipe.wandb_run.finish.assert_called_once()
    # The final checkpoint is success-only and must NOT be written after a crash.
    recipe._maybe_save_final_checkpoint.assert_not_called()


def test_run_loop_finalizes_on_success():
    """Normal completion saves the final checkpoint and releases resources."""
    recipe = _make_recipe()
    recipe._train_epochs = mock.MagicMock()
    recipe.run_train_validation_loop()
    recipe._train_epochs.assert_called_once()
    recipe._maybe_save_final_checkpoint.assert_called_once_with(1)
    recipe.target_wrapper.close.assert_called_once()
    recipe.wandb_run.finish.assert_called_once()


def test_early_return_finalizes():
    """The 'all epochs already completed' resume path also releases resources."""
    recipe = _make_recipe(_resume_epoch=5, num_epochs=1)
    recipe._train_epochs = mock.MagicMock()
    recipe.run_train_validation_loop()
    recipe._train_epochs.assert_not_called()
    recipe.target_wrapper.close.assert_called_once()
    recipe.wandb_run.finish.assert_called_once()


def test_finalize_is_best_effort():
    """A failure in one teardown step does not block the others."""
    recipe = _make_recipe()
    recipe.target_wrapper.close.side_effect = RuntimeError("server gone")
    recipe._finalize_training()  # must not raise
    recipe.wandb_run.finish.assert_called_once()


def test_finalize_handles_missing_backends():
    """The cached backend has no target_wrapper and W&B may be disabled (None)."""
    recipe = _make_recipe(target_wrapper=None, wandb_run=None)
    recipe._finalize_training()  # no error, no calls to make
