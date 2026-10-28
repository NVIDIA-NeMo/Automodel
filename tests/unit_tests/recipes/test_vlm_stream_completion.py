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

"""VLM recipe completion uses the same terminal checkpoint protocol as LLM."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


class _FiniteLoader:
    synchronized_stream = True

    def __init__(self):
        self.values = iter({"value": index} for index in range(5))
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def close(self):
        self.closed = True


def _recipe(max_steps=99):
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.dataloader = _FiniteLoader()
    recipe.step_scheduler = StepSchedulerConfig(
        global_batch_size=2, max_steps=max_steps, ckpt_every_steps=2, preemption_signal=None
    ).build(recipe.dataloader, 1, 1)
    recipe.model_parts = [torch.nn.Linear(1, 1)]
    recipe.max_grad_norm = 1.0
    recipe.val_dataloader = None
    recipe.metric_logger_train = SimpleNamespace(close=MagicMock())
    recipe.metric_logger_valid = SimpleNamespace(close=MagicMock())
    recipe.best_metric_key = "default"
    recipe._make_progress_bar = MagicMock(return_value=None)
    recipe._run_train_optim_step = MagicMock(return_value=SimpleNamespace(metrics={"loss": 0.25}))
    recipe.log_train_metrics = MagicMock()
    recipe._update_progress_bar = MagicMock()
    recipe._maybe_collect_garbage = MagicMock()
    recipe._finalize_and_close_checkpointer = MagicMock()
    return recipe


def test_vlm_saves_terminal_checkpoint_at_closed_pass_frontier():
    recipe = _recipe()
    saved = []

    def checkpoint(epoch, step, train_loss, val_loss, best_metric_key):
        assert not recipe.dataloader.closed
        saved.append((epoch, step, recipe.step_scheduler.state_dict(), recipe.step_scheduler.is_last_step))

    recipe.save_checkpoint = checkpoint
    recipe.run_train_validation_loop()
    assert recipe._run_train_optim_step.call_count == 2
    assert saved == [
        (0, 1, {"step": 2, "epoch": 0, "stream_exhausted": False}, False),
        (1, 1, {"step": 2, "epoch": 1, "stream_exhausted": True}, True),
    ]
    assert recipe.dataloader.closed
    recipe._finalize_and_close_checkpointer.assert_called_once()


def test_vlm_closes_stream_after_training_error():
    recipe = _recipe()
    recipe._run_train_optim_step.side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        recipe.run_train_validation_loop()
    assert recipe.dataloader.closed


def test_vlm_step_cap_saves_once_and_closes_stream():
    recipe = _recipe(max_steps=1)
    recipe.save_checkpoint = MagicMock()
    recipe.run_train_validation_loop()
    recipe._run_train_optim_step.assert_called_once()
    recipe.save_checkpoint.assert_called_once()
    assert not recipe.step_scheduler.stream_exhausted
    assert recipe.dataloader.closed


@pytest.mark.parametrize("training_error", [False, True])
def test_vlm_closes_validation_loader_before_progress_bar(training_error):
    recipe = _recipe(max_steps=1)
    closed = []
    recipe.val_dataloader = SimpleNamespace(close=lambda: closed.append("validation"))
    recipe._run_validation_loader = MagicMock(return_value=None)
    recipe._make_progress_bar.return_value = SimpleNamespace(close=lambda: closed.append("progress"))
    recipe.save_checkpoint = MagicMock()
    if training_error:
        recipe._run_train_optim_step.side_effect = RuntimeError("training failed")
        with pytest.raises(RuntimeError, match="training failed"):
            recipe.run_train_validation_loop()
    else:
        recipe.run_train_validation_loop()
    assert recipe.dataloader.closed
    assert closed == ["validation", "progress"]
