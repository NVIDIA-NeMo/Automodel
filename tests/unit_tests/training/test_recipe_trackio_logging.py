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

"""Recipes route train / validation metrics to the Trackio logger built in ``setup``."""

import types
from unittest.mock import Mock, call

import pytest
import torch

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.trackio_utils import TrackioLogger


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    # Avoid cuda calls on environments without GPUs
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)


def _train_sample(step=7):
    return MetricsSample(
        step=step,
        epoch=1,
        metrics={
            "loss": 1.25,
            "grad_norm": 0.5,
            "lr": 1e-3,
            "mem": 0.1,
            "tps": 10.0,
            "tps_per_gpu": 5.0,
            "num_label_tokens": 42,
        },
    )


def _ft_recipe(run, recipe_cls=None):
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    recipe = (recipe_cls or TrainFinetuneRecipeForNextTokenPrediction)(cfg=None)
    # Minimal attributes required by the logging methods
    recipe.dist_env = types.SimpleNamespace(is_main=True)
    recipe.step_scheduler = types.SimpleNamespace(step=7, is_remote_logging_step=True)
    recipe.metric_logger_train = types.SimpleNamespace(log=lambda x: None)
    recipe.comet_logger = None
    recipe._moe_layer_loads = None
    recipe.trackio_logger = TrackioLogger(run)
    return recipe


def test_log_train_metrics_calls_trackio():
    run = Mock()
    recipe = _ft_recipe(run)

    recipe.log_train_metrics(_train_sample())

    run.log.assert_called_once()
    args, kwargs = run.log.call_args
    assert kwargs.get("step") == 7
    assert args[0]["loss"] == 1.25 and args[0]["epoch"] == 1.0
    # ``step`` / ``timestamp`` are reserved by Trackio and must not be sent as metrics.
    assert "step" not in args[0] and "timestamp" not in args[0]


def test_log_train_metrics_skips_trackio_off_the_remote_logging_step():
    run = Mock()
    recipe = _ft_recipe(run)
    recipe.step_scheduler.is_remote_logging_step = False

    recipe.log_train_metrics(_train_sample())

    run.log.assert_not_called()


def test_log_val_metrics_calls_trackio_with_named_set_prefix():
    run = Mock()
    recipe = _ft_recipe(run)
    log_data = MetricsSample(step=5, epoch=0, metrics={"val_loss": 0.75, "lr": 1e-3, "num_label_tokens": 3})

    recipe.log_val_metrics("default", log_data, metric_logger=None)
    recipe.log_val_metrics("squad", log_data, metric_logger=None)

    assert run.log.call_args_list == [
        call({"val_loss": 0.75, "lr": 1e-3, "num_label_tokens": 3.0}, step=5),
        call({"val_squad/val_loss": 0.75, "val_squad/lr": 1e-3, "val_squad/num_label_tokens": 3.0}, step=5),
    ]


def test_log_train_metrics_without_trackio_logger_does_not_raise():
    """Recipes built without a ``trackio:`` section (or before setup) skip Trackio."""
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg=None)
    recipe.dist_env = types.SimpleNamespace(is_main=True)
    recipe.step_scheduler = types.SimpleNamespace(step=7, is_remote_logging_step=True)
    recipe.metric_logger_train = types.SimpleNamespace(log=lambda x: None)
    recipe.comet_logger = None
    recipe._moe_layer_loads = None

    recipe.log_train_metrics(_train_sample())


def test_dllm_log_train_metrics_sends_the_window_mean_to_trackio():
    from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe

    run = Mock()
    recipe = _ft_recipe(run, DiffusionLMSFTRecipe)
    first, second = _train_sample(step=6), _train_sample(step=7)
    second.metrics["loss"] = 0.75
    for sample in (first, second):
        # dLLM samples carry the diffusion loss and the strategy name as a string
        sample.metrics.update(dllm_loss=sample.metrics["loss"], mode="uno")

    recipe.step_scheduler.is_remote_logging_step = False
    recipe.log_train_metrics(first)
    recipe.step_scheduler.is_remote_logging_step = True
    recipe.log_train_metrics(second)

    run.log.assert_called_once()
    args, kwargs = run.log.call_args
    assert kwargs.get("step") == 7
    assert args[0]["loss"] == pytest.approx(1.0)
    assert "mode" not in args[0]


def test_speculative_trainer_log_helper_calls_trackio():
    from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe

    run = Mock()
    recipe = TrainEagle1Recipe.__new__(TrainEagle1Recipe)
    recipe.wandb_run = None
    recipe.trackio_logger = TrackioLogger(run)

    recipe._wandb_log({"train/loss": 0.5}, step=3)

    run.log.assert_called_once_with({"train/loss": 0.5}, step=3)
