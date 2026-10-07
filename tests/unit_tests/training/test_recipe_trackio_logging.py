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

import pytest
import torch

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.trackio_utils import TrackioLogger


class _RecordingRun:
    def __init__(self):
        self.calls = []

    def log(self, metrics, step=None):
        self.calls.append((metrics, step))

    def finish(self):
        pass


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
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
            "dllm_loss": 1.25,
        },
    )


def _ft_recipe(run, recipe_cls=None):
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    recipe = (recipe_cls or TrainFinetuneRecipeForNextTokenPrediction)(cfg=None)
    recipe.dist_env = types.SimpleNamespace(is_main=True)
    recipe.step_scheduler = types.SimpleNamespace(step=7, is_remote_logging_step=True)
    recipe.metric_logger_train = types.SimpleNamespace(log=lambda x: None)
    recipe.comet_logger = None
    recipe.trackio_logger = TrackioLogger(run)
    return recipe


def test_train_ft_logs_train_metrics_at_the_scheduler_step():
    run = _RecordingRun()
    recipe = _ft_recipe(run)
    recipe._moe_layer_loads = None

    recipe.log_train_metrics(_train_sample())

    ((metrics, step),) = run.calls
    assert step == 7
    assert metrics["loss"] == 1.25 and metrics["epoch"] == 1.0
    assert "step" not in metrics and "timestamp" not in metrics


def test_train_ft_skips_trackio_off_the_remote_logging_step():
    run = _RecordingRun()
    recipe = _ft_recipe(run)
    recipe.step_scheduler.is_remote_logging_step = False

    recipe.log_train_metrics(_train_sample())

    assert run.calls == []


def test_train_ft_prefixes_named_validation_sets():
    run = _RecordingRun()
    recipe = _ft_recipe(run)
    val = MetricsSample(step=5, epoch=0, metrics={"val_loss": 0.75, "lr": 1e-3, "num_label_tokens": 3})

    recipe.log_val_metrics("default", val)
    recipe.log_val_metrics("squad", val)

    assert run.calls == [
        ({"val_loss": 0.75, "lr": 1e-3, "num_label_tokens": 3.0}, 5),
        ({"val_squad/val_loss": 0.75, "val_squad/lr": 1e-3, "val_squad/num_label_tokens": 3.0}, 5),
    ]


def test_train_ft_without_trackio_logs_nothing():
    """Recipes built without a ``trackio:`` section (or before setup) skip Trackio."""
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg=None)
    recipe.dist_env = types.SimpleNamespace(is_main=True)
    recipe.step_scheduler = types.SimpleNamespace(step=7, is_remote_logging_step=True)
    recipe.metric_logger_train = types.SimpleNamespace(log=lambda x: None)
    recipe.comet_logger = None

    recipe.log_train_metrics(_train_sample())  # no trackio_logger attribute: must not raise


def test_dllm_logs_the_window_mean_to_trackio():
    from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe

    run = _RecordingRun()
    recipe = _ft_recipe(run, DiffusionLMSFTRecipe)
    first, second = _train_sample(step=6), _train_sample(step=7)
    second.metrics["loss"] = 0.75
    for sample in (first, second):
        sample.metrics["mode"] = "uno"  # dLLM samples carry the strategy name as a string
    recipe.step_scheduler.is_remote_logging_step = False
    recipe.log_train_metrics(first)
    recipe.step_scheduler.is_remote_logging_step = True
    recipe.log_train_metrics(second)

    ((metrics, step),) = run.calls
    assert step == 7
    assert metrics["loss"] == pytest.approx(1.0)
    assert "mode" not in metrics


def test_speculative_trainer_log_helper_forwards_to_trackio():
    from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe

    run = _RecordingRun()
    recipe = TrainEagle1Recipe.__new__(TrainEagle1Recipe)
    recipe.wandb_run = None
    recipe.trackio_logger = TrackioLogger(run)

    recipe._wandb_log({"train/loss": 0.5}, step=3)

    assert run.calls == [({"train/loss": 0.5}, 3)]
