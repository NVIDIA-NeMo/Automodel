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

from nemo_automodel.components.config.loader import ConfigNode
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
            "num_tokens_per_step": 64,
            "mfu": 0.35,
        },
    )


def _ft_recipe(run, recipe_cls=None):
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    recipe = (recipe_cls or TrainFinetuneRecipeForNextTokenPrediction)(cfg=ConfigNode({}))
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
    assert args[0]["num_tokens_per_step"] == 64
    assert args[0]["mfu"] == pytest.approx(0.35)
    # ``step`` / ``timestamp`` are reserved by Trackio and must not be sent as metrics.
    assert "step" not in args[0] and "timestamp" not in args[0]


def test_log_train_metrics_skips_trackio_off_the_remote_logging_step():
    run = Mock()
    recipe = _ft_recipe(run)
    recipe.step_scheduler.is_remote_logging_step = False

    recipe.log_train_metrics(_train_sample())

    run.log.assert_not_called()


@pytest.mark.parametrize("recipe_name", ["llm", "kd"])
def test_remote_logs_receive_the_latest_completed_throughput_window(monkeypatch, recipe_name):
    from nemo_automodel.components.loggers import step_metrics
    from nemo_automodel.recipes.llm.kd import KnowledgeDistillationRecipeForNextTokenPrediction

    run = Mock()
    recipe = _ft_recipe(run, KnowledgeDistillationRecipeForNextTokenPrediction if recipe_name == "kd" else None)
    recipe.kd_ratio = 0.5
    recipe.metric_logger_train = Mock()
    reporter = step_metrics.TrainingStepReporter(
        torch.device("cpu"), world_size=1, pack_size=0, domain_names=(), reference_mfu=None, calculate_mfu=Mock()
    )
    # Deliver two samples per poll. Windows close on steps 21 and 41, while the
    # remote sink only receives steps divisible by ten.
    clock = iter(range(26))
    monkeypatch.setattr(step_metrics, "perf_counter", lambda: next(clock))
    for step in range(52):
        sample = _train_sample(step=step)
        sample.metrics.update(ce_loss=1.0, kd_loss=1.5, kd_ratio=0.5, temperature=1.0)
        sample.metrics["num_tokens_per_step"] = 64 if step < 22 else 32
        reporter.enqueue(sample, start=0.0, end=1.0, remote_log=step % 10 == 0)
        if step % 2:
            for done, remote, _ in reporter.poll():
                recipe.log_train_metrics(done, remote_log=remote)
    reporter.close()

    local_rates = {
        sample.step: sample.metrics["end_to_end_tps"]
        for (sample,), _ in recipe.metric_logger_train.log.call_args_list
        if "end_to_end_tps" in sample.metrics
    }
    remote_rates = {
        kwargs["step"]: payload["end_to_end_tps"]
        for (payload,), kwargs in run.log.call_args_list
        if "end_to_end_tps" in payload
    }
    assert local_rates == {21: 128, 41: 64}
    assert remote_rates == {30: 128, 50: 64}


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

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg=ConfigNode({}))
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
    second.metrics["end_to_end_tps"] = 12.0
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
    assert args[0]["end_to_end_tps"] == 12.0
    assert "mode" not in args[0]


@pytest.mark.parametrize("recipe_name", ["kd", "dllm"])
def test_loop_subclass_loggers_use_the_reported_step_and_cadence(recipe_name):
    """The inherited training loop logs a step after the scheduler has moved past it."""
    from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe
    from nemo_automodel.recipes.llm.kd import KnowledgeDistillationRecipeForNextTokenPrediction

    recipe_cls = {"kd": KnowledgeDistillationRecipeForNextTokenPrediction, "dllm": DiffusionLMSFTRecipe}[recipe_name]
    run = Mock()
    recipe = _ft_recipe(run, recipe_cls)
    recipe.kd_ratio = 0.5
    recipe.step_scheduler = types.SimpleNamespace(step=9, is_remote_logging_step=False)
    sample = _train_sample(step=7)
    sample.metrics.update(ce_loss=1.0, kd_loss=1.5, kd_ratio=0.5, temperature=1.0, dllm_loss=1.25, mode="mdlm")

    recipe.log_train_metrics(sample, remote_log=True)

    run.log.assert_called_once()
    assert run.log.call_args.kwargs["step"] == 7


@pytest.mark.parametrize("recipe_name", ["eagle1", "eagle3", "dflash", "dspark"])
@pytest.mark.parametrize("wandb_active,trackio_active", [(False, False), (False, True), (True, False), (True, True)])
def test_speculative_trainer_log_helper_calls_trackio(recipe_name, wandb_active, trackio_active):
    from nemo_automodel.recipes.speculative_decoding.train_dflash import TrainDFlashRecipe
    from nemo_automodel.recipes.speculative_decoding.train_dspark import TrainDSparkRecipe
    from nemo_automodel.recipes.speculative_decoding.train_eagle1 import TrainEagle1Recipe
    from nemo_automodel.recipes.speculative_decoding.train_eagle3 import TrainEagle3Recipe

    recipe_cls = {
        "eagle1": TrainEagle1Recipe,
        "eagle3": TrainEagle3Recipe,
        "dflash": TrainDFlashRecipe,
        "dspark": TrainDSparkRecipe,
    }[recipe_name]
    run = Mock()
    wandb_run = Mock()
    recipe = recipe_cls.__new__(recipe_cls)
    recipe.wandb_run = wandb_run if wandb_active else None
    recipe.trackio_logger = TrackioLogger(run) if trackio_active else None

    recipe._wandb_log({"train/loss": 0.5}, step=3)

    expected = [call({"train/loss": 0.5}, step=3)]
    assert run.log.call_args_list == (expected if trackio_active else [])
    assert wandb_run.log.call_args_list == (expected if wandb_active else [])
