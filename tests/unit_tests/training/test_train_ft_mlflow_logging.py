# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import types
from unittest.mock import Mock

import pytest

from nemo_automodel.components.loggers.metric_logger import MetricsSample


def _install_fake_wandb():
    """
    Provide a minimal 'wandb' package so train_ft can be imported without the real dependency.
    """
    wandb = types.ModuleType("wandb")
    wandb.run = None

    class Settings:
        def __init__(self, *a, **kw):
            pass

    def init(*a, **kw):
        class Run:
            url = "http://example.com/run"

        return Run()

    wandb.Settings = Settings
    wandb.init = init
    sys.modules["wandb"] = wandb


@pytest.fixture(autouse=True)
def _ensure_fake_wandb():
    original = dict(sys.modules)
    _install_fake_wandb()
    yield
    # restore sys.modules
    for k in list(sys.modules.keys()):
        if k.startswith("wandb") and k not in original:
            del sys.modules[k]
    for k, v in original.items():
        sys.modules[k] = v


def test_log_train_metrics_calls_mlflow(monkeypatch):
    # Defer import until after fake wandb is in place
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg=None)
    # Minimal attributes required by the method
    recipe.dist_env = types.SimpleNamespace(is_main=True)
    recipe.step_scheduler = types.SimpleNamespace(step=7, is_remote_logging_step=True)
    recipe.metric_logger_train = types.SimpleNamespace(log=lambda x: None)
    mlflow_mock = Mock()
    recipe.mlflow_logger = types.SimpleNamespace(log_metrics=mlflow_mock)

    # Avoid cuda calls on environments without GPUs
    import torch.cuda

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)

    log_data = MetricsSample(step=7, epoch=1, metrics={"loss": 1.23, "grad_norm": 0.5, "lr": 1e-3, "mem": 0.1, "tps": 10.0, "tps_per_gpu": 5.0, "num_label_tokens": 42})
    recipe.log_train_metrics(log_data)

    mlflow_mock.assert_called_once()
    args, kwargs = mlflow_mock.call_args
    # First arg is a flat dict of metrics + step/epoch/timestamp
    assert isinstance(args[0], dict) and kwargs.get("step") == log_data.step


def test_log_val_metrics_calls_mlflow(monkeypatch):
    # Defer import until after fake wandb is in place
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg=None)
    recipe.dist_env = types.SimpleNamespace(is_main=True)
    mlflow_mock = Mock()
    recipe.mlflow_logger = types.SimpleNamespace(log_metrics=mlflow_mock)
    # No JSONL logger passed (None) to keep test minimal

    log_data = MetricsSample(step=3, epoch=0, metrics={"val_loss": 0.99, "lr": 5e-4, "num_label_tokens": 100, "mem": 0.2})
    recipe.log_val_metrics("default", log_data, metric_logger=None)

    mlflow_mock.assert_called_once()
    args, kwargs = mlflow_mock.call_args
    assert isinstance(args[0], dict) and kwargs.get("step") == log_data.step
