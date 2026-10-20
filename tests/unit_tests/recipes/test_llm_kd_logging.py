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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.recipes.llm import kd


@pytest.mark.parametrize("trackio_enabled", [False, True])
def test_validation_preserves_training_and_other_validation_metrics(monkeypatch, trackio_enabled):
    """W&B merges all logs at one step; each dataset must retain its own values."""
    history = {}

    def log(metrics, *, step):
        history.setdefault(step, {}).update(metrics)

    monkeypatch.setattr(kd, "_HAS_WANDB", True)
    monkeypatch.setattr(kd, "wandb", SimpleNamespace(run=object(), log=log))
    recipe = object.__new__(kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.dist_env = SimpleNamespace(is_main=True)
    recipe.step_scheduler = SimpleNamespace(is_remote_logging_step=True)
    recipe.metric_logger_train = Mock()
    recipe.trackio_logger = Mock() if trackio_enabled else None
    recipe.kd_ratio = 0.5
    training = MetricsSample(
        step=9,
        epoch=0,
        metrics={
            "loss": 2.0,
            "ce_loss": 3.0,
            "kd_loss": 1.0,
            "grad_norm": 0.5,
            "lr": 1e-5,
            "mem": 30.0,
            "tps": 100.0,
            "kd_ratio": 0.5,
            "temperature": 1.0,
        },
    )
    recipe.log_train_metrics(training)
    if trackio_enabled:
        recipe.trackio_logger.log_metrics.assert_called_once_with(training.to_dict(), step=training.step)
    for name, ce_loss, kd_loss in (("default", 5.0, 7.0), ("held_out", 9.0, 11.0)):
        validation = MetricsSample(
            step=9,
            epoch=0,
            metrics={
                "val_loss": (ce_loss + kd_loss) / 2,
                "ce_loss": ce_loss,
                "kd_loss": kd_loss,
                "lr": 1e-5,
                "mem": 12.0,
                "num_label_tokens": 100,
            },
        )
        local_logger = Mock()
        recipe.log_val_metrics(name, validation, local_logger)
        local_logger.log.assert_called_once_with(validation)
        if trackio_enabled:
            recipe.trackio_logger.log_metrics.assert_called_with(validation.to_dict(), step=validation.step)
        assert history[9][f"val_{name}/ce_loss"] == ce_loss
        assert history[9][f"val_{name}/kd_loss"] == kd_loss

    recipe.metric_logger_train.log.assert_called_once_with(training)
    for key, value in training.to_dict().items():
        assert history[9][key] == value
    assert history[9]["val_default/ce_loss"] == 5.0
    if trackio_enabled:
        assert recipe.trackio_logger.log_metrics.call_count == 3
