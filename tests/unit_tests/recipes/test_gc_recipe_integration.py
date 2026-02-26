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

from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.recipes.biencoder.train_biencoder import TrainBiencoderRecipe
from nemo_automodel.recipes.llm.train_seq_cls import TrainFinetuneRecipeForSequenceClassification
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


class _OneStepScheduler:
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.epochs = [0]
        self.is_val_step = False
        self.is_ckpt_step = False

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        yield ["dummy-batch"]


def _dummy_metrics():
    return MetricsSample(step=0, epoch=0, metrics={"loss": 1.0})


def test_seq_cls_loop_calls_gc_hook():
    recipe = TrainFinetuneRecipeForSequenceClassification.__new__(TrainFinetuneRecipeForSequenceClassification)
    recipe.model_parts = [MagicMock()]
    recipe.step_scheduler = _OneStepScheduler()
    recipe._run_train_optim_step = MagicMock(return_value=_dummy_metrics())
    recipe._maybe_collect_garbage = MagicMock()
    recipe.log_train_metrics = MagicMock()
    recipe.log_val_metrics = MagicMock()
    recipe.save_checkpoint = MagicMock()
    recipe.val_dataloader = None
    recipe.metric_logger_train = SimpleNamespace(close=MagicMock())
    recipe.metric_logger_valid = SimpleNamespace(close=MagicMock())
    recipe.checkpointer = SimpleNamespace(close=MagicMock())
    recipe.best_metric_key = "default"

    recipe.run_train_validation_loop()

    recipe._maybe_collect_garbage.assert_called_once()


def test_biencoder_loop_calls_gc_hook():
    recipe = TrainBiencoderRecipe.__new__(TrainBiencoderRecipe)
    recipe.model_parts = [MagicMock()]
    recipe.step_scheduler = _OneStepScheduler()
    recipe.max_grad_norm = 1.0
    recipe._run_train_optim_step = MagicMock(return_value=_dummy_metrics())
    recipe._maybe_collect_garbage = MagicMock()
    recipe.log_train_metrics = MagicMock()
    recipe.log_val_metrics = MagicMock()
    recipe.save_checkpoint = MagicMock()
    recipe.val_dataloader = None
    recipe.metric_logger_train = SimpleNamespace(close=MagicMock())
    recipe.metric_logger_valid = SimpleNamespace(close=MagicMock())
    recipe.checkpointer = SimpleNamespace(close=MagicMock())

    recipe.run_train_validation_loop()

    recipe._maybe_collect_garbage.assert_called_once()


def test_vlm_loop_calls_gc_hook():
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.model_parts = [MagicMock()]
    recipe.step_scheduler = _OneStepScheduler()
    recipe.max_grad_norm = 1.0
    recipe._run_train_optim_step = MagicMock(return_value=_dummy_metrics())
    recipe._maybe_collect_garbage = MagicMock()
    recipe.log_train_metrics = MagicMock()
    recipe.log_val_metrics = MagicMock()
    recipe.save_checkpoint = MagicMock()
    recipe.val_dataloader = None
    recipe.metric_logger_train = SimpleNamespace(close=MagicMock())
    recipe.metric_logger_valid = SimpleNamespace(close=MagicMock())
    recipe.checkpointer = SimpleNamespace(close=MagicMock())
    recipe.best_metric_key = "default"
    recipe.pp_enabled = False

    recipe.run_train_validation_loop()

    recipe._maybe_collect_garbage.assert_called_once()
