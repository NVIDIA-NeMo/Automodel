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

from types import SimpleNamespace
from unittest.mock import MagicMock

from torch import nn

import nemo_automodel.components.cuda_graphs.partial as partial_graphs
from nemo_automodel.components.models.common import BackendConfig, CudaGraphConfig
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    _build_partial_cuda_graph_manager,
)


def _bare_recipe() -> TrainFinetuneRecipeForNextTokenPrediction:
    return TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)


def test_builder_forwards_runtime_safety_context(monkeypatch):
    model = nn.Linear(2, 2)
    model.backend = BackendConfig(attn="te", cuda_graph=CudaGraphConfig(modules=["te_dpa"]))
    model_parts = [model]
    manager = SimpleNamespace(capture=MagicMock())
    discover = MagicMock(return_value=manager)
    monkeypatch.setattr(partial_graphs.PartialCudaGraphManager, "from_model_parts", discover)

    result = _build_partial_cuda_graph_manager(
        model_parts,
        activation_checkpointing=True,
        pipeline_parallel=False,
    )

    discover.assert_called_once_with(
        model_parts,
        activation_checkpointing=True,
        pipeline_parallel=False,
    )
    assert result is manager


def test_builder_does_not_use_manager_when_no_scope_is_enabled(monkeypatch):
    discover = MagicMock()
    monkeypatch.setattr(partial_graphs.PartialCudaGraphManager, "from_model_parts", discover)

    result = _build_partial_cuda_graph_manager(
        [nn.Linear(2, 2)],
        activation_checkpointing=False,
        pipeline_parallel=False,
    )

    assert result is None
    discover.assert_not_called()


def test_training_loop_captures_after_first_complete_step_and_closes():
    events = []

    class _TwoStepScheduler:
        step = 0
        epoch = 0
        epochs = [0]
        is_val_step = False
        is_ckpt_step = False
        sigterm_flag = False

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            yield ["step-0"]
            self.step = 1
            yield ["step-1"]

    manager = SimpleNamespace(
        capture=lambda: events.append("capture"),
        close=lambda: events.append("close"),
    )
    recipe = _bare_recipe()
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.step_scheduler = _TwoStepScheduler()
    recipe.max_grad_norm = 1.0
    recipe.partial_cuda_graph_manager = manager
    recipe._partial_cuda_graph_capture_pending = True
    recipe._enable_qat_if_delayed = lambda _step: None
    recipe._run_train_optim_step = lambda batches, _norm: (
        events.append(("train-step", tuple(batches))) or SimpleNamespace(metrics={"loss": 1.0})
    )
    recipe._collect_moe_load_balance = lambda: None
    recipe.log_train_metrics = lambda _metrics: None
    recipe._update_progress_bar = lambda _pbar, _metrics: None
    recipe._make_progress_bar = lambda: SimpleNamespace(close=lambda: events.append("progress-close"))
    recipe.val_dataloaders = {}
    recipe.save_checkpoint = lambda *_args, **_kwargs: None
    recipe._maybe_collect_garbage = lambda: None
    recipe.metric_logger_train = SimpleNamespace(close=lambda: events.append("metrics-close"))
    recipe.metric_logger_valid = {}
    recipe.checkpointer = SimpleNamespace(close=lambda: events.append("checkpointer-close"))
    recipe.best_metric_key = "default"

    recipe.run_train_validation_loop()

    assert events == [
        ("train-step", ("step-0",)),
        "capture",
        ("train-step", ("step-1",)),
        "progress-close",
        "metrics-close",
        "checkpointer-close",
        "close",
    ]
    assert recipe.partial_cuda_graph_manager is None
    assert not recipe._partial_cuda_graph_capture_pending
