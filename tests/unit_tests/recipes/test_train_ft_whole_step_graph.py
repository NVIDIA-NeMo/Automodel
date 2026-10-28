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

import pytest
import torch
from torch import nn

from nemo_automodel.components.cuda_graphs import WholeStepGraph
from nemo_automodel.components.models.common import BackendConfig, CudaGraphConfig
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    _build_whole_step_graph,
)


def _bare_recipe() -> TrainFinetuneRecipeForNextTokenPrediction:
    return TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)


def _model(mode: str) -> nn.Module:
    model = nn.Linear(2, 2)
    model.backend = BackendConfig(dispatcher="torch", cuda_graph=CudaGraphConfig(mode=mode))
    return model


def _step_fn(batches):
    return [torch.zeros(()) for _ in batches]


class TestBuildWholeStepGraph:
    def test_partial_mode_builds_nothing(self):
        result = _build_whole_step_graph(
            [_model("partial")],
            step_fn=_step_fn,
            device=torch.device("cpu"),
            pipeline_parallel=False,
            partial_graphs_enabled=False,
        )
        assert result is None

    def test_model_without_backend_builds_nothing(self):
        result = _build_whole_step_graph(
            [nn.Linear(2, 2)],
            step_fn=_step_fn,
            device=torch.device("cpu"),
            pipeline_parallel=False,
            partial_graphs_enabled=False,
        )
        assert result is None

    def test_whole_step_mode_builds_uncaptured_graph(self):
        result = _build_whole_step_graph(
            [_model("whole_step")],
            step_fn=_step_fn,
            device=torch.device("cpu"),
            pipeline_parallel=False,
            partial_graphs_enabled=False,
        )
        assert isinstance(result, WholeStepGraph)
        assert not result.captured

    def test_pipeline_parallel_is_rejected(self):
        with pytest.raises(ValueError, match="pipeline parallelism"):
            _build_whole_step_graph(
                [_model("whole_step")],
                step_fn=_step_fn,
                device=torch.device("cpu"),
                pipeline_parallel=True,
                partial_graphs_enabled=False,
            )

    def test_partial_graphs_are_rejected(self):
        with pytest.raises(ValueError, match="partial cuda_graph.modules"):
            _build_whole_step_graph(
                [_model("whole_step")],
                step_fn=_step_fn,
                device=torch.device("cpu"),
                pipeline_parallel=False,
                partial_graphs_enabled=True,
            )

    def test_multiple_model_parts_are_rejected(self):
        with pytest.raises(ValueError, match="single model part"):
            _build_whole_step_graph(
                [_model("whole_step"), _model("whole_step")],
                step_fn=_step_fn,
                device=torch.device("cpu"),
                pipeline_parallel=False,
                partial_graphs_enabled=False,
            )

    def test_gate_bias_update_models_are_rejected(self):
        class _GateBiasModel(nn.Linear):
            def update_moe_gate_bias(self) -> None:
                return None

        model = _GateBiasModel(2, 2)
        model.backend = BackendConfig(dispatcher="torch", cuda_graph=CudaGraphConfig(mode="whole_step"))
        with pytest.raises(NotImplementedError, match="update_moe_gate_bias"):
            _build_whole_step_graph(
                [model],
                step_fn=_step_fn,
                device=torch.device("cpu"),
                pipeline_parallel=False,
                partial_graphs_enabled=False,
            )


def test_training_loop_captures_first_step_batches_and_resets():
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
            yield ["step-0a", "step-0b"]
            self.step = 1
            yield ["step-1a", "step-1b"]

    graph = SimpleNamespace(
        capture=lambda batches: events.append(("capture", tuple(batches))),
        reset=lambda: events.append("reset"),
    )
    recipe = _bare_recipe()
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.step_scheduler = _TwoStepScheduler()
    recipe.max_grad_norm = 1.0
    recipe.partial_cuda_graph_manager = None
    recipe._partial_cuda_graph_capture_pending = False
    recipe.whole_step_graph = graph
    recipe._whole_step_graph_capture_pending = True
    recipe._enable_qat_if_delayed = lambda _step: None
    recipe._run_train_optim_step = lambda batches, _norm: (
        events.append(("train-step", tuple(batches))) or SimpleNamespace(metrics={"loss": 1.0})
    )
    recipe._collect_moe_load_balance = lambda: None
    recipe.log_train_metrics = lambda _metrics: None
    recipe._update_progress_bar = lambda _pbar, _metrics: None
    recipe._make_progress_bar = lambda: None
    recipe.val_dataloaders = {}
    recipe.save_checkpoint = lambda *_args, **_kwargs: None
    recipe._maybe_collect_garbage = lambda: None
    recipe.metric_logger_train = SimpleNamespace(close=lambda: None)
    recipe.metric_logger_valid = {}
    recipe.checkpointer = SimpleNamespace(finalize=lambda: None)
    recipe.best_metric_key = "default"

    recipe.run_train_validation_loop()

    assert events == [
        ("train-step", ("step-0a", "step-0b")),
        ("capture", ("step-0a", "step-0b")),
        ("train-step", ("step-1a", "step-1b")),
        "reset",
    ]
    assert recipe.whole_step_graph is None
    assert not recipe._whole_step_graph_capture_pending


def test_recipe_without_setup_stays_on_eager_path():
    recipe = _bare_recipe()
    assert recipe.whole_step_graph is None
    assert recipe._whole_step_num_label_tokens is None
    assert not recipe._whole_step_graph_validate_pending
