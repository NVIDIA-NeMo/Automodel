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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import nemo_automodel.recipes.llm.partial_cuda_graphs as partial_graphs
import nemo_automodel.recipes.llm.train_ft as train_ft
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


class _RecordingOptimizer:
    def __init__(self, events: list[object] | None = None) -> None:
        self.events = events if events is not None else []
        self.param_groups = [{"lr": 0.01}]

    def step(self) -> None:
        self.events.append("optimizer-step")

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.events.append(("zero-grad", set_to_none))


class _NoSetToNoneOptimizer:
    def zero_grad(self) -> None:
        pass


def _bare_recipe() -> TrainFinetuneRecipeForNextTokenPrediction:
    return TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)


def test_setup_partial_graphs_forwards_runtime_safety_context(monkeypatch):
    recipe = _bare_recipe()
    model_parts = [nn.Linear(2, 2)]
    manager = SimpleNamespace(capture=MagicMock())
    optimizer = _RecordingOptimizer()
    recipe.model_parts = model_parts
    recipe.optimizer = [optimizer]
    recipe.activation_checkpointing = True
    recipe.pp_enabled = False

    discover = MagicMock(return_value=manager)
    monkeypatch.setattr(partial_graphs.PartialCudaGraphManager, "from_model_parts", discover)

    recipe._setup_partial_cuda_graphs()

    discover.assert_called_once_with(
        model_parts,
        activation_checkpointing=True,
        pipeline_parallel=False,
    )
    assert recipe.partial_cuda_graph_manager is manager
    assert recipe._partial_cuda_graph_capture_pending is True


def test_setup_partial_graphs_stays_inert_when_no_feature_is_enabled(monkeypatch):
    recipe = _bare_recipe()
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.optimizer = [_NoSetToNoneOptimizer()]
    recipe.activation_checkpointing = False
    recipe.pp_enabled = False
    monkeypatch.setattr(partial_graphs.PartialCudaGraphManager, "from_model_parts", MagicMock(return_value=None))

    recipe._setup_partial_cuda_graphs()

    assert recipe.partial_cuda_graph_manager is None
    assert recipe._partial_cuda_graph_capture_pending is False


def test_setup_weight_cache_optimizer_hooks_replaces_old_handles(monkeypatch):
    import nemo_automodel.components.moe.experts as experts_module

    recipe = _bare_recipe()
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.optimizer = [object()]
    old_handle = SimpleNamespace(remove=MagicMock())
    new_handle = object()
    recipe._te_ops_mxfp8_weight_cache_optimizer_hook_handles = (old_handle,)
    register = MagicMock(return_value=(new_handle,))
    monkeypatch.setattr(experts_module, "register_te_ops_mxfp8_weight_cache_optimizer_hooks", register)

    recipe._setup_te_ops_mxfp8_weight_cache_optimizer_hooks()

    old_handle.remove.assert_called_once_with()
    register.assert_called_once_with(recipe.model_parts, recipe.optimizer)
    assert recipe._te_ops_mxfp8_weight_cache_optimizer_hook_handles == (new_handle,)


def test_setup_weight_cache_refresh_graph_excludes_managed_owners_from_hooks(monkeypatch):
    import nemo_automodel.components.moe.experts as experts_module
    import nemo_automodel.recipes.llm.mxfp8_cache_refresh_cuda_graph as cache_graph_module

    recipe = _bare_recipe()
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.optimizer = [_RecordingOptimizer()]
    recipe._full_iteration_backend = SimpleNamespace(
        te_ops_mxfp8_weight_cache_cuda_graph=True,
        full_iteration_cuda_graph_single_mempool=True,
    )
    managed_ids = frozenset({17, 19})
    graph_hook_handle = object()
    owning_optimizer = SimpleNamespace(register_step_post_hook=MagicMock(return_value=graph_hook_handle))
    target = SimpleNamespace(
        managed_owner_ids=managed_ids,
        graph_signature=lambda: ((1,),),
        optimizer=owning_optimizer,
    )
    build_target = MagicMock(return_value=target)
    manager = SimpleNamespace(managed_owner_ids=managed_ids, target=target, close=MagicMock())
    manager_ctor = MagicMock(return_value=manager)
    register = MagicMock(return_value=())
    monkeypatch.setattr(experts_module, "build_te_ops_mxfp8_weight_cache_refresh_target", build_target)
    monkeypatch.setattr(experts_module, "register_te_ops_mxfp8_weight_cache_optimizer_hooks", register)
    monkeypatch.setattr(cache_graph_module, "MXFP8CacheRefreshCudaGraphManager", manager_ctor)

    recipe._setup_te_ops_mxfp8_weight_cache_optimizer_hooks()

    build_target.assert_called_once_with(recipe.model_parts, recipe.optimizer)
    manager_ctor.assert_called_once_with(target, use_single_mempool=True)
    register.assert_called_once_with(recipe.model_parts, recipe.optimizer, excluded_owner_ids=managed_ids)
    owning_optimizer.register_step_post_hook.assert_called_once()
    assert recipe.te_ops_mxfp8_cache_refresh_cuda_graph_manager is manager
    assert recipe._te_ops_mxfp8_weight_cache_optimizer_hook_handles == (graph_hook_handle,)


def test_partial_graphs_reject_optimizer_without_preserving_zero_grad():
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = object()
    recipe.optimizer = [_NoSetToNoneOptimizer()]

    with pytest.raises(RuntimeError, match=r"zero_grad\(set_to_none=False\)"):
        recipe._validate_partial_graph_optimizers()


@pytest.mark.parametrize(
    ("manager", "expected"),
    [
        (None, ("zero-grad", True)),
        (object(), ("zero-grad", False)),
    ],
)
def test_optimizer_zeroing_preserves_graph_grad_buffers_only_when_active(manager, expected):
    recipe = _bare_recipe()
    optimizer = _RecordingOptimizer()
    recipe.optimizer = [optimizer]
    recipe.partial_cuda_graph_manager = manager

    recipe._zero_optimizer_gradients()

    assert optimizer.events == [expected]


def test_gradient_accumulation_finishes_before_optimizer_and_preserving_zero(monkeypatch):
    events: list[object] = []
    recipe = _bare_recipe()
    recipe.cfg = {}
    recipe.pp_enabled = False
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.optimizer = [_RecordingOptimizer(events)]
    recipe.partial_cuda_graph_manager = object()
    recipe.lr_scheduler = None
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: events.append("staging-wait"))
    recipe.step_scheduler = SimpleNamespace(step=1, epoch=0)
    recipe.timestamp = 0.0
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe._get_dp_group_size = lambda include_cp=False: 1
    recipe._get_cp_group_size = lambda: 1
    recipe._dp_allreduce = lambda value, include_cp=False: value

    def forward_backward(index, _batch, **_kwargs):
        events.append(("microbatch", index))
        _kwargs["loss_buffer"].append(torch.tensor(0.5))

    recipe._forward_backward_step = forward_backward
    monkeypatch.setattr(train_ft, "prepare_for_grad_accumulation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_ft, "prepare_for_final_backward", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_ft, "prepare_after_first_microbatch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_ft, "scale_grads_and_clip_grad_norm", lambda *_args, **_kwargs: torch.tensor(1.0))
    monkeypatch.setattr(train_ft.time, "perf_counter", lambda: 1.0)
    batches = [
        {"labels": torch.tensor([[1, 2]])},
        {"labels": torch.tensor([[3, 4]])},
    ]

    recipe._run_train_optim_step(batches, max_grad_norm=1.0)

    assert events == [
        ("microbatch", 0),
        ("microbatch", 1),
        "staging-wait",
        "optimizer-step",
        ("zero-grad", False),
    ]


def test_training_loop_captures_once_after_first_complete_eager_step():
    events: list[object] = []

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
            yield ["step-0-microbatch-0", "step-0-microbatch-1"]
            self.step = 1
            yield ["step-1-microbatch-0", "step-1-microbatch-1"]

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
    recipe._zero_optimizer_gradients = lambda: events.append("capture-zero")
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
    recipe.checkpointer = SimpleNamespace(close=lambda: None)
    recipe.best_metric_key = "default"

    recipe.run_train_validation_loop()

    assert events == [
        ("train-step", ("step-0-microbatch-0", "step-0-microbatch-1")),
        "capture-zero",
        "capture",
        "capture-zero",
        ("train-step", ("step-1-microbatch-0", "step-1-microbatch-1")),
        "close",
    ]
    assert recipe._partial_cuda_graph_capture_pending is False
    assert recipe.partial_cuda_graph_manager is None


def test_training_loop_closes_partial_graphs_when_a_step_raises():
    events = []

    class _OneStepScheduler:
        step = 0
        epoch = 0
        epochs = [0]

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            yield ["failing-step"]

    recipe = _bare_recipe()
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.step_scheduler = _OneStepScheduler()
    recipe.max_grad_norm = 1.0
    recipe.partial_cuda_graph_manager = SimpleNamespace(close=lambda: events.append("close"))
    recipe._partial_cuda_graph_capture_pending = False
    recipe._enable_qat_if_delayed = lambda _step: None
    recipe._run_train_optim_step = MagicMock(side_effect=RuntimeError("step failed"))
    recipe._make_progress_bar = lambda: None

    with pytest.raises(RuntimeError, match="step failed"):
        recipe.run_train_validation_loop()

    assert events == ["close"]
    assert recipe.partial_cuda_graph_manager is None
    assert recipe._partial_cuda_graph_capture_pending is False


def test_successful_capture_rezeros_grads_without_replacing_buffers():
    events: list[object] = []
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(capture=lambda: events.append("capture"))
    recipe._partial_cuda_graph_capture_pending = True
    recipe.optimizer = [_RecordingOptimizer(events)]

    recipe._capture_partial_cuda_graphs_after_eager_step()

    assert events == [("zero-grad", False), "capture", ("zero-grad", False)]
    assert recipe._partial_cuda_graph_capture_pending is False


def test_failed_capture_remains_pending_for_fail_closed_shutdown():
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(capture=MagicMock(side_effect=RuntimeError("capture failed")))
    recipe._partial_cuda_graph_capture_pending = True
    recipe._zero_optimizer_gradients = MagicMock()

    with pytest.raises(RuntimeError, match="capture failed"):
        recipe._capture_partial_cuda_graphs_after_eager_step()

    assert recipe._partial_cuda_graph_capture_pending is True
    recipe._zero_optimizer_gradients.assert_called_once_with()
