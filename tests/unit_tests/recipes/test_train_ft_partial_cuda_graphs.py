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

import random
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

import nemo_automodel.components.moe.paged_stash as paged_stash
import nemo_automodel.recipes.llm.partial_cuda_graphs as partial_graphs
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def _bare_recipe() -> TrainFinetuneRecipeForNextTokenPrediction:
    return TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)


def test_setup_forwards_runtime_safety_context(monkeypatch):
    recipe = _bare_recipe()
    model_parts = [nn.Linear(2, 2)]
    manager = SimpleNamespace(capture=MagicMock())
    recipe.model_parts = model_parts
    recipe.activation_checkpointing = True
    recipe.pp_enabled = False
    discover = MagicMock(return_value=manager)
    monkeypatch.setattr(partial_graphs.PartialCudaGraphManager, "from_model_parts", discover)

    recipe._setup_partial_cuda_graphs()

    discover.assert_called_once_with(
        model_parts,
        activation_checkpointing=True,
        activation_checkpointing_modules=None,
        activation_checkpointing_scope=("all",),
        pipeline_parallel=False,
    )
    assert recipe.partial_cuda_graph_manager is manager
    assert recipe._partial_cuda_graph_capture_pending is True


def test_setup_is_inert_when_no_scope_is_enabled(monkeypatch):
    recipe = _bare_recipe()
    recipe.model_parts = [nn.Linear(2, 2)]
    recipe.activation_checkpointing = False
    recipe.pp_enabled = False
    monkeypatch.setattr(partial_graphs.PartialCudaGraphManager, "from_model_parts", MagicMock(return_value=None))

    recipe._setup_partial_cuda_graphs()

    assert recipe.partial_cuda_graph_manager is None
    assert recipe._partial_cuda_graph_capture_pending is False


def test_setup_forwards_attention_checkpoint_boundary(monkeypatch):
    recipe = _bare_recipe()
    model_parts = [nn.Linear(2, 2)]
    recipe.model_parts = model_parts
    recipe.activation_checkpointing = True
    recipe.moe_parallel_config = SimpleNamespace(activation_checkpointing_modules=("attn",))
    recipe.distributed_config = SimpleNamespace(activation_checkpointing_scope=("language",))
    recipe.pp_enabled = False
    manager = SimpleNamespace(capture=MagicMock())
    discover = MagicMock(return_value=manager)
    monkeypatch.setattr(partial_graphs.PartialCudaGraphManager, "from_model_parts", discover)

    recipe._setup_partial_cuda_graphs()

    discover.assert_called_once_with(
        model_parts,
        activation_checkpointing=True,
        activation_checkpointing_modules=("attn",),
        activation_checkpointing_scope=("language",),
        pipeline_parallel=False,
    )
    assert recipe.partial_cuda_graph_manager is manager


def test_capture_runs_once_and_changes_no_optimizer_contract():
    manager = SimpleNamespace(capture=MagicMock())
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = manager
    recipe._partial_cuda_graph_capture_pending = True

    recipe._capture_partial_cuda_graphs_after_eager_step()
    recipe._capture_partial_cuda_graphs_after_eager_step()

    manager.capture.assert_called_once_with()
    assert recipe._partial_cuda_graph_capture_pending is False


def test_failed_capture_remains_pending_for_fail_closed_shutdown():
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(capture=MagicMock(side_effect=RuntimeError("capture failed")))
    recipe._partial_cuda_graph_capture_pending = True

    with pytest.raises(RuntimeError, match="capture failed"):
        recipe._capture_partial_cuda_graphs_after_eager_step()

    assert recipe._partial_cuda_graph_capture_pending is True


def test_paged_stash_is_prepared_before_partial_capture(monkeypatch):
    events = []
    stash = SimpleNamespace(
        prepare=lambda: events.append("prepare"),
        check_overflow=lambda: torch.zeros(1, dtype=torch.int64),
        check_host_spill=lambda: torch.zeros(1, dtype=torch.int64),
        check_allocator_imbalance=lambda: torch.zeros(1, dtype=torch.int64),
        clear_host_spill=lambda: None,
        finish_iteration=lambda: events.append("finish"),
        diagnostics=lambda: {},
    )
    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: stash)
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(capture=lambda: events.append("capture"))
    recipe._partial_cuda_graph_capture_pending = True
    recipe._partial_cuda_graph_paged_stash_enabled = True

    recipe._capture_partial_cuda_graphs_after_eager_step()

    assert events == ["prepare", "capture", "finish"]


def test_capture_overflow_is_reduced_before_all_ranks_close_graphs(monkeypatch):
    events = []
    stash = SimpleNamespace(
        prepare=lambda: events.append("prepare"),
        check_overflow=lambda: torch.zeros(1, dtype=torch.int64),
        check_host_spill=lambda: torch.zeros(1, dtype=torch.int64),
        check_allocator_imbalance=lambda: torch.zeros(1, dtype=torch.int64),
        clear_host_spill=lambda: None,
        finish_iteration=lambda: events.append("finish"),
        diagnostics=lambda: {},
        close=lambda: events.append("stash-close"),
    )

    def all_reduce(rank_flags, *, op):
        assert op is torch.distributed.ReduceOp.SUM
        events.append("all-reduce")
        rank_flags.copy_(torch.tensor([2, 0, 0], dtype=torch.int32))

    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: stash)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(
        capture=lambda: events.append("capture"),
        close=lambda: events.append("graph-close"),
    )
    recipe._partial_cuda_graph_capture_pending = True
    recipe._partial_cuda_graph_paged_stash_enabled = True

    with pytest.raises(RuntimeError, match="capture on 2 ranks"):
        recipe._capture_partial_cuda_graphs_after_eager_step()

    assert events == ["prepare", "capture", "all-reduce", "graph-close", "stash-close"]
    assert recipe.partial_cuda_graph_manager is None
    assert recipe._partial_cuda_graph_paged_stash_enabled is False


def test_paged_stash_host_spill_is_consumed_without_disabling_graphs(monkeypatch):
    clear_host_spill = MagicMock()
    finish_iteration = MagicMock()
    stash = SimpleNamespace(
        is_active=True,
        check_overflow=lambda: torch.zeros(1, dtype=torch.int64),
        check_host_spill=lambda: torch.ones(1, dtype=torch.int64),
        check_allocator_imbalance=lambda: torch.zeros(1, dtype=torch.int64),
        clear_host_spill=clear_host_spill,
        finish_iteration=finish_iteration,
    )
    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: stash)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    recipe = _bare_recipe()
    recipe._partial_cuda_graph_paged_stash_enabled = True

    loss_buffer = [torch.tensor(1.0)]
    result = recipe._rerun_after_paged_stash_overflow(
        ["same-batch"],
        num_label_tokens=7,
        loss_buffer=loss_buffer,
        rng_state=None,
    )

    assert result is loss_buffer
    assert recipe._partial_cuda_graph_logged_host_spill is True
    clear_host_spill.assert_called_once_with()
    finish_iteration.assert_called_once_with()


def test_paged_stash_allocator_imbalance_disables_graphs(monkeypatch):
    events = []
    stash = SimpleNamespace(
        is_active=True,
        check_overflow=lambda: torch.zeros(1, dtype=torch.int64),
        check_host_spill=lambda: torch.zeros(1, dtype=torch.int64),
        check_allocator_imbalance=lambda: torch.ones(1, dtype=torch.int64),
        clear_host_spill=lambda: events.append("clear-host-spill"),
        close=lambda: events.append("stash-close"),
    )
    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: stash)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(close=lambda: events.append("graph-close"))
    recipe._partial_cuda_graph_capture_pending = False
    recipe._partial_cuda_graph_paged_stash_enabled = True

    with pytest.raises(RuntimeError, match="did not recover every CUDA/host page"):
        recipe._rerun_after_paged_stash_overflow(
            ["same-batch"],
            num_label_tokens=7,
            loss_buffer=[torch.tensor(1.0)],
            rng_state=None,
        )

    assert events == ["clear-host-spill", "graph-close", "stash-close"]
    assert recipe.partial_cuda_graph_manager is None


def test_paged_stash_overflow_closes_graph_discards_gradients_and_reruns(monkeypatch):
    events = []
    stash = SimpleNamespace(
        is_active=True,
        check_overflow=lambda: torch.ones(1, dtype=torch.int64),
        check_host_spill=lambda: torch.zeros(1, dtype=torch.int64),
        check_allocator_imbalance=lambda: torch.zeros(1, dtype=torch.int64),
        clear_host_spill=lambda: None,
        close=lambda: events.append("stash-close"),
    )
    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: stash)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(close=lambda: events.append("graph-close"))
    recipe._partial_cuda_graph_capture_pending = False
    recipe._partial_cuda_graph_paged_stash_enabled = True
    recipe._partial_cuda_graph_paged_stash_reruns = 0
    recipe.optimizer = [SimpleNamespace(zero_grad=lambda: events.append("zero-grad"))]
    recipe.rng = SimpleNamespace(load_state_dict=lambda state: events.append(("restore-rng", state)))
    recipe._run_forward_backward_batches = lambda batches, num_label_tokens: (
        events.append(("rerun", batches, num_label_tokens)) or [torch.tensor(2.0)]
    )

    result = recipe._rerun_after_paged_stash_overflow(
        ["same-batch"],
        num_label_tokens=7,
        loss_buffer=[torch.tensor(1.0)],
        rng_state="pre-attempt-state",
    )

    assert events == [
        "graph-close",
        "stash-close",
        "zero-grad",
        ("restore-rng", "pre-attempt-state"),
        ("rerun", ["same-batch"], 7),
    ]
    assert result[0].item() == 2.0
    assert recipe._partial_cuda_graph_paged_stash_reruns == 1
    assert recipe.partial_cuda_graph_manager is None


def test_active_paged_stash_snapshots_recipe_rng(monkeypatch):
    state = object()
    monkeypatch.setattr(
        paged_stash,
        "get_paged_stash_manager",
        lambda: SimpleNamespace(is_active=True),
    )
    recipe = _bare_recipe()
    recipe._partial_cuda_graph_paged_stash_enabled = True
    recipe.rng = SimpleNamespace(state_dict=lambda: state)

    assert recipe._snapshot_paged_stash_rng_state() is state


def test_overflow_retry_reproduces_python_numpy_and_torch_rng(monkeypatch):
    events = []
    stash = SimpleNamespace(
        is_active=True,
        check_overflow=lambda: torch.ones(1, dtype=torch.int64),
        check_host_spill=lambda: torch.zeros(1, dtype=torch.int64),
        check_allocator_imbalance=lambda: torch.zeros(1, dtype=torch.int64),
        clear_host_spill=lambda: None,
        close=lambda: None,
    )
    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: stash)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(close=lambda: None)
    recipe._partial_cuda_graph_capture_pending = False
    recipe._partial_cuda_graph_paged_stash_enabled = True
    recipe._partial_cuda_graph_paged_stash_reruns = 0
    recipe.optimizer = [SimpleNamespace(zero_grad=lambda: None)]
    recipe.rng = StatefulRNG(seed=1234)

    rng_state = recipe._snapshot_paged_stash_rng_state()
    expected_draws = (random.random(), np.random.rand(), torch.rand(3))
    random.random()
    np.random.rand()
    torch.rand(3)

    def rerun(_batches, *, num_label_tokens):
        assert num_label_tokens == 7
        events.append((random.random(), np.random.rand(), torch.rand(3)))
        return [torch.tensor(2.0)]

    recipe._run_forward_backward_batches = rerun
    recipe._rerun_after_paged_stash_overflow(
        ["same-batch"],
        num_label_tokens=7,
        loss_buffer=[torch.tensor(1.0)],
        rng_state=rng_state,
    )

    assert events[0][0] == expected_draws[0]
    assert events[0][1] == expected_draws[1]
    torch.testing.assert_close(events[0][2], expected_draws[2])


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
        ("train-step", ("step-0",)),
        "capture",
        ("train-step", ("step-1",)),
        "close",
    ]
    assert recipe.partial_cuda_graph_manager is None


def test_training_loop_closes_graphs_when_a_step_raises():
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


def test_validation_disables_partial_graphs_as_one_eager_region():
    events = []

    @contextmanager
    def eager_execution():
        events.append("enter-eager")
        try:
            yield
        finally:
            events.append("exit-eager")

    recipe = _bare_recipe()
    recipe.partial_cuda_graph_manager = SimpleNamespace(eager_execution=eager_execution)
    recipe.model_parts = [nn.Identity()]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), is_main=True)
    recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 1.0e-3}])]
    recipe.step_scheduler = SimpleNamespace(step=1, epoch=0)
    recipe.pp_enabled = False
    recipe._forward_backward_step = lambda _index, _batch, *, loss_buffer, **_kwargs: loss_buffer.append(
        torch.tensor(2.0)
    )
    recipe._dp_allreduce = lambda value, **_kwargs: value
    batch = {"labels": torch.tensor([[1, 2, -100]])}

    metrics = recipe._run_validation_epoch([batch])

    assert events == ["enter-eager", "exit-eager"]
    assert metrics.metrics["val_loss"] == 1.0
