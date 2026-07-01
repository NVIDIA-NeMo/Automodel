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

from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


class _GraphManager:
    def __init__(self, result, events):
        self.result = result
        self.events = events
        self.is_captured = True
        self.completed_warmups = 2
        self.capture_count = 1
        self.replay_count = 3

    def run(self, batches):
        self.events.append(("graph", batches))
        return self.result

    def reset(self):
        self.events.append("graph-reset")
        self.is_captured = False
        self.completed_warmups = 0


class _Dispatcher:
    def __init__(self, events):
        self.events = events
        self._comm_manager = SimpleNamespace(_cuda_graph_handles=[object()])

    def set_static_rank_budget(self, factor):
        self.events.append(("rank-budget", factor))

    def reset_over_budget(self):
        self.events.append("overflow-reset")


def test_full_iteration_setup_allows_checkpointing_but_disables_paged_stash(monkeypatch):
    stash_manager = SimpleNamespace(configure=lambda **kwargs: setattr(stash_manager, "configured", kwargs))
    monkeypatch.setattr(
        "nemo_automodel.components.moe.paged_stash.get_paged_stash_manager",
        lambda: stash_manager,
    )
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    model_part = torch.nn.Module()
    model_part.backend = SimpleNamespace(
        full_iteration_cuda_graph=True,
        full_iteration_cuda_graph_warmup_steps=2,
        full_iteration_cuda_graph_single_mempool=True,
        moe_expert_rank_capacity_factor=1.125,
        moe_paged_stash=True,
    )
    recipe.model_parts = [model_part]
    recipe.pp_enabled = False
    recipe._get_cp_group_size = lambda: 1
    recipe.activation_checkpointing = True
    recipe.moe_parallel_config = SimpleNamespace(enable_fsdp2_prefetch=False)
    recipe.cfg = {}
    recipe.optimizer = []

    recipe._setup_full_iteration_cuda_graphs()

    assert recipe.full_iteration_cuda_graph_manager is not None
    assert recipe._full_iteration_paged_stash_enabled is False
    assert stash_manager.configured == {"enabled": False}


def test_full_iteration_setup_arms_paged_stash_overlap_only_after_pp1_checks(monkeypatch):
    overlap_calls = []
    stash_manager = SimpleNamespace(
        configure_full_iteration_stream_overlap=lambda **kwargs: overlap_calls.append(kwargs)
    )
    monkeypatch.setattr(
        "nemo_automodel.components.moe.paged_stash.get_paged_stash_manager",
        lambda: stash_manager,
    )
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    model_part = torch.nn.Module()
    model_part.backend = SimpleNamespace(
        full_iteration_cuda_graph=True,
        full_iteration_cuda_graph_warmup_steps=2,
        full_iteration_cuda_graph_single_mempool=True,
        moe_expert_rank_capacity_factor=1.125,
        moe_paged_stash=True,
    )
    recipe.model_parts = [model_part]
    recipe.pp_enabled = False
    recipe._get_cp_group_size = lambda: 1
    recipe.activation_checkpointing = False
    recipe.moe_parallel_config = SimpleNamespace(enable_fsdp2_prefetch=False)
    recipe.cfg = {}
    recipe.optimizer = []

    recipe._setup_full_iteration_cuda_graphs()

    assert overlap_calls == [{"enabled": True}]
    assert recipe._full_iteration_paged_stash_enabled is True


def _recipe(manager, events):
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.full_iteration_cuda_graph_manager = manager
    recipe._full_iteration_num_label_tokens = None
    recipe._full_iteration_num_label_tokens_tensor = None
    recipe._full_iteration_label_normalizer_mode = None
    recipe._full_iteration_paged_stash_enabled = False
    recipe._full_iteration_dispatchers = ()
    recipe._full_iteration_backend = SimpleNamespace(moe_expert_rank_capacity_factor=1.125)
    recipe._full_iteration_overflow_reruns = 0
    recipe._maybe_prepare_full_iteration_paged_stash = lambda: events.append("stash-prepare")
    recipe._reset_full_iteration_dispatch_overflow = lambda: events.append("overflow-reset-all")
    recipe._release_full_iteration_backend_handles = lambda: events.append("release-handles")
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.loss_fn = SimpleNamespace(reduction="sum")
    return recipe


def test_guarded_full_iteration_returns_graph_result_without_eager_rerun():
    events = []
    graph_loss = [torch.tensor(1.25)]
    manager = _GraphManager(graph_loss, events)
    recipe = _recipe(manager, events)
    recipe._collect_full_iteration_overflow = lambda: (0, 0)
    recipe._run_forward_backward_batches = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("unexpected eager rerun")
    )
    batches = [{"labels": torch.ones(1, 2, dtype=torch.long)}]

    result = recipe._run_guarded_forward_backward(batches, num_label_tokens=2)

    assert result is graph_loss
    assert recipe._full_iteration_num_label_tokens == 2
    assert events == ["stash-prepare", "overflow-reset-all", ("graph", batches)]


def test_full_iteration_execution_error_is_not_masked_by_exceptional_cleanup(monkeypatch):
    events = []

    class BrokenGraphManager(_GraphManager):
        def run(self, batches):
            del batches
            raise ValueError("original graph failure")

        def reset(self):
            events.append("graph-reset-attempt")
            raise RuntimeError("secondary graph reset failure")

    broken_manager = BrokenGraphManager([], events)
    recipe = _recipe(broken_manager, events)
    recipe._full_iteration_paged_stash_enabled = True
    recipe._release_full_iteration_backend_handles = lambda: (_ for _ in ()).throw(
        RuntimeError("secondary handle cleanup failure")
    )
    stash_manager = SimpleNamespace(force_abort_after_error=lambda: events.append("unsafe-stash-release"))
    monkeypatch.setattr(
        "nemo_automodel.components.moe.paged_stash.get_paged_stash_manager",
        lambda: stash_manager,
    )

    with pytest.raises(ValueError, match="original graph failure"):
        recipe._run_guarded_forward_backward(
            [{"labels": torch.ones(1, 2, dtype=torch.long)}],
            num_label_tokens=2,
        )

    assert "graph-reset-attempt" in events
    assert "unsafe-stash-release" not in events
    assert recipe._full_iteration_abandoned_graph_manager is broken_manager
    assert recipe.full_iteration_cuda_graph_manager is None


def test_hybridep_overflow_discards_graph_gradients_and_reruns_same_batches_dropless(monkeypatch):
    events = []
    manager = _GraphManager([torch.tensor(-1.0)], events)
    recipe = _recipe(manager, events)
    dispatcher = _Dispatcher(events)
    recipe._full_iteration_dispatchers = (dispatcher,)
    recipe._collect_full_iteration_overflow = lambda: (4, 0)
    recipe._zero_optimizer_gradients = lambda: events.append("zero-grad")
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.gc.collect", lambda: events.append("gc-collect"))
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: events.append("empty-cache"))
    eager_loss = [torch.tensor(0.75)]
    batches = [{"labels": torch.ones(1, 2, dtype=torch.long)}]

    def eager_rerun(actual_batches, *, num_label_tokens):
        assert actual_batches is batches
        assert num_label_tokens == 2
        events.append("eager-rerun")
        return eager_loss

    recipe._run_forward_backward_batches = eager_rerun

    result = recipe._run_guarded_forward_backward(batches, num_label_tokens=2)

    assert result is eager_loss
    assert recipe._full_iteration_overflow_reruns == 1
    assert events == [
        "stash-prepare",
        "overflow-reset-all",
        ("graph", batches),
        "graph-reset",
        "release-handles",
        "zero-grad",
        "gc-collect",
        "empty-cache",
        ("rank-budget", None),
        "eager-rerun",
        ("rank-budget", 1.125),
        "overflow-reset-all",
    ]


def test_changed_loss_normalizer_updates_persistent_scalar_without_recapture():
    events = []
    manager = _GraphManager([torch.tensor(1.0)], events)
    recipe = _recipe(manager, events)
    recipe._collect_full_iteration_overflow = lambda: (0, 0)
    batches = [{"labels": torch.ones(1, 2, dtype=torch.long)}]

    recipe._run_guarded_forward_backward(batches, num_label_tokens=7)
    scalar = recipe._full_iteration_num_label_tokens_tensor
    assert scalar is not None
    scalar_data_ptr = scalar.data_ptr()

    recipe._run_guarded_forward_backward(batches, num_label_tokens=11)

    assert recipe._full_iteration_num_label_tokens == 11
    assert recipe._full_iteration_num_label_tokens_tensor is scalar
    assert scalar.data_ptr() == scalar_data_ptr
    assert scalar.item() == 11
    assert "graph-reset" not in events
    assert "release-handles" not in events


def test_device_loss_normalizer_scales_main_gradient_without_scaling_moe_aux_gradient():
    previous_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(2.0)
    try:
        for divisor in (3.0, 7.0):
            main = torch.tensor(5.0, requires_grad=True)
            aux = torch.tensor(1.0, requires_grad=True)
            routed = MoEAuxLossAutoScaler.apply(main, aux)
            normalizer = torch.tensor(divisor)
            loss = TrainFinetuneRecipeForNextTokenPrediction._normalize_full_iteration_loss(
                routed,
                normalizer,
            )
            loss.backward()

            torch.testing.assert_close(main.grad, torch.tensor(1.0 / divisor))
            torch.testing.assert_close(aux.grad, torch.tensor(2.0))
    finally:
        MoEAuxLossAutoScaler.main_loss_backward_scale = previous_scale


def test_device_loss_normalizer_rejects_non_sum_loss_before_graph_execution():
    events = []
    recipe = _recipe(_GraphManager([torch.tensor(1.0)], events), events)
    recipe.loss_fn.reduction = "mean"
    recipe._collect_full_iteration_overflow = lambda: (0, 0)

    with pytest.raises(RuntimeError, match="loss_fn.reduction='sum'"):
        recipe._run_guarded_forward_backward(
            [{"labels": torch.ones(1, 2, dtype=torch.long)}],
            num_label_tokens=2,
        )

    assert not any(isinstance(event, tuple) and event[0] == "graph" for event in events)


def test_delayed_qat_toggle_resets_full_iteration_graph_before_the_next_step():
    events = []
    manager = _GraphManager([torch.tensor(1.0)], events)
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.model_parts = [object()]
    recipe._qat_enable_after = 3
    recipe._qat_enable_fn = lambda model: events.append(("enable-qat", model))
    recipe.full_iteration_cuda_graph_manager = manager
    recipe._release_full_iteration_backend_handles = lambda: events.append("release-handles")

    recipe._enable_qat_if_delayed(2)
    assert events == []

    recipe._enable_qat_if_delayed(3)

    assert events == [
        ("enable-qat", recipe.model_parts[0]),
        "graph-reset",
        "release-handles",
    ]
    assert recipe._qat_enable_after is None


def test_full_iteration_diagnostics_are_disabled_for_an_unconfigured_recipe():
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)

    assert recipe._full_iteration_cuda_graph_diagnostics() == {
        "enabled": False,
        "captured": False,
        "completed_warmups": 0,
        "captures": 0,
        "replays": 0,
        "overflow_reruns": 0,
        "paged_stash_enabled": False,
    }


def test_full_iteration_diagnostics_report_paged_stash_stream_state(monkeypatch):
    events = []
    recipe = _recipe(_GraphManager([torch.tensor(1.0)], events), events)
    recipe._full_iteration_paged_stash_enabled = True
    stash_manager = SimpleNamespace(
        diagnostics=lambda: {
            "state": "active",
            "page_size": 64,
            "buffer_size_factor": 1.1,
            "buffer_tokens": {(torch.uint8, 16): 128},
            "live_groups": 0,
            "full_iteration_stream_overlap": True,
            "transfer_stream_status": "idle",
            "backward_schedule_depth": 0,
        }
    )
    monkeypatch.setattr(
        "nemo_automodel.components.moe.paged_stash.get_paged_stash_manager",
        lambda: stash_manager,
    )

    diagnostics = recipe._full_iteration_cuda_graph_diagnostics()

    assert diagnostics["paged_stash"] == {
        "state": "active",
        "page_size": 64,
        "buffer_size_factor": 1.1,
        "buffer_tokens": {"torch.uint8:16": 128},
        "live_groups": 0,
        "stream_overlap": True,
        "transfer_stream_status": "idle",
        "backward_schedule_depth": 0,
    }
