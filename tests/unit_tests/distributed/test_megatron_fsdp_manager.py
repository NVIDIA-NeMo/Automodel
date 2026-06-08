#!/usr/bin/env python3
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

import logging
from unittest.mock import MagicMock

import torch

from nemo_automodel.components.distributed import megatron_fsdp as mfsdp
from nemo_automodel.components.distributed.config import MegatronFSDPConfig


class _FakeModel:
    """Tiny stand-in with `.to()` chaining and optional checkpointing support."""

    def __init__(self, *, supports_gradient_checkpointing: bool):
        self.to_calls = []
        self.gradient_checkpointing_enabled = False
        if supports_gradient_checkpointing:
            self.gradient_checkpointing_enable = MagicMock(side_effect=self._enable_gc)

    def _enable_gc(self):
        self.gradient_checkpointing_enabled = True

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


def _make_manager(mesh, **config_kwargs):
    """Helper to create a MegatronFSDPManager with a mock mesh and config overrides."""
    config = MegatronFSDPConfig(**config_kwargs)
    return mfsdp.MegatronFSDPManager(config=config, device_mesh=mesh)


class _FakeMegatronFSDP:
    def __init__(self):
        self.model_auto_sync = False
        self.finish_grad_sync = MagicMock()
        self.install_optimized_model_weights = MagicMock()
        self.zero_grad_buffer = MagicMock()


class _FakeOptimizer:
    def __init__(self):
        self.step_calls = []
        self.zero_grad_calls = []

    def step(self, *args, **kwargs):
        self.step_calls.append((args, kwargs))

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls.append((args, kwargs))


def test_parallelize_world_size_one_moves_to_cuda_bf16_and_enables_checkpointing_when_supported(monkeypatch):
    monkeypatch.setattr(mfsdp, "dist", MagicMock(get_world_size=lambda: 1), raising=True)

    mesh = MagicMock()
    mgr = _make_manager(mesh, activation_checkpointing=True)
    model = _FakeModel(supports_gradient_checkpointing=True)
    optimizer = MagicMock()

    out_model, out_opt = mgr.parallelize(model, optimizer=optimizer)
    assert out_model is model
    assert out_opt is optimizer

    # `.to("cuda").to(torch.bfloat16)` chain
    assert [args for (args, _kwargs) in model.to_calls] == [("cuda",), (torch.bfloat16,)]
    model.gradient_checkpointing_enable.assert_called_once_with()
    assert model.gradient_checkpointing_enabled is True


def test_parallelize_world_size_one_logs_error_when_checkpointing_not_supported(monkeypatch, caplog):
    monkeypatch.setattr(mfsdp, "dist", MagicMock(get_world_size=lambda: 1), raising=True)

    mesh = MagicMock()
    mgr = _make_manager(mesh, activation_checkpointing=True)
    model = _FakeModel(supports_gradient_checkpointing=False)

    caplog.set_level(logging.ERROR)
    mgr.parallelize(model, optimizer=None)
    assert "Model does not support gradient checkpointing. Skipping." in caplog.text


def test_parallelize_world_size_gt_one_selects_tp_plan_passes_dims_and_warns_on_nonzero3(monkeypatch, capsys, caplog):
    monkeypatch.setattr(mfsdp, "dist", MagicMock(get_world_size=lambda: 8), raising=True)

    # Device mesh with tp > 1 and dp_cp flattened dim
    mesh = MagicMock()
    mesh.get_rank.return_value = 0
    mesh.mesh_dim_names = ("dp_cp", "tp")
    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 2
    mesh.__getitem__ = lambda self, key: tp_mesh if key == "tp" else MagicMock()

    # Plan selection and strategy call should be delegated
    tp_plan = {"some.layer": object()}
    get_plan_mock = MagicMock(return_value=tp_plan)
    strat_mock = MagicMock(return_value=("parallel_model", "parallel_opt"))
    monkeypatch.setattr(mfsdp, "_get_parallel_plan", get_plan_mock, raising=True)
    monkeypatch.setattr(mfsdp, "megatron_fsdp_strategy_parallelize", strat_mock, raising=True)

    mgr = _make_manager(mesh, activation_checkpointing=True, zero_dp_strategy=2)

    caplog.set_level(logging.ERROR)
    out_model, out_opt = mgr.parallelize(model=object(), optimizer="opt")
    assert (out_model, out_opt) == ("parallel_model", "parallel_opt")

    # Activation checkpointing is not supported here; should emit an error log.
    assert "Activation checkpointing is not yet supported with MegatronFSDP. Skipping." in caplog.text

    # zero_dp_strategy warning printed only on rank 0
    assert "Warning: MegatronFSDP zero_dp_strategy is not 3" in capsys.readouterr().out

    # TP plan should be selected when tp mesh size > 1
    get_plan_mock.assert_called_once()
    plan_args, plan_kwargs = get_plan_mock.call_args
    assert plan_args[0] is not None  # model object
    assert plan_kwargs["sequence_parallel"] is False
    assert plan_kwargs["tp_shard_plan"] is None

    # Strategy should receive computed mesh dim names
    strat_mock.assert_called_once()
    strat_kwargs = strat_mock.call_args.kwargs
    assert strat_kwargs["device_mesh"] is mesh
    assert strat_kwargs["tp_shard_plan"] == tp_plan
    assert strat_kwargs["dp_shard_dim"] == "dp_cp"
    assert strat_kwargs["tp_dim"] == "tp"


def test_parallelize_world_size_gt_one_skips_tp_plan_when_tp_size_is_one(monkeypatch, capsys):
    monkeypatch.setattr(mfsdp, "dist", MagicMock(get_world_size=lambda: 2), raising=True)

    mesh = MagicMock()
    mesh.get_rank.return_value = 0
    mesh.mesh_dim_names = ("dp", "cp", "tp")
    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 1
    mesh.__getitem__ = lambda self, key: tp_mesh if key == "tp" else MagicMock()

    get_plan_mock = MagicMock()
    strat_mock = MagicMock(return_value=("m", "o"))
    monkeypatch.setattr(mfsdp, "_get_parallel_plan", get_plan_mock, raising=True)
    monkeypatch.setattr(mfsdp, "megatron_fsdp_strategy_parallelize", strat_mock, raising=True)

    mgr = _make_manager(mesh)
    out_model, out_opt = mgr.parallelize(model=object(), optimizer=object())
    assert (out_model, out_opt) == ("m", "o")

    # No TP -> do not ask for a TP plan
    get_plan_mock.assert_not_called()

    # dp_shard_dim should be "dp" when dp_cp is not in mesh_dim_names
    strat_kwargs = strat_mock.call_args.kwargs
    assert strat_kwargs["tp_shard_plan"] is None
    assert strat_kwargs["dp_shard_dim"] == "dp"

    # zero_dp_strategy default is 3 -> no warning print
    assert capsys.readouterr().out == ""


def test_fully_shard_optimizer_wraps_megatron_fsdp_lifecycle(monkeypatch):
    monkeypatch.setattr(mfsdp, "MegatronFSDP", _FakeMegatronFSDP, raising=True)
    monkeypatch.setattr(mfsdp, "HAS_MEGATRON_FSDP", True, raising=True)

    model = _FakeMegatronFSDP()
    optimizer = _FakeOptimizer()

    patched = mfsdp.fully_shard_optimizer(model, optimizer)
    assert patched is optimizer
    assert mfsdp.fully_shard_optimizer(model, optimizer) is optimizer

    optimizer.step()
    model.finish_grad_sync.assert_called_once_with()
    assert len(optimizer.step_calls) == 1
    model.install_optimized_model_weights.assert_called_once_with()

    optimizer.zero_grad(set_to_none=True)
    assert optimizer.zero_grad_calls == [((), {"set_to_none": True})]
    model.zero_grad_buffer.assert_called_once_with()

    optimizer.step(sync_grad_before_optimizer_step=False, install_optimized_model_weights=False)
    model.finish_grad_sync.assert_called_once()
    assert len(optimizer.step_calls) == 2
    model.install_optimized_model_weights.assert_called_once()
