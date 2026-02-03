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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.distributed import megatron_fsdp as mfsdp


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


def test_setup_distributed_raises_when_dist_not_available(monkeypatch):
    fake_dist = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)

    with pytest.raises(RuntimeError, match="torch.distributed not available"):
        mfsdp.MegatronFSDPManager(world_size=1, backend="gloo")


def test_setup_distributed_raises_when_dist_not_initialized(monkeypatch):
    fake_dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)

    with pytest.raises(RuntimeError, match="expected torch.distributed to be initialized"):
        mfsdp.MegatronFSDPManager(world_size=1, backend="gloo")


def test_setup_distributed_defaults_tp_cp_to_one_and_uses_cpu_mesh_when_backend_not_nccl(monkeypatch):
    fake_dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True)
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)

    mesh = MagicMock()
    init_device_mesh_mock = MagicMock(return_value=mesh)
    monkeypatch.setattr(mfsdp, "init_device_mesh", init_device_mesh_mock, raising=True)

    mgr = mfsdp.MegatronFSDPManager(tp_size=0, cp_size=0, dp_size=None, world_size=4, backend="gloo")

    assert mgr.tp_size == 1
    assert mgr.cp_size == 1
    assert mgr.dp_size == 4
    assert mgr.device_mesh is mesh

    init_device_mesh_mock.assert_called_once()
    call_kwargs = init_device_mesh_mock.call_args.kwargs
    assert call_kwargs["device_type"] == "cpu"
    assert call_kwargs["mesh_shape"] == (4, 1, 1)
    assert call_kwargs["mesh_dim_names"] == ("dp", "cp", "tp")


def test_setup_distributed_infers_dp_size_and_flattens_dp_cp_when_cp_gt_one(monkeypatch):
    fake_dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True)
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)

    mesh = MagicMock()
    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 2
    dp_cp_mesh = MagicMock()

    mesh.__getitem__.side_effect = lambda key: {
        "tp": tp_mesh,
        ("dp", "cp"): dp_cp_mesh,
    }[key]

    init_device_mesh_mock = MagicMock(return_value=mesh)
    monkeypatch.setattr(mfsdp, "init_device_mesh", init_device_mesh_mock, raising=True)

    mgr = mfsdp.MegatronFSDPManager(dp_size=None, tp_size=2, cp_size=2, world_size=8, backend="nccl")

    # inferred dp_size so that dp * cp * tp == world_size
    assert mgr.dp_size == 2
    assert mgr.device_mesh is mesh

    # backend="nccl" selects cuda mesh
    init_device_mesh_mock.assert_called_once()
    call_kwargs = init_device_mesh_mock.call_args.kwargs
    assert call_kwargs["device_type"] == "cuda"
    assert call_kwargs["mesh_shape"] == (2, 2, 2)

    # cp_size > 1 triggers dp+cp flattening
    dp_cp_mesh._flatten.assert_called_once_with(mesh_dim_name="dp_cp")


def test_setup_distributed_raises_when_world_size_not_divisible_by_tp_times_cp(monkeypatch):
    fake_dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True)
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)

    with pytest.raises(ValueError, match="must be divisible by \\(tp_size \\* cp_size\\)"):
        mfsdp.MegatronFSDPManager(dp_size=None, tp_size=3, cp_size=2, world_size=8, backend="gloo")


def test_parallelize_world_size_one_moves_to_cuda_bf16_and_enables_checkpointing_when_supported(monkeypatch):
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 1,
    )
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)
    monkeypatch.setattr(mfsdp, "init_device_mesh", MagicMock(return_value=MagicMock()), raising=True)

    mgr = mfsdp.MegatronFSDPManager(world_size=1, backend="gloo", activation_checkpointing=True)
    model = _FakeModel(supports_gradient_checkpointing=True)
    optimizer = MagicMock()

    out_model, out_opt = mgr.parallelize(model, optimizer=optimizer)
    assert out_model is model
    assert out_opt is optimizer

    # `.to("cuda").to(torch.bfloat16)` chain should be attempted even in CPU-only tests
    assert [args for (args, _kwargs) in model.to_calls] == [("cuda",), (torch.bfloat16,)]
    model.gradient_checkpointing_enable.assert_called_once_with()
    assert model.gradient_checkpointing_enabled is True


def test_parallelize_world_size_one_logs_error_when_checkpointing_not_supported(monkeypatch, caplog):
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 1,
    )
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)
    monkeypatch.setattr(mfsdp, "init_device_mesh", MagicMock(return_value=MagicMock()), raising=True)

    mgr = mfsdp.MegatronFSDPManager(world_size=1, backend="gloo", activation_checkpointing=True)
    model = _FakeModel(supports_gradient_checkpointing=False)

    caplog.set_level(logging.ERROR)
    mgr.parallelize(model, optimizer=None)
    assert "Model does not support gradient checkpointing. Skipping." in caplog.text


def test_parallelize_world_size_gt_one_selects_tp_plan_passes_dims_and_warns_on_nonzero3(monkeypatch, capsys, caplog):
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 8,
    )
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)

    # Device mesh used by manager.parallelize
    mesh = MagicMock()
    mesh.get_rank.return_value = 0
    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 2
    dp_cp_mesh = MagicMock()
    mesh.__getitem__.side_effect = lambda key: {
        "tp": tp_mesh,
        ("dp", "cp"): dp_cp_mesh,
    }[key]
    monkeypatch.setattr(mfsdp, "init_device_mesh", MagicMock(return_value=mesh), raising=True)

    # Plan selection and strategy call should be delegated
    tp_plan = {"some.layer": object()}
    get_plan_mock = MagicMock(return_value=tp_plan)
    strat_mock = MagicMock(return_value=("parallel_model", "parallel_opt"))
    monkeypatch.setattr(mfsdp, "_get_parallel_plan", get_plan_mock, raising=True)
    monkeypatch.setattr(mfsdp, "megatron_fsdp_strategy_parallelize", strat_mock, raising=True)

    mgr = mfsdp.MegatronFSDPManager(
        dp_size=None,
        tp_size=2,
        cp_size=2,
        world_size=8,
        backend="gloo",
        activation_checkpointing=True,  # should log error but continue
        zero_dp_strategy=2,  # triggers warning print on rank 0
    )

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
    assert plan_kwargs["use_hf_tp_plan"] is mgr.use_hf_tp_plan

    # Strategy should receive computed mesh dim names
    strat_mock.assert_called_once()
    strat_kwargs = strat_mock.call_args.kwargs
    assert strat_kwargs["device_mesh"] is mesh
    assert strat_kwargs["tp_shard_plan"] == tp_plan
    assert strat_kwargs["dp_shard_dim"] == "dp_cp"
    assert strat_kwargs["tp_dim"] == "tp"


def test_parallelize_world_size_gt_one_skips_tp_plan_when_tp_size_is_one(monkeypatch, capsys):
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 2,
    )
    monkeypatch.setattr(mfsdp, "dist", fake_dist, raising=True)

    mesh = MagicMock()
    mesh.get_rank.return_value = 0
    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 1
    mesh.__getitem__.side_effect = lambda key: {"tp": tp_mesh}[key]
    monkeypatch.setattr(mfsdp, "init_device_mesh", MagicMock(return_value=mesh), raising=True)

    get_plan_mock = MagicMock()
    strat_mock = MagicMock(return_value=("m", "o"))
    monkeypatch.setattr(mfsdp, "_get_parallel_plan", get_plan_mock, raising=True)
    monkeypatch.setattr(mfsdp, "megatron_fsdp_strategy_parallelize", strat_mock, raising=True)

    mgr = mfsdp.MegatronFSDPManager(dp_size=2, tp_size=1, cp_size=1, world_size=2, backend="gloo")
    out_model, out_opt = mgr.parallelize(model=object(), optimizer=object())
    assert (out_model, out_opt) == ("m", "o")

    # No TP -> do not ask for a TP plan
    get_plan_mock.assert_not_called()

    # dp_shard_dim should be "dp" when cp_size == 1
    strat_kwargs = strat_mock.call_args.kwargs
    assert strat_kwargs["tp_shard_plan"] is None
    assert strat_kwargs["dp_shard_dim"] == "dp"

    # zero_dp_strategy default is 3 -> no warning print
    assert capsys.readouterr().out == ""

