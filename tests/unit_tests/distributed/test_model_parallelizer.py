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

"""Tests for the model-owned parallelization contract."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
    MeshContext,
    ModelParallelizer,
)
from nemo_automodel.components.distributed.model_parallelizer import (
    _parallelize_moe,
    get_model_parallelizer,
    parallelize_model,
)


class _Sidecar:
    def __init__(self) -> None:
        self.context = None

    def parallelize(self, model, context, /):
        self.context = context
        model.was_parallelized = True
        return model


def _context() -> MeshContext:
    return MeshContext(strategy_config=FSDP2Config())


def test_contract_is_runtime_lightweight():
    assert hasattr(ModelParallelizer, "parallelize")
    assert not hasattr(ModelParallelizer(), "strategy")
    assert {
        "device_mesh",
        "moe_mesh",
        "strategy_config",
        "moe_parallel_config",
        "activation_checkpointing",
        "reapply_trainability",
    } <= MeshContext.__dataclass_fields__.keys()


def test_model_class_supplies_sidecar():
    sidecar = _Sidecar()

    class Model(nn.Module):
        parallelizer = sidecar

    model = Model()
    assert parallelize_model(model, _context()) is model
    assert model.was_parallelized is True
    assert sidecar.context is not None


def test_model_inherits_sidecar_from_base_class():
    sidecar = _Sidecar()

    class Base(nn.Module):
        parallelizer = sidecar

    class Model(Base):
        pass

    assert get_model_parallelizer(Model()) is sidecar


def test_missing_sidecar_uses_default():
    assert isinstance(get_model_parallelizer(nn.Linear(2, 2)), ModelParallelizer)


def test_invalid_sidecar_fails_at_setup():
    class Model(nn.Module):
        parallelizer = object()

    with pytest.raises(TypeError, match="must implement parallelize"):
        get_model_parallelizer(Model())


def test_legacy_manager_classes_are_deprecated(monkeypatch):
    from nemo_automodel.components.distributed import ddp, fsdp2, megatron_fsdp

    monkeypatch.setattr(ddp.DDPManager, "_setup_distributed", lambda self: None)

    with pytest.warns(DeprecationWarning, match="DDPManager is deprecated and will be removed in 0.8"):
        ddp.DDPManager(DDPConfig())
    with pytest.warns(DeprecationWarning, match="FSDP2Manager is deprecated and will be removed in 0.8"):
        fsdp2.FSDP2Manager(FSDP2Config(), device_mesh=Mock())
    with pytest.warns(DeprecationWarning, match="MegatronFSDPManager is deprecated and will be removed in 0.8"):
        megatron_fsdp.MegatronFSDPManager(MegatronFSDPConfig(), device_mesh=Mock())


def test_fsdp2_dispatch_receives_model_parallelizer(monkeypatch):
    parallelizer = ModelParallelizer()
    model = nn.Linear(2, 2)
    sentinel = nn.Linear(2, 2)
    context = _context()
    call = Mock(return_value=sentinel)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_fsdp2", call)

    assert parallelizer.parallelize(model, context) is sentinel
    call.assert_called_once_with(model, context, parallelizer=parallelizer)


def test_specialized_sidecar_routes_ep_through_unified_moe_executor(monkeypatch):
    parallelizer = ModelParallelizer()
    model = nn.Linear(2, 2)
    context = SimpleNamespace(cp_size=1, ep_size=2, strategy_config=FSDP2Config())
    call = Mock(return_value=model)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_moe", call)

    assert parallelizer.parallelize(model, context) is model
    call.assert_called_once_with(model, context, parallelizer=parallelizer)


def test_moe_executor_uses_generic_sharding_by_default(monkeypatch):
    parallelizer = ModelParallelizer()
    model = nn.Linear(2, 2)
    context = SimpleNamespace(
        ep_size=2,
        device_mesh=object(),
        moe_mesh=object(),
        strategy_config=FSDP2Config(),
        moe_parallel_config=None,
        activation_checkpointing=False,
        reapply_trainability=None,
        parallelize_axis_kwargs=lambda: {},
    )
    executor = Mock()
    monkeypatch.setattr("nemo_automodel.components.moe.parallelizer.parallelize_model", executor)

    assert _parallelize_moe(model, context, parallelizer=parallelizer) is model
    assert executor.call_args.kwargs["model_parallelizer"] is None


def test_moe_executor_receives_opted_in_model_parallelizer(monkeypatch):
    class MoEModelParallelizer(ModelParallelizer):
        _customizes_moe_fsdp = True

    parallelizer = MoEModelParallelizer()
    model = nn.Linear(2, 2)
    context = SimpleNamespace(
        ep_size=2,
        device_mesh=object(),
        moe_mesh=object(),
        strategy_config=FSDP2Config(),
        moe_parallel_config=None,
        activation_checkpointing=False,
        reapply_trainability=None,
        parallelize_axis_kwargs=lambda: {},
    )
    executor = Mock()
    monkeypatch.setattr("nemo_automodel.components.moe.parallelizer.parallelize_model", executor)

    assert _parallelize_moe(model, context, parallelizer=parallelizer) is model
    assert executor.call_args.kwargs["model_parallelizer"] is parallelizer


@pytest.mark.parametrize(
    ("strategy", "executor_name"),
    [
        (DDPConfig(), "_parallelize_ddp"),
        (MegatronFSDPConfig(), "_parallelize_megatron_fsdp"),
    ],
)
def test_model_parallelizer_dispatches_non_fsdp2_strategies(monkeypatch, strategy, executor_name):
    model = nn.Linear(2, 2)
    context = MeshContext(strategy_config=strategy)
    executor = Mock(return_value=model)
    monkeypatch.setattr(f"nemo_automodel.components.distributed.model_parallelizer.{executor_name}", executor)

    assert ModelParallelizer().parallelize(model, context) is model
    executor.assert_called_once_with(model, context)


@pytest.fixture(params=["default", "qwen35", "nemotron"])
def mtp_parallelizer(request):
    from nemo_automodel.components.models.nemotron_v3.parallelization import NemotronHModelParallelizer
    from nemo_automodel.components.models.qwen3_5.parallelization import Qwen3_5ModelParallelizer

    return {
        "default": ModelParallelizer,
        "qwen35": Qwen3_5ModelParallelizer,
        "nemotron": NemotronHModelParallelizer,
    }[request.param]()


def _mtp_context(*, pp_size=2, cp_size=2, ep_size=1):
    mesh = MagicMock(spec=DeviceMesh)
    mesh.mesh_dim_names = ("pp", "cp", "tp")
    axes = {name: Mock() for name in mesh.mesh_dim_names}
    for name, size in (("pp", pp_size), ("cp", cp_size), ("tp", 1)):
        axes[name].size.return_value = size
    mesh.__getitem__.side_effect = axes.__getitem__
    moe_mesh = MagicMock(spec=DeviceMesh)
    moe_mesh.mesh_dim_names = ("ep",)
    moe_mesh.__getitem__.return_value.size.return_value = ep_size
    return MeshContext(device_mesh=mesh, moe_mesh=moe_mesh, strategy_config=FSDP2Config())


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize(
    "pp_size, supports_mtp_cp, trimmed_stage, error, message",
    [
        (2, True, False, NotImplementedError, "use PP size 1 or CP size 1"),
        (2, False, False, NotImplementedError, "use PP size 1 or CP size 1"),
        (1, True, True, NotImplementedError, "use PP size 1 or CP size 1"),
        (1, False, False, RuntimeError, "does not support MTP with context parallelism"),
    ],
)
def test_mtp_rejected_before_strategy_dispatch(
    monkeypatch, mtp_parallelizer, ep_size, pp_size, supports_mtp_cp, trimmed_stage, error, message
):
    """Both EP routes reject before any TP, CP, or FSDP surgery can begin."""

    class Model(nn.Module):
        parallelizer = mtp_parallelizer

    model = Model()
    model.supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=supports_mtp_cp, supports_mtp_cp_pp=False)
    if trimmed_stage:
        model._is_pipeline_parallel_stage = lambda: True
    context = _mtp_context(pp_size=pp_size, ep_size=ep_size)
    fsdp = Mock(side_effect=AssertionError("FSDP2 surgery reached"))
    moe = Mock(side_effect=AssertionError("MoE surgery reached"))
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_fsdp2", fsdp)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_moe", moe)

    with pytest.raises(error, match=message):
        parallelize_model(model, context)

    fsdp.assert_not_called()
    moe.assert_not_called()
    assert not hasattr(model, "cp_mesh")


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize(
    "pp_size, cp_size, mtp_enabled, supports_mtp_cp, supports_mtp_cp_pp",
    [
        (1, 2, True, True, False),
        (2, 1, True, False, False),
        (2, 2, False, False, False),
        (2, 2, True, True, True),
    ],
)
def test_mtp_supported_configurations_reach_strategy_dispatch(
    monkeypatch, mtp_parallelizer, ep_size, pp_size, cp_size, mtp_enabled, supports_mtp_cp, supports_mtp_cp_pp
):
    class Model(nn.Module):
        parallelizer = mtp_parallelizer

    model = Model()
    model.supports = SimpleNamespace(
        mtp_enabled=mtp_enabled, supports_mtp_cp=supports_mtp_cp, supports_mtp_cp_pp=supports_mtp_cp_pp
    )
    context = _mtp_context(pp_size=pp_size, cp_size=cp_size, ep_size=ep_size)
    fsdp = Mock(return_value=model)
    moe = Mock(return_value=model)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_fsdp2", fsdp)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_moe", moe)

    assert parallelize_model(model, context) is model
    selected, unused = (moe, fsdp) if ep_size > 1 else (fsdp, moe)
    selected.assert_called_once_with(model, context, parallelizer=mtp_parallelizer)
    unused.assert_not_called()


def test_cp_dispatch_accepts_modules_without_transformers_capabilities(monkeypatch):
    """Diffusers components reach this public API without model.supports."""
    model = nn.Linear(2, 2)
    context = _mtp_context(pp_size=1)
    fsdp = Mock(return_value=model)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_fsdp2", fsdp)

    assert parallelize_model(model, context) is model
    fsdp.assert_called_once_with(model, context, parallelizer=get_model_parallelizer(model))
    assert not hasattr(model, "supports")
