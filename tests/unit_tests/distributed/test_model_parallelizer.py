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
from unittest.mock import Mock

import pytest
import torch.nn as nn

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
    context = SimpleNamespace(ep_size=2, strategy_config=FSDP2Config())
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
