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
    ParallelizeContext,
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


def _context() -> ParallelizeContext:
    return ParallelizeContext(mesh=MeshContext(), strategy=FSDP2Config())


def test_contract_is_runtime_lightweight():
    assert hasattr(ModelParallelizer, "parallelize")
    assert ModelParallelizer.__dataclass_fields__.keys() == {"strategy", "moe_strategy"}
    assert ParallelizeContext.__dataclass_fields__.keys() == {
        "mesh",
        "strategy",
        "moe",
        "activation_checkpointing",
        "reapply_trainability",
    }


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


def test_specialized_fsdp2_sidecar_uses_strategy(monkeypatch):
    strategy = Mock()
    adapter = ModelParallelizer(strategy)
    model = nn.Linear(2, 2)
    sentinel = nn.Linear(2, 2)
    context = _context()
    call = Mock(return_value=sentinel)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_fsdp2", call)

    assert adapter.parallelize(model, context) is sentinel
    call.assert_called_once_with(model, context, strategy=strategy)


def test_specialized_sidecar_routes_ep_through_unified_moe_executor(monkeypatch):
    strategy = Mock()
    adapter = ModelParallelizer(strategy, strategy)
    model = nn.Linear(2, 2)
    context = ParallelizeContext(mesh=SimpleNamespace(ep_size=2), strategy=FSDP2Config())
    call = Mock(return_value=model)
    monkeypatch.setattr("nemo_automodel.components.distributed.model_parallelizer._parallelize_moe", call)

    assert adapter.parallelize(model, context) is model
    call.assert_called_once_with(model, context, strategy=strategy)


def test_moe_executor_receives_model_owned_strategy(monkeypatch):
    strategy = Mock()
    model = nn.Linear(2, 2)
    mesh = SimpleNamespace(
        ep_size=2,
        device_mesh=object(),
        moe_mesh=object(),
        parallelize_axis_kwargs=lambda: {},
    )
    context = ParallelizeContext(mesh=mesh, strategy=FSDP2Config())
    executor = Mock()
    monkeypatch.setattr("nemo_automodel.components.moe.parallelizer.parallelize_model", executor)

    assert _parallelize_moe(model, context, strategy=strategy) is model
    assert executor.call_args.kwargs["parallelization_strategy"] is strategy


@pytest.mark.parametrize(
    ("strategy", "executor_name"),
    [
        (DDPConfig(), "_parallelize_ddp"),
        (MegatronFSDPConfig(), "_parallelize_megatron_fsdp"),
    ],
)
def test_model_parallelizer_dispatches_non_fsdp2_strategies(monkeypatch, strategy, executor_name):
    model = nn.Linear(2, 2)
    context = ParallelizeContext(mesh=MeshContext(), strategy=strategy)
    executor = Mock(return_value=model)
    monkeypatch.setattr(f"nemo_automodel.components.distributed.model_parallelizer.{executor_name}", executor)

    assert ModelParallelizer().parallelize(model, context) is model
    executor.assert_called_once_with(model, context)
