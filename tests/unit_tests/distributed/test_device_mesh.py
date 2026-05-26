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

"""Tests for the high-level mesh context factory."""

import pytest

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config, MegatronFSDPConfig
from nemo_automodel.components.distributed.device_mesh import create_mesh_context
from nemo_automodel.components.distributed.mesh import MeshAxisName, MeshContext
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.moe.config import MoEParallelizerConfig


class _FakeAxis:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _FakeMesh:
    def __init__(self, sizes: dict[MeshAxisName, int]) -> None:
        self.mesh_dim_names = tuple(sizes)
        self._sizes = sizes

    def __getitem__(self, axis: MeshAxisName) -> _FakeAxis:
        return _FakeAxis(self._sizes[axis])


@pytest.fixture
def captured_raw_mesh_call(monkeypatch):
    captured: dict = {}

    def fake_create_device_meshes(strategy_config, **kwargs):
        captured["strategy_config"] = strategy_config
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.device_mesh._create_device_meshes",
        fake_create_device_meshes,
    )
    return captured


def test_create_mesh_context_accepts_ddp_strategy_name(captured_raw_mesh_call):
    ctx = create_mesh_context("ddp", world_size=4, activation_checkpointing=True, backend="gloo")

    assert isinstance(ctx, MeshContext)
    assert isinstance(ctx.strategy_config, DDPConfig)
    assert ctx.strategy_config.backend == "gloo"
    assert ctx.strategy_config.activation_checkpointing is True
    assert not hasattr(ctx, "activation_checkpointing")
    assert captured_raw_mesh_call["world_size"] == 4


def test_create_mesh_context_accepts_mfsdp_alias(captured_raw_mesh_call):
    ctx = create_mesh_context("mfsdp", world_size=4, backend="gloo")

    assert isinstance(ctx.strategy_config, MegatronFSDPConfig)
    assert ctx.strategy_config.backend == "gloo"
    assert captured_raw_mesh_call["strategy_config"] is ctx.strategy_config


def test_create_mesh_context_accepts_existing_config(captured_raw_mesh_call):
    config = FSDP2Config(backend="gloo")

    ctx = create_mesh_context(config, world_size=8)

    assert ctx.strategy_config is config
    assert captured_raw_mesh_call["strategy_config"] is config
    assert captured_raw_mesh_call["world_size"] == 8


def test_create_mesh_context_accepts_distributed_config_keyword(captured_raw_mesh_call):
    config = FSDP2Config(backend="gloo")

    ctx = create_mesh_context(distributed_config=config, world_size=8)

    assert ctx.strategy_config is config
    assert captured_raw_mesh_call["strategy_config"] is config


def test_create_mesh_context_passes_parallelism_to_raw_mesh_builder(captured_raw_mesh_call):
    create_mesh_context(
        "fsdp2",
        dp_size=4,
        dp_replicate_size=2,
        tp_size=2,
        cp_size=2,
        world_size=16,
    )

    assert captured_raw_mesh_call["dp_size"] == 4
    assert captured_raw_mesh_call["dp_replicate_size"] == 2
    assert captured_raw_mesh_call["tp_size"] == 2
    assert captured_raw_mesh_call["pp_size"] == 1
    assert captured_raw_mesh_call["cp_size"] == 2
    assert captured_raw_mesh_call["ep_size"] == 1


def test_create_mesh_context_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_mesh_context("unknown", world_size=1)


def test_create_mesh_context_rejects_unknown_strategy_kwarg(captured_raw_mesh_call):
    with pytest.raises(ValueError, match="Unknown options for strategy 'fsdp2'"):
        create_mesh_context("fsdp2", world_size=1, does_not_exist=True)


def test_create_mesh_context_rejects_strategy_kwargs_with_config_object(captured_raw_mesh_call):
    with pytest.raises(ValueError, match="keyword arguments require strategy to be a string"):
        create_mesh_context(FSDP2Config(), world_size=1, sequence_parallel=True)


def test_create_mesh_context_defaults_parallel_subconfigs(monkeypatch):
    device_mesh = _FakeMesh(
        {
            MeshAxisName.PP: 2,
            MeshAxisName.DP_REPLICATE: 1,
            MeshAxisName.DP_SHARD: 1,
            MeshAxisName.CP: 1,
            MeshAxisName.TP: 1,
        }
    )
    moe_mesh = _FakeMesh({MeshAxisName.EP_SHARD: 1, MeshAxisName.EP: 2})

    def fake_create_device_meshes(strategy_config, **kwargs):
        return device_mesh, moe_mesh

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.device_mesh._create_device_meshes",
        fake_create_device_meshes,
    )

    ctx = create_mesh_context("fsdp2", pp_size=2, ep_size=2, world_size=4)

    assert isinstance(ctx.pipeline_config, PipelineConfig)
    assert isinstance(ctx.moe_config, MoEParallelizerConfig)


def test_create_mesh_context_keeps_activation_checkpointing_on_strategy_config(monkeypatch):
    device_mesh = _FakeMesh(
        {
            MeshAxisName.PP: 1,
            MeshAxisName.DP_REPLICATE: 1,
            MeshAxisName.DP_SHARD: 2,
            MeshAxisName.CP: 1,
            MeshAxisName.TP: 1,
        }
    )
    moe_mesh = _FakeMesh({MeshAxisName.EP_SHARD: 1, MeshAxisName.EP: 2})

    def fake_create_device_meshes(strategy_config, **kwargs):
        return device_mesh, moe_mesh

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.device_mesh._create_device_meshes",
        fake_create_device_meshes,
    )

    ctx = create_mesh_context("fsdp2", ep_size=2, activation_checkpointing=True, world_size=2)

    assert not hasattr(ctx, "activation_checkpointing")
    assert ctx.strategy_config.activation_checkpointing is True
