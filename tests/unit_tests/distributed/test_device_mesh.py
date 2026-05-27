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

"""Tests for the high-level mesh context builder."""

import pytest

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    DistributedSetup,
    FSDP2Config,
    MegatronFSDPConfig,
    MoEParallelizerConfig,
)
from nemo_automodel.components.distributed.mesh import MeshAxisName, MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig


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

    def fake_create_device_meshes(strategy_config, parallelism, **kwargs):
        captured["strategy_config"] = strategy_config
        captured["parallelism"] = parallelism
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.mesh_utils._create_device_meshes",
        fake_create_device_meshes,
    )
    return captured


def test_mesh_context_build_accepts_ddp_config(captured_raw_mesh_call):
    config = DDPConfig()

    ctx = MeshContext.build(config, world_size=4)

    assert isinstance(ctx, MeshContext)
    assert not hasattr(ctx, "strategy_config")
    assert captured_raw_mesh_call["strategy_config"] is config
    assert not hasattr(ctx, "activation_checkpointing")
    assert captured_raw_mesh_call["world_size"] == 4


@pytest.mark.parametrize("strategy", ["megatron_fsdp", "megatron-fsdp", "mfsdp"])
def test_distributed_setup_config_accepts_megatron_fsdp_names(strategy, captured_raw_mesh_call):
    setup = DistributedSetup.build(strategy=strategy, world_size=4)

    assert isinstance(setup.mesh_context, MeshContext)
    assert isinstance(captured_raw_mesh_call["strategy_config"], MegatronFSDPConfig)
    assert captured_raw_mesh_call["world_size"] == 4


def test_mesh_context_build_accepts_existing_config(captured_raw_mesh_call):
    config = FSDP2Config()

    ctx = MeshContext.build(config, world_size=8)

    assert isinstance(ctx, MeshContext)
    assert captured_raw_mesh_call["strategy_config"] is config
    assert captured_raw_mesh_call["world_size"] == 8


def test_mesh_context_build_passes_parallelism_to_raw_mesh_builder(captured_raw_mesh_call):
    MeshContext.build(
        FSDP2Config(),
        parallelism_sizes=ParallelismSizes(dp_size=4, dp_replicate_size=2, tp_size=2, cp_size=2),
        world_size=16,
    )

    parallelism = captured_raw_mesh_call["parallelism"]
    assert parallelism.dp_size == 4
    assert parallelism.dp_replicate_size == 2
    assert parallelism.tp_size == 2
    assert parallelism.pp_size == 1
    assert parallelism.cp_size == 2
    assert parallelism.ep_size == 1


def test_mesh_context_build_requires_strategy_config():
    with pytest.raises(ValueError, match="Unknown distributed strategy config type"):
        MeshContext.build("ddp", world_size=1)  # type: ignore[arg-type]


def test_distributed_setup_config_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        DistributedSetup.build(strategy="unknown", world_size=1)


def test_distributed_setup_config_defaults_parallel_subconfigs(monkeypatch):
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

    def fake_create_device_meshes(strategy_config, parallelism, **kwargs):
        return device_mesh, moe_mesh

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.mesh_utils._create_device_meshes",
        fake_create_device_meshes,
    )

    setup = DistributedSetup.build(
        strategy="fsdp2",
        parallelism_sizes=ParallelismSizes(pp_size=2, ep_size=2),
        world_size=4,
    )

    assert isinstance(setup, DistributedSetup)
    assert isinstance(setup.pipeline_config, PipelineConfig)
    assert isinstance(setup.moe_parallel_config, MoEParallelizerConfig)


def test_distributed_setup_config_keeps_activation_checkpointing_separate(monkeypatch):
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

    def fake_create_device_meshes(strategy_config, parallelism, **kwargs):
        return device_mesh, moe_mesh

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.mesh_utils._create_device_meshes",
        fake_create_device_meshes,
    )

    setup = DistributedSetup.build(
        strategy="fsdp2",
        parallelism_sizes=ParallelismSizes(ep_size=2),
        activation_checkpointing=True,
        world_size=2,
    )

    assert not hasattr(setup.mesh_context, "activation_checkpointing")
    assert setup.activation_checkpointing is True
    assert setup.strategy_config.activation_checkpointing is False


def test_distributed_setup_config_does_not_infer_activation_checkpointing_from_strategy_config(captured_raw_mesh_call):
    setup = DistributedSetup.build(
        strategy=FSDP2Config(activation_checkpointing=True),
        world_size=1,
    )

    assert setup.activation_checkpointing is False
    assert captured_raw_mesh_call["strategy_config"].activation_checkpointing is True


def test_distributed_setup_config_activation_checkpointing_override(captured_raw_mesh_call):
    setup = DistributedSetup.build(
        strategy=FSDP2Config(activation_checkpointing=True),
        activation_checkpointing=False,
        world_size=1,
    )

    assert setup.activation_checkpointing is False
    assert setup.strategy_config.activation_checkpointing is True


def test_distributed_setup_config_rejects_pipeline_config_without_pipeline_parallelism(captured_raw_mesh_call):
    with pytest.raises(ValueError, match="pipeline_config requires pp_size > 1"):
        DistributedSetup.build(
            strategy=FSDP2Config(),
            pipeline_config=PipelineConfig(),
            parallelism_sizes=ParallelismSizes(pp_size=1),
            world_size=1,
        )


def test_distributed_setup_config_rejects_moe_config_without_expert_parallelism(captured_raw_mesh_call):
    with pytest.raises(ValueError, match="moe_parallel_config requires ep_size > 1"):
        DistributedSetup.build(
            strategy=FSDP2Config(),
            moe_parallel_config=MoEParallelizerConfig(),
            parallelism_sizes=ParallelismSizes(ep_size=1),
            world_size=1,
        )


def test_distributed_setup_config_builds_runtime_setup(captured_raw_mesh_call):
    setup = DistributedSetup.build(
        strategy=FSDP2Config(sequence_parallel=True),
        parallelism_sizes=ParallelismSizes(tp_size=2),
        activation_checkpointing=True,
        world_size=4,
    )

    assert isinstance(setup, DistributedSetup)
    assert setup.strategy_config.sequence_parallel is True
    assert setup.activation_checkpointing is True
    assert captured_raw_mesh_call["parallelism"].tp_size == 2
    assert captured_raw_mesh_call["world_size"] == 4
