# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the **component-layer** mesh module (MeshContext, validation, STRATEGY_MAP).

Dict-parsing tests live in ``tests/unit_tests/recipes/test_dist_setup.py``.
"""

import pytest

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config, MegatronFSDPConfig
from nemo_automodel.components.distributed.mesh import (
    STRATEGY_MAP,
    MeshAxisName,
    MeshContext,
    _validate_distributed_setup,
)
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.moe.config import MoEParallelizerConfig


# ---------------------------------------------------------------------------
# MeshContext – defaults (no mesh attached)
# ---------------------------------------------------------------------------


class TestMeshContextDefaults:
    def test_sizes_default_to_1_or_none(self):
        ctx = MeshContext()

        assert ctx.tp_size == 1
        assert ctx.pp_size == 1
        assert ctx.cp_size == 1
        assert ctx.ep_size == 1
        assert ctx.dp_size is None
        assert ctx.dp_replicate_size is None

    def test_pp_enabled_false_by_default(self):
        ctx = MeshContext()
        assert ctx.pp_enabled is False

    def test_default_config_fields(self):
        ctx = MeshContext()
        assert ctx.strategy_config is None
        assert ctx.pipeline_config is None
        assert ctx.moe_config is None
        assert ctx.activation_checkpointing is False
        assert ctx.device_mesh is None
        assert ctx.moe_mesh is None

    def test_with_strategy_config(self):
        cfg = FSDP2Config()
        ctx = MeshContext(strategy_config=cfg)
        assert ctx.strategy_config is cfg

    def test_with_pipeline_config(self):
        pc = PipelineConfig(pp_schedule="1f1b", pp_microbatch_size=4)
        ctx = MeshContext(pipeline_config=pc)
        assert ctx.pipeline_config is pc

    def test_with_moe_config(self):
        mc = MoEParallelizerConfig(ignore_router_for_ac=True)
        ctx = MeshContext(moe_config=mc)
        assert ctx.moe_config is mc

    def test_activation_checkpointing_flag(self):
        ctx = MeshContext(activation_checkpointing=True)
        assert ctx.activation_checkpointing is True


# ---------------------------------------------------------------------------
# MeshContext.from_meshes (no real mesh — smoke test)
# ---------------------------------------------------------------------------


class TestFromMeshes:
    def test_from_none_meshes(self):
        ctx = MeshContext.from_meshes(None)
        assert ctx.device_mesh is None
        assert ctx.moe_mesh is None
        assert ctx.tp_size == 1

    def test_from_meshes_with_strategy(self):
        cfg = FSDP2Config()
        ctx = MeshContext.from_meshes(None, strategy_config=cfg)
        assert ctx.strategy_config is cfg


# ---------------------------------------------------------------------------
# MeshContext – helper methods
# ---------------------------------------------------------------------------


class TestMeshAxisNameEnum:
    def test_enum_is_str(self):
        """MeshAxisName members compare equal to plain strings."""
        assert MeshAxisName.TP == "tp"
        assert MeshAxisName.PP == "pp"
        assert MeshAxisName.DP_SHARD_CP == "dp_shard_cp"
        assert isinstance(MeshAxisName.TP, str)

    def test_all_expected_members(self):
        names = {m.value for m in MeshAxisName}
        assert names == {
            "pp", "dp", "dp_replicate", "dp_shard", "dp_shard_cp",
            "dp_cp", "cp", "tp", "ep", "ep_shard",
        }


class TestHelperMethods:
    def test_pipeline_axis_kwargs(self):
        ctx = MeshContext()
        kwargs = ctx.pipeline_axis_kwargs()
        assert "pp_axis_name" in kwargs
        assert kwargs["pp_axis_name"] == MeshAxisName.PP
        assert kwargs["dp_axis_names"] == (MeshAxisName.DP_SHARD_CP,)

    def test_parallelize_axis_kwargs(self):
        ctx = MeshContext()
        kwargs = ctx.parallelize_axis_kwargs()
        assert "pp_axis_name" not in kwargs
        assert kwargs["dp_axis_names"] == (MeshAxisName.DP_SHARD_CP,)


# ---------------------------------------------------------------------------
# validate_distributed_setup – happy paths
# ---------------------------------------------------------------------------


class TestValidateHappyPaths:
    def test_minimal_fsdp2(self):
        _validate_distributed_setup(MeshContext(strategy_config=FSDP2Config()))

    def test_minimal_megatron_fsdp(self):
        _validate_distributed_setup(MeshContext(strategy_config=MegatronFSDPConfig()))

    def test_minimal_ddp(self):
        _validate_distributed_setup(MeshContext(strategy_config=DDPConfig()))

    def test_activation_checkpointing_on_strategy(self):
        _validate_distributed_setup(
            MeshContext(strategy_config=FSDP2Config(activation_checkpointing=True)),
        )


# ---------------------------------------------------------------------------
# validate_distributed_setup – constraint violations
# ---------------------------------------------------------------------------


class TestValidation:
    def test_megatron_fsdp_rejects_sequence_parallel(self):
        with pytest.raises(ValueError, match="sequence_parallel"):
            _validate_distributed_setup(
                MeshContext(strategy_config=MegatronFSDPConfig(sequence_parallel=True)),
            )

    def test_pipeline_requires_pp_gt_1(self):
        pc = PipelineConfig(pp_schedule="1f1b")
        with pytest.raises(ValueError, match="pp_size > 1"):
            _validate_distributed_setup(
                MeshContext(strategy_config=FSDP2Config(), pipeline_config=pc),
            )

    def test_moe_requires_ep_gt_1(self):
        mc = MoEParallelizerConfig()
        with pytest.raises(ValueError, match="ep_size > 1"):
            _validate_distributed_setup(
                MeshContext(strategy_config=FSDP2Config(), moe_config=mc),
            )


# ---------------------------------------------------------------------------
# STRATEGY_MAP
# ---------------------------------------------------------------------------


class TestStrategyMap:
    def test_strategy_map_entries(self):
        assert STRATEGY_MAP == {
            "fsdp2": FSDP2Config,
            "megatron_fsdp": MegatronFSDPConfig,
            "ddp": DDPConfig,
        }


# ---------------------------------------------------------------------------
# Integration: validate with full configs
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_megatron_fsdp_with_valid_options(self):
        _validate_distributed_setup(
            MeshContext(
                strategy_config=MegatronFSDPConfig(
                    zero_dp_strategy=2,
                    overlap_grad_reduce=False,
                    activation_checkpointing=True,
                ),
            ),
        )

    def test_fsdp2_validates_at_construction(self):
        """MeshContext.__post_init__ runs validation automatically."""
        ctx = MeshContext(
            strategy_config=FSDP2Config(
                sequence_parallel=True,
                activation_checkpointing=True,
                defer_fsdp_grad_sync=False,
            ),
        )
        # No meshes → sizes default to 1 / None, which is valid for FSDP2.
        assert ctx.tp_size == 1

    @pytest.mark.parametrize(
        "strategy_config",
        [FSDP2Config(backend="gloo"), MegatronFSDPConfig(backend="gloo"), DDPConfig(backend="gloo")],
        ids=["fsdp2", "megatron_fsdp", "ddp"],
    )
    def test_backend_configuration(self, strategy_config):
        _validate_distributed_setup(MeshContext(strategy_config=strategy_config))
