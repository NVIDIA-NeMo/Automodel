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

"""Tests for the **component-layer** config_factory (typed builder + validation).

Dict-parsing tests live in ``tests/unit_tests/recipes/test_dist_setup.py``.
"""

import pytest

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config, MegatronFSDPConfig
from nemo_automodel.components.distributed.config_factory import (
    STRATEGY_MAP,
    DistributedSetup,
    build_distributed_setup,
    validate_distributed_setup,
)
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.moe.config import MoEParallelizerConfig


# ---------------------------------------------------------------------------
# build_distributed_setup – happy paths
# ---------------------------------------------------------------------------


class TestBuildDistributedSetup:
    def test_minimal_fsdp2(self):
        result = build_distributed_setup(FSDP2Config())

        assert isinstance(result, DistributedSetup)
        assert isinstance(result.strategy_config, FSDP2Config)
        assert result.tp_size == 1
        assert result.pp_size == 1
        assert result.cp_size == 1
        assert result.ep_size == 1
        assert result.dp_size is None
        assert result.dp_replicate_size is None
        assert result.pp_enabled is False
        assert result.pipeline_config is None
        assert result.moe_config is None
        assert result.device_mesh is None
        assert result.moe_mesh is None
        assert result.activation_checkpointing is False

    def test_megatron_fsdp(self):
        result = build_distributed_setup(MegatronFSDPConfig())
        assert isinstance(result.strategy_config, MegatronFSDPConfig)
        assert result.strategy_config.zero_dp_strategy == 3

    def test_ddp(self):
        result = build_distributed_setup(DDPConfig())
        assert isinstance(result.strategy_config, DDPConfig)
        assert result.strategy_config.backend == "nccl"

    def test_all_parallelism_keys(self):
        result = build_distributed_setup(
            FSDP2Config(),
            dp_size=4,
            tp_size=2,
            pp_size=2,
            cp_size=2,
            ep_size=2,
            dp_replicate_size=2,
        )
        assert result.dp_size == 4
        assert result.tp_size == 2
        assert result.pp_size == 2
        assert result.cp_size == 2
        assert result.ep_size == 2
        assert result.dp_replicate_size == 2

    def test_pp_enabled_when_pp_gt_1(self):
        result = build_distributed_setup(FSDP2Config(), pp_size=2)
        assert result.pp_enabled is True

    def test_pp_disabled_when_pp_eq_1(self):
        result = build_distributed_setup(FSDP2Config(), pp_size=1)
        assert result.pp_enabled is False

    def test_with_pipeline_config(self):
        pc = PipelineConfig(pp_schedule="1f1b", pp_microbatch_size=4)
        result = build_distributed_setup(FSDP2Config(), pp_size=2, pipeline_config=pc)
        assert result.pp_enabled is True
        assert result.pipeline_config is pc
        assert result.pipeline_config.pp_microbatch_size == 4

    def test_with_moe_config(self):
        mc = MoEParallelizerConfig(ignore_router_for_ac=True)
        result = build_distributed_setup(FSDP2Config(), ep_size=2, moe_config=mc)
        assert result.moe_config is mc
        assert result.moe_config.ignore_router_for_ac is True

    def test_activation_checkpointing_forwarded(self):
        result = build_distributed_setup(
            FSDP2Config(activation_checkpointing=True),
            activation_checkpointing=True,
        )
        assert result.activation_checkpointing is True
        assert result.strategy_config.activation_checkpointing is True


# ---------------------------------------------------------------------------
# validate_distributed_setup – constraint violations
# ---------------------------------------------------------------------------


class TestValidation:
    def test_megatron_fsdp_rejects_pp(self):
        with pytest.raises(ValueError, match="pipeline parallelism"):
            build_distributed_setup(MegatronFSDPConfig(), pp_size=2)

    def test_megatron_fsdp_rejects_ep(self):
        with pytest.raises(ValueError, match="expert parallelism"):
            build_distributed_setup(MegatronFSDPConfig(), ep_size=2)

    def test_megatron_fsdp_rejects_sequence_parallel(self):
        with pytest.raises(ValueError, match="sequence_parallel"):
            build_distributed_setup(MegatronFSDPConfig(sequence_parallel=True))

    def test_ddp_rejects_tp(self):
        with pytest.raises(ValueError, match="tensor parallelism"):
            build_distributed_setup(DDPConfig(), tp_size=2)

    def test_ddp_rejects_pp(self):
        with pytest.raises(ValueError, match="pipeline parallelism"):
            build_distributed_setup(DDPConfig(), pp_size=2)

    def test_ddp_rejects_cp(self):
        with pytest.raises(ValueError, match="context parallelism"):
            build_distributed_setup(DDPConfig(), cp_size=2)

    def test_ddp_rejects_ep(self):
        with pytest.raises(ValueError, match="expert parallelism"):
            build_distributed_setup(DDPConfig(), ep_size=2)

    def test_ddp_rejects_hsdp(self):
        with pytest.raises(ValueError, match="HSDP"):
            build_distributed_setup(DDPConfig(), dp_replicate_size=2)

    def test_pipeline_requires_pp_gt_1(self):
        pc = PipelineConfig(pp_schedule="1f1b")
        with pytest.raises(ValueError, match="pp_size > 1"):
            build_distributed_setup(FSDP2Config(), pp_size=1, pipeline_config=pc)

    def test_moe_requires_ep_gt_1(self):
        mc = MoEParallelizerConfig()
        with pytest.raises(ValueError, match="ep_size > 1"):
            build_distributed_setup(FSDP2Config(), ep_size=1, moe_config=mc)


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
# Integration: typed builder with full configs
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_megatron_fsdp_with_valid_options(self):
        result = build_distributed_setup(
            MegatronFSDPConfig(
                zero_dp_strategy=2,
                overlap_grad_reduce=False,
                activation_checkpointing=True,
            ),
            tp_size=2,
            activation_checkpointing=True,
        )
        assert result.strategy_config.zero_dp_strategy == 2
        assert result.strategy_config.overlap_grad_reduce is False
        assert result.strategy_config.activation_checkpointing is True
        assert result.tp_size == 2

    def test_fsdp2_full_config(self):
        result = build_distributed_setup(
            FSDP2Config(
                sequence_parallel=True,
                activation_checkpointing=True,
                defer_fsdp_grad_sync=False,
            ),
            tp_size=4,
            pp_size=2,
            cp_size=2,
            dp_replicate_size=2,
            activation_checkpointing=True,
            pipeline_config=PipelineConfig(pp_schedule="1f1b", pp_microbatch_size=2),
        )
        assert result.strategy_config.sequence_parallel is True
        assert result.strategy_config.activation_checkpointing is True
        assert result.pp_enabled is True
        assert isinstance(result.pipeline_config, PipelineConfig)

    def test_combined_pipeline_and_moe(self):
        result = build_distributed_setup(
            FSDP2Config(),
            pp_size=2,
            ep_size=2,
            pipeline_config=PipelineConfig(pp_schedule="1f1b"),
            moe_config=MoEParallelizerConfig(ignore_router_for_ac=True),
        )
        assert result.pp_enabled is True
        assert isinstance(result.pipeline_config, PipelineConfig)
        assert isinstance(result.moe_config, MoEParallelizerConfig)
        assert result.moe_config.ignore_router_for_ac is True

    @pytest.mark.parametrize(
        "strategy_config",
        [FSDP2Config(backend="gloo"), MegatronFSDPConfig(backend="gloo"), DDPConfig(backend="gloo")],
        ids=["fsdp2", "megatron_fsdp", "ddp"],
    )
    def test_backend_configuration(self, strategy_config):
        result = build_distributed_setup(strategy_config)
        assert result.strategy_config.backend == "gloo"
