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

"""Tests for the **recipe-layer** YAML / dict parsing (``_dist_setup``).

Typed validation tests live in ``tests/unit_tests/distributed/test_mesh.py``.
"""

import pytest

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config, MegatronFSDPConfig
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.moe.config import MoEParallelizerConfig
from nemo_automodel.recipes._dist_setup import parse_distributed_section

# ---------------------------------------------------------------------------
# Basic dict parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_minimal_fsdp2(self):
        result = parse_distributed_section({"strategy": "fsdp2"})

        assert isinstance(result, dict)
        assert isinstance(result["strategy_config"], FSDP2Config)
        assert result["strategy_config"].sequence_parallel is False
        assert result["strategy_config"].activation_checkpointing is False
        assert result["tp_size"] == 1
        assert result["pp_size"] == 1
        assert result["cp_size"] == 1
        assert result["ep_size"] == 1
        assert result["dp_size"] is None
        assert result["dp_replicate_size"] is None
        assert result["pp_enabled"] is False
        assert result["pipeline_config"] is None
        assert result["moe_config"] is None

    def test_default_strategy_is_fsdp2(self):
        result = parse_distributed_section({})
        assert isinstance(result["strategy_config"], FSDP2Config)

    def test_megatron_fsdp(self):
        result = parse_distributed_section({"strategy": "megatron_fsdp"})
        assert isinstance(result["strategy_config"], MegatronFSDPConfig)
        assert result["strategy_config"].zero_dp_strategy == 3

    def test_ddp(self):
        result = parse_distributed_section({"strategy": "ddp"})
        assert isinstance(result["strategy_config"], DDPConfig)
        assert result["strategy_config"].backend == "nccl"

    def test_all_parallelism_keys(self):
        cfg = {
            "strategy": "fsdp2",
            "dp_size": 4,
            "tp_size": 2,
            "pp_size": 2,
            "cp_size": 2,
            "ep_size": 2,
            "dp_replicate_size": 2,
        }
        result = parse_distributed_section(cfg)
        assert result["dp_size"] == 4
        assert result["tp_size"] == 2
        assert result["pp_size"] == 2
        assert result["cp_size"] == 2
        assert result["ep_size"] == 2
        assert result["dp_replicate_size"] == 2

    def test_dp_size_defaults_to_none(self):
        result = parse_distributed_section({"strategy": "fsdp2"})
        assert result["dp_size"] is None

    def test_strategy_specific_fields_pass_through(self):
        cfg = {"strategy": "fsdp2", "sequence_parallel": True, "defer_fsdp_grad_sync": False}
        result = parse_distributed_section(cfg)
        assert result["strategy_config"].sequence_parallel is True
        assert result["strategy_config"].defer_fsdp_grad_sync is False

    def test_config_dict_not_mutated(self):
        original = {"strategy": "fsdp2", "tp_size": 2, "activation_checkpointing": True}
        copy = original.copy()
        parse_distributed_section(original)
        assert original == copy


# ---------------------------------------------------------------------------
# Pipeline sub-config parsing
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_pipeline_config_created(self):
        cfg = {"pp_size": 2, "pipeline": {"pp_schedule": "1f1b", "pp_microbatch_size": 4}}
        result = parse_distributed_section(cfg)
        assert result["pp_enabled"] is True
        assert isinstance(result["pipeline_config"], PipelineConfig)
        assert result["pipeline_config"].pp_schedule == "1f1b"
        assert result["pipeline_config"].pp_microbatch_size == 4

    def test_pipeline_fields_pass_through(self):
        cfg = {
            "pp_size": 2,
            "pipeline": {"pp_schedule": "gpipe", "pp_microbatch_size": 8, "pp_batch_size": 32, "layers_per_stage": 12},
        }
        result = parse_distributed_section(cfg)
        assert result["pipeline_config"].pp_schedule == "gpipe"
        assert result["pipeline_config"].pp_batch_size == 32
        assert result["pipeline_config"].layers_per_stage == 12

    def test_empty_pipeline_dict_uses_defaults(self):
        result = parse_distributed_section({"pp_size": 2, "pipeline": {}})
        assert isinstance(result["pipeline_config"], PipelineConfig)
        assert result["pipeline_config"].pp_schedule == "1f1b"

    def test_pp_enabled_true_when_pp_gt_1(self):
        assert parse_distributed_section({"pp_size": 2})["pp_enabled"] is True

    def test_pp_enabled_false_when_pp_eq_1(self):
        assert parse_distributed_section({"pp_size": 1})["pp_enabled"] is False


# ---------------------------------------------------------------------------
# MoE sub-config parsing
# ---------------------------------------------------------------------------


class TestMoE:
    def test_moe_config_created(self):
        cfg = {"ep_size": 2, "moe": {"ignore_router_for_ac": True}}
        result = parse_distributed_section(cfg)
        assert isinstance(result["moe_config"], MoEParallelizerConfig)
        assert result["moe_config"].ignore_router_for_ac is True

    def test_moe_fields_pass_through(self):
        cfg = {
            "ep_size": 2,
            "moe": {"ignore_router_for_ac": True, "reshard_after_forward": True, "wrap_outer_model": False},
        }
        result = parse_distributed_section(cfg)
        assert result["moe_config"].reshard_after_forward is True
        assert result["moe_config"].wrap_outer_model is False

    def test_empty_moe_dict_uses_defaults(self):
        result = parse_distributed_section({"ep_size": 2, "moe": {}})
        assert isinstance(result["moe_config"], MoEParallelizerConfig)
        assert result["moe_config"].ignore_router_for_ac is False

    def test_mp_policy_none_when_omitted(self):
        result = parse_distributed_section({"ep_size": 2, "moe": {}})
        assert result["moe_config"].mp_policy is None

    def test_mp_policy_target_instantiated(self):
        """mp_policy with resolved _target_ callable is instantiated to MixedPrecisionPolicy."""
        import torch
        from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy

        cfg = {
            "ep_size": 2,
            "moe": {
                "mp_policy": {
                    "_target_": MixedPrecisionPolicy,
                    "param_dtype": "bfloat16",
                    "reduce_dtype": "float32",
                    "output_dtype": "bfloat16",
                    "cast_forward_inputs": True,
                }
            },
        }
        result = parse_distributed_section(cfg)
        mp = result["moe_config"].mp_policy
        assert isinstance(mp, MixedPrecisionPolicy)
        assert mp.param_dtype == torch.bfloat16
        assert mp.reduce_dtype == torch.float32
        assert mp.output_dtype == torch.bfloat16
        assert mp.cast_forward_inputs is True

    def test_mp_policy_in_to_dict(self):
        from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy

        cfg = {
            "ep_size": 2,
            "moe": {
                "mp_policy": {
                    "_target_": MixedPrecisionPolicy,
                    "param_dtype": "bfloat16",
                    "reduce_dtype": "float32",
                }
            },
        }
        result = parse_distributed_section(cfg)
        d = result["moe_config"].to_dict()
        assert "mp_policy" in d
        assert isinstance(d["mp_policy"], MixedPrecisionPolicy)

    def test_mp_policy_passthrough_when_already_instantiated(self):
        """MixedPrecisionPolicy object passed directly is kept as-is."""
        import torch
        from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy

        policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
        cfg = {"ep_size": 2, "moe": {"mp_policy": policy}}
        result = parse_distributed_section(cfg)
        assert result["moe_config"].mp_policy is policy


# ---------------------------------------------------------------------------
# activation_checkpointing routing (EP-aware)
# ---------------------------------------------------------------------------


class TestActivationCheckpointingRouting:
    def test_routes_to_strategy_when_no_ep(self):
        result = parse_distributed_section({"strategy": "fsdp2", "activation_checkpointing": True, "ep_size": 1})
        assert result["strategy_config"].activation_checkpointing is True
        assert result["activation_checkpointing"] is True

    def test_not_on_strategy_when_ep_gt_1(self):
        result = parse_distributed_section(
            {"strategy": "fsdp2", "activation_checkpointing": True, "ep_size": 2, "moe": {}}
        )
        assert result["strategy_config"].activation_checkpointing is False
        assert result["activation_checkpointing"] is True

    def test_defaults_to_false(self):
        result = parse_distributed_section({"strategy": "fsdp2"})
        assert result["strategy_config"].activation_checkpointing is False
        assert result["activation_checkpointing"] is False

    def test_works_with_ddp(self):
        result = parse_distributed_section({"strategy": "ddp", "activation_checkpointing": True})
        assert result["strategy_config"].activation_checkpointing is True


# ---------------------------------------------------------------------------
# Validation errors surfaced through dict parsing
# ---------------------------------------------------------------------------


class TestValidation:
    """YAML-level sanity checks (strategy constraints are tested in test_mesh.py)."""

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            parse_distributed_section({"strategy": "unknown"})

    def test_pipeline_requires_pp_gt_1(self):
        """Pipeline section is silently discarded when pp_size <= 1
        (common when a YAML template is overridden via CLI)."""
        result = parse_distributed_section({"pp_size": 1, "pipeline": {"pp_schedule": "1f1b"}})
        assert result["pipeline_config"] is None
        assert result["pp_enabled"] is False

    def test_pipeline_rejects_default_pp_size(self):
        """Pipeline section is silently discarded when pp_size defaults to 1."""
        result = parse_distributed_section({"pipeline": {"pp_schedule": "1f1b"}})
        assert result["pipeline_config"] is None
        assert result["pp_enabled"] is False

    def test_moe_requires_ep_gt_1(self):
        """MoE section is silently discarded when ep_size <= 1
        (common when a YAML template is overridden via CLI)."""
        result = parse_distributed_section({"ep_size": 1, "moe": {"ignore_router_for_ac": True}})
        assert result["moe_config"] is None

    def test_moe_rejects_default_ep_size(self):
        """MoE section is silently discarded when ep_size defaults to 1."""
        result = parse_distributed_section({"moe": {"ignore_router_for_ac": True}})
        assert result["moe_config"] is None

    def test_unknown_field_for_strategy(self):
        with pytest.raises(ValueError, match="Unknown options"):
            parse_distributed_section({"strategy": "ddp", "sequence_parallel": True})

    def test_hydra_meta_keys_ignored(self):
        """Hydra/OmegaConf internal keys (``_target_``, ``_recursive_``, etc.)
        must be silently stripped and not treated as unknown strategy options."""
        cfg = {"strategy": "megatron_fsdp", "_target_": "some.hydra.Target"}
        result = parse_distributed_section(cfg)
        assert isinstance(result["strategy_config"], MegatronFSDPConfig)

    @pytest.mark.parametrize("meta_key", ["_target_", "_recursive_", "_convert_"])
    def test_various_hydra_meta_keys_ignored(self, meta_key):
        cfg = {"strategy": "fsdp2", meta_key: "value"}
        result = parse_distributed_section(cfg)
        assert isinstance(result["strategy_config"], FSDP2Config)


# ---------------------------------------------------------------------------
# Integration: full YAML-like dicts
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_megatron_fsdp_with_valid_options(self):
        cfg = {
            "strategy": "megatron_fsdp",
            "tp_size": 2,
            "zero_dp_strategy": 2,
            "overlap_grad_reduce": False,
            "activation_checkpointing": True,
        }
        result = parse_distributed_section(cfg)
        assert result["strategy_config"].zero_dp_strategy == 2
        assert result["strategy_config"].overlap_grad_reduce is False
        assert result["strategy_config"].activation_checkpointing is True
        assert result["tp_size"] == 2

    def test_fsdp2_full_config(self):
        cfg = {
            "strategy": "fsdp2",
            "tp_size": 4,
            "pp_size": 2,
            "cp_size": 2,
            "dp_replicate_size": 2,
            "sequence_parallel": True,
            "activation_checkpointing": True,
            "defer_fsdp_grad_sync": False,
            "pipeline": {"pp_schedule": "1f1b", "pp_microbatch_size": 2},
        }
        result = parse_distributed_section(cfg)
        assert result["strategy_config"].sequence_parallel is True
        assert result["strategy_config"].activation_checkpointing is True
        assert result["pp_enabled"] is True
        assert isinstance(result["pipeline_config"], PipelineConfig)

    def test_combined_pipeline_and_moe(self):
        cfg = {
            "strategy": "fsdp2",
            "pp_size": 2,
            "ep_size": 2,
            "pipeline": {"pp_schedule": "1f1b"},
            "moe": {"ignore_router_for_ac": True},
        }
        result = parse_distributed_section(cfg)
        assert result["pp_enabled"] is True
        assert isinstance(result["pipeline_config"], PipelineConfig)
        assert isinstance(result["moe_config"], MoEParallelizerConfig)
        assert result["moe_config"].ignore_router_for_ac is True

    @pytest.mark.parametrize("strategy", ["fsdp2", "megatron_fsdp", "ddp"])
    def test_backend_configuration(self, strategy):
        result = parse_distributed_section({"strategy": strategy, "backend": "gloo"})
        assert result["strategy_config"].backend == "gloo"
