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

import importlib as _importlib
from typing import TYPE_CHECKING

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    DistributedSetup,
    FSDP2Config,
    MegatronFSDPConfig,
    MoEParallelizerConfig,
    MultimodalDistributedConfig,
)
from nemo_automodel.components.distributed.init_utils import DistInfo, initialize_distributed
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.context_parallel.magi import make_magi_attn_func

__all__ = [
    "DDPConfig",
    "DistributedSetup",
    "DistInfo",
    "FSDP2Config",
    "MegatronFSDPConfig",
    "MeshContext",
    "MoEParallelizerConfig",
    "MultimodalDistributedConfig",
    "ParallelismSizes",
    "PipelineConfig",
    "get_fsdp_dp_mesh",
    "initialize_distributed",
]

_LAZY_ATTRS = {
    "AutoPipeline": (".pipelining.autopipeline", "AutoPipeline"),
    "BlockdiagCpModelState": (".blockdiag_cp", "BlockdiagCpModelState"),
    "ContextParallelSharder": (".context_parallel.sharder", "ContextParallelSharder"),
    "CpVisionFrameShardingConfig": (".cp_vision_frame_shard", "CpVisionFrameShardingConfig"),
    "DDPManager": (".ddp", "DDPManager"),
    "DefaultParallelizationStrategy": (".parallelizer", "DefaultParallelizationStrategy"),
    "DistributedStrategyConfig": (".config", "DistributedStrategyConfig"),
    "FSDP2Manager": (".fsdp2", "FSDP2Manager"),
    "FirstRankPerNode": (".utils", "FirstRankPerNode"),
    "HunyuanParallelizationStrategy": (".parallelizer", "HunyuanParallelizationStrategy"),
    "MULTIMODAL_SUFFIXES": (".pipelining.hf_utils", "MULTIMODAL_SUFFIXES"),
    "MagiState": (".context_parallel.magi", "MagiState"),
    "MambaContextParallel": (".context_parallel.mamba", "MambaContextParallel"),
    "MegatronFSDPManager": (".megatron_fsdp", "MegatronFSDPManager"),
    "MultimodalVisionConfig": (".config", "MultimodalVisionConfig"),
    "PARALLELIZATION_STRATEGIES": (".parallelizer", "PARALLELIZATION_STRATEGIES"),
    "PARALLELIZE_FUNCTIONS": (".optimized_tp_plans", "PARALLELIZE_FUNCTIONS"),
    "SELECTIVE_AC_WRAPPER_FLAG": (".activation_checkpointing", "SELECTIVE_AC_WRAPPER_FLAG"),
    "ShardLayout": (".context_parallel.sharder", "ShardLayout"),
    "WanParallelizationStrategy": (".parallelizer", "WanParallelizationStrategy"),
    "apply_selective_checkpointing_to_layers": (".activation_checkpointing", "apply_selective_checkpointing_to_layers"),
    "apply_submodule_checkpointing": (".activation_checkpointing", "apply_submodule_checkpointing"),
    "attach_context_parallel_hooks": (".context_parallel.utils", "attach_context_parallel_hooks"),
    "attach_cp_sdpa_hooks": (".context_parallel.utils", "attach_cp_sdpa_hooks"),
    "attach_te_context_parallel": (".context_parallel.utils", "attach_te_context_parallel"),
    "configure_fsdp_unused_param_reduction": (".parallelizer_utils", "configure_fsdp_unused_param_reduction"),
    "contiguous_local_indices": (".context_parallel.sharder", "contiguous_local_indices"),
    "convert_attention_mask_to_padding_mask": (".context_parallel.sharder", "convert_attention_mask_to_padding_mask"),
    "cp_blockdiag_sdpa": (".blockdiag_cp", "cp_blockdiag_sdpa"),
    "cp_dispatcher_suspended": (".context_parallel.utils", "cp_dispatcher_suspended"),
    "cp_vision_frame_sharding_active": (".cp_vision_frame_shard", "cp_vision_frame_sharding_active"),
    "create_ring_ulysses_mesh": (".mesh_utils", "create_ring_ulysses_mesh"),
    "current_blockdiag_cp_state": (".blockdiag_cp", "current_blockdiag_cp_state"),
    "dp_eval_sample_shard": (".utils", "dp_eval_sample_shard"),
    "ensure_fsdp_ops_sac_ignored": (".activation_checkpointing", "ensure_fsdp_ops_sac_ignored"),
    "ensure_profiler_ops_sac_ignored": (".activation_checkpointing", "ensure_profiler_ops_sac_ignored"),
    "fsdp2_sharding_enabled": (".fsdp2", "fsdp2_sharding_enabled"),
    "fully_shard_by_dtype": (".parallelizer_utils", "fully_shard_by_dtype"),
    "get_class_qualname": (".optimized_tp_plans", "_get_class_qualname"),
    "get_flat_mesh": (".mesh_utils", "get_flat_mesh"),
    "get_internal_fsdp_mp_policy": (".parallelizer_utils", "get_internal_fsdp_mp_policy"),
    "get_local_world_size_preinit": (".init_utils", "get_local_world_size_preinit"),
    "get_model_layer_groups": (".parallelizer", "get_model_layer_groups"),
    "get_parallel_plan": (".parallelizer", "_get_parallel_plan"),
    "get_submesh": (".mesh_utils", "get_submesh"),
    "get_sync_ctx": (".utils", "get_sync_ctx"),
    "get_text_module": (".pipelining.hf_utils", "get_text_module"),
    "get_world_size_safe": (".init_utils", "get_world_size_safe"),
    "is_selective_activation_checkpointing": (".activation_checkpointing", "is_selective_activation_checkpointing"),
    "make_cp_blockdiag_batch_and_ctx": (".blockdiag_cp", "make_cp_blockdiag_batch_and_ctx"),
    "make_magi_attn_func": (".context_parallel.magi", "make_magi_attn_func"),
    "make_selective_checkpoint_context_fn": (".activation_checkpointing", "make_selective_checkpoint_context_fn"),
    "maybe_distribute_visual": (".cp_vision_frame_shard", "maybe_distribute_visual"),
    "maybe_shard_optimizer": (".megatron_fsdp", "maybe_shard_optimizer"),
    "normalize_activation_checkpointing_scope": (".config", "normalize_activation_checkpointing_scope"),
    "register_parallel_strategy": (".parallelizer", "register_parallel_strategy"),
    "reject_unsupported_mtp_cp": (".parallelizer_utils", "reject_unsupported_mtp_cp"),
    "reject_unsupported_mtp_cp_pp": (".parallelizer_utils", "reject_unsupported_mtp_cp_pp"),
    "reset_cp_vision_group": (".cp_vision_frame_shard", "reset_cp_vision_group"),
    "resolve_strategy_config": (".config", "_resolve_strategy_config"),
    "restore_distributed_param_attrs": (".megatron_fsdp", "restore_distributed_param_attrs"),
    "round_robin_local_indices": (".context_parallel.sharder", "round_robin_local_indices"),
    "sdpa_backend_snapshot_context_fn": (".activation_checkpointing", "sdpa_backend_snapshot_context_fn"),
    "set_cp_vision_group": (".cp_vision_frame_shard", "set_cp_vision_group"),
    "setup_magi": (".context_parallel.magi", "setup_magi"),
    "shard_batch_aux_only": (".context_parallel.sharder", "shard_batch_aux_only"),
    "shard_batch_contiguous": (".context_parallel.sharder", "shard_batch_contiguous"),
    "shard_sequence_for_cp_contiguous": (".context_parallel.sharder", "shard_sequence_for_cp_contiguous"),
    "shard_sequence_for_cp_round_robin": (".context_parallel.sharder", "shard_sequence_for_cp_round_robin"),
    "snapshot_distributed_param_attrs": (".megatron_fsdp", "snapshot_distributed_param_attrs"),
    "split_batch_into_thd_chunks": (".thd_utils", "split_batch_into_thd_chunks"),
    "thd_padding_mask_from_token_ids": (".thd_utils", "thd_padding_mask_from_token_ids"),
    "transformer_engine_attention_backend_snapshot_context_fn": (
        ".activation_checkpointing",
        "transformer_engine_attention_backend_snapshot_context_fn",
    ),
    "translate_to_lora": (".parallel_styles", "translate_to_lora"),
    "unshard_context_parallel_tensor": (".context_parallel.utils", "unshard_context_parallel_tensor"),
    "unwrap_checkpoint_wrapper": (".activation_checkpointing", "unwrap_checkpoint_wrapper"),
}

__all__ += sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)
