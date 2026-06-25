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

"""
Strategy-specific distributed training configuration classes.

Design principle:
- Size params (dp_size, dp_replicate_size, tp_size, pp_size, cp_size, ep_size)
  are grouped in ``ParallelismSizes``.
- dp_replicate_size is FSDP2-only: raises assertion if passed with non-FSDP2 config
- Strategy-specific configs contain only *additional* flags unique to each strategy
- Managers become normal classes that accept (config, device_mesh)

Usage:
    from nemo_automodel.components.distributed.config import FSDP2Config, MegatronFSDPConfig, DDPConfig

    # FSDP2 with custom options
    config = FSDP2Config(sequence_parallel=True, activation_checkpointing=True)

    # MegatronFSDP with custom options
    config = MegatronFSDPConfig(zero_dp_strategy=3, overlap_grad_reduce=True)

    # DDP with activation checkpointing
    config = DDPConfig(activation_checkpointing=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
    from nemo_automodel.components.distributed.pipelining.config import PipelineConfig

# Type aliases for API signatures.
ActivationCheckpointingMode = Union[bool, Literal["selective"]]
DistributedStrategyConfig = Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig"]
# Backwards-compatible alias for external / type-checking references.
DistributedConfig = DistributedStrategyConfig


@dataclass(frozen=True)
class DistributedSetup:
    """Resolved distributed topology and execution policies."""

    mesh_context: "MeshContext"
    strategy_config: DistributedStrategyConfig | None = None
    pipeline_config: "PipelineConfig | None" = None
    moe_parallel_config: "MoEParallelizerConfig | None" = None
    activation_checkpointing: ActivationCheckpointingMode = False

    @classmethod
    def build(
        cls,
        strategy: str | DistributedStrategyConfig = "fsdp2",
        parallelism_sizes: "ParallelismSizes | None" = None,
        pipeline_config: "PipelineConfig | dict | None" = None,
        moe_parallel_config: "MoEParallelizerConfig | dict | None" = None,
        activation_checkpointing: ActivationCheckpointingMode = False,
        world_size: int | None = None,
    ) -> "DistributedSetup":
        """Create a resolved distributed setup from sizes and policy configs.

        Intentionally, this function is forgiving wrt the input types, allowing
        strings for the strategy and dicts for the pipeline and MoE configs.
        """
        from nemo_automodel.components.distributed.init_utils import get_world_size_safe
        from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
        from nemo_automodel.components.distributed.pipelining.config import PipelineConfig

        if world_size is None:
            world_size = get_world_size_safe()

        strategy_config = _resolve_strategy_config(strategy)

        if parallelism_sizes is None:
            parallelism_sizes = ParallelismSizes()

        pp_size = parallelism_sizes.pp_size
        ep_size = parallelism_sizes.ep_size
        if pipeline_config is not None and pp_size <= 1:
            raise ValueError("pipeline_config requires pp_size > 1")
        if moe_parallel_config is not None and ep_size <= 1:
            raise ValueError("moe_parallel_config requires ep_size > 1")
        if pp_size > 1 and pipeline_config is None:
            pipeline_config = PipelineConfig()
        if isinstance(pipeline_config, dict):
            pipeline_config = PipelineConfig(**pipeline_config)
        if ep_size > 1 and moe_parallel_config is None:
            moe_parallel_config = MoEParallelizerConfig()
        if isinstance(moe_parallel_config, dict):
            moe_parallel_config = MoEParallelizerConfig(**moe_parallel_config)

        mesh_context = MeshContext.build(
            strategy_config,
            parallelism_sizes=parallelism_sizes,
            world_size=world_size,
        )

        return cls(
            mesh_context=mesh_context,
            strategy_config=strategy_config,
            pipeline_config=pipeline_config,
            moe_parallel_config=moe_parallel_config,
            activation_checkpointing=activation_checkpointing,
        )


@dataclass
class MoEParallelizerConfig:
    """Configuration for MoE model parallelization (EP + FSDP settings)."""

    ignore_router_for_ac: bool = False
    reshard_after_forward: bool = False
    lm_head_precision: Optional[Union[str, torch.dtype]] = None
    wrap_outer_model: bool = True
    mp_policy: Optional[MixedPrecisionPolicy] = None

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class FSDP2Config:
    """
    Additional configuration for FSDP2 distributed training.

    Note: Size parameters (dp_size, dp_replicate_size, tp_size, pp_size, cp_size, ep_size)
    are grouped separately in ``ParallelismSizes``.

    Attributes:
        sequence_parallel (bool): Enable sequence parallelism in TP plan.
        tp_plan (Optional[dict]): Custom TP plan. If None, auto-selected based on model type.
        patch_is_packed_sequence (bool): Patch ``transformers._is_packed_sequence`` to always
            return Python ``False``. This does two things: (1) removes a CPU-GPU sync per
            attention layer (``aten::is_nonzero`` triggered by HF when batch_size==1), and
            (2) ensures static attention shapes for ``torch.compile``. Safe for standard
            (non-packed) training only. Disable if using packed-sequence training
            (position_ids that reset to 0 mid-sequence). Default ``False``.
        mp_policy (Optional[MixedPrecisionPolicy]): MixedPrecisionPolicy for FSDP2.
            If ``None`` (default), uses bf16 forward/backward compute with fp32
            gradient reduction. Pair this with ``model.torch_dtype: float32`` for
            the Megatron-style fp32 master-weights pattern. Override from YAML
            using the ``_target_`` pattern::

                mp_policy:
                  _target_: torch.distributed.fsdp.MixedPrecisionPolicy
                  param_dtype: bfloat16
                  reduce_dtype: float32
                  output_dtype: bfloat16

            See ``docs/guides/mixed-precision-training.md`` for the full set of recommended
            patterns and the bf16-storage trap.
        offload_policy (Optional[CPUOffloadPolicy]): CPUOffloadPolicy for CPU offloading.
        autocast_dtype (Optional[torch.dtype]): If set, wraps the forward pass in
            ``torch.autocast(device_type="cuda", dtype=autocast_dtype)``.  Use with
            ``output_dtype=float32`` in mp_policy to keep the residual stream in fp32
            while running matmuls in lower precision.  Set to ``None`` to disable.
            Can be set from YAML as a string (e.g. ``autocast_dtype: bfloat16``).
        activation_checkpointing (bool | "selective"): Enable activation checkpointing. ``True`` keeps the existing
            full activation checkpointing behavior. ``"selective"`` wraps transformer blocks with PyTorch selective
            activation checkpointing.
        defer_fsdp_grad_sync (bool): Defer FSDP gradient sync to final micro-batch.
        reshard_after_forward (Optional[bool]): Override layer-level FSDP2 resharding.
            If ``None`` (default), AutoModel reshares all but the last layer outside
            pipeline parallelism. Set ``False`` for a ZeRO-2-like benchmark where
            gathered parameters stay resident after forward.
        enable_async_tensor_parallel (bool): Enable async tensor parallelism via
            ``torch._inductor.config._micro_pipeline_tp``.  Overlaps ReduceScatter with
            compute in row-parallel layers.  Requires ``sequence_parallel=True`` (forced
            automatically with a warning if not set).  Also enables symmetric memory for
            the TP group.
        enable_compile (bool): Apply per-layer ``torch.compile`` to transformer decoder
            layers (with NO_REENTRANT activation checkpointing inside each compiled layer).
            Skips whole-model compile so that checkpoint loading does not produce
            ``_orig_mod`` key-prefix mismatches.
        enable_fsdp2_prefetch (bool): Enable explicit forward/backward prefetch chains
            between FSDP2 sharded layers.  Default ``True``.
        fsdp2_backward_prefetch_depth (int): Number of FSDP units to prefetch during
            backward pass.  ``2`` hides AllGather behind compute; ``1`` reduces peak
            memory at a small throughput cost.  Default ``2``.
        fsdp2_forward_prefetch_depth (int): Number of FSDP units to prefetch during
            forward pass.  Default ``1``.
    """

    sequence_parallel: bool = False
    tp_plan: Optional[dict] = None
    patch_is_packed_sequence: bool = False
    mp_policy: Optional[MixedPrecisionPolicy] = field(
        default_factory=lambda: MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
    )
    offload_policy: Optional[CPUOffloadPolicy] = None
    autocast_dtype: Optional[torch.dtype] = None
    activation_checkpointing: ActivationCheckpointingMode = False
    defer_fsdp_grad_sync: bool = True
    reshard_after_forward: Optional[bool] = None
    enable_async_tensor_parallel: bool = False
    enable_compile: bool = False
    enable_fsdp2_prefetch: bool = False
    fsdp2_backward_prefetch_depth: int = 2
    fsdp2_forward_prefetch_depth: int = 1

    def __post_init__(self):
        if self.mp_policy is None:
            # FSDP2 default: bf16 compute and fp32 gradient reduction. Pair with
            # ``model.torch_dtype: float32`` for fp32 optimizer state. See
            # ``docs/guides/mixed-precision-training.md``.
            self.mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (shallow, preserves policy objects)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class MegatronFSDPConfig:
    """
    Additional configuration for MegatronFSDP distributed training.

    Note: Size parameters (dp_size, tp_size, cp_size) are grouped separately in
    ``ParallelismSizes``. MegatronFSDP does not
    support pp_size, dp_replicate_size, or ep_size.

    Attributes:
        megatron_fsdp_unit_modules (Optional[List[str]]): List of unit modules to be
            wrapped with MegatronFSDP.
        zero_dp_strategy (int): Data parallel sharding strategy.
        init_fsdp_with_meta_device (bool): Initialize MegatronFSDP with meta device if True.
        grad_reduce_in_fp32 (bool): Reduce gradients in fp32 if True.
        preserve_fp32_weights (bool): Preserve fp32 weights if True.
        overlap_grad_reduce (bool): Overlap gradient reduction if True.
        overlap_param_gather (bool): Overlap parameter gathering if True.
        check_for_nan_in_grad (bool): Check for NaN in gradients if True.
        average_in_collective (bool): Average in collective if True.
        disable_bucketing (bool): Disable bucketing if True.
        calculate_per_token_loss (bool): Calculate per token loss if True.
        keep_fp8_transpose_cache (bool): Keep fp8 transpose cache when using custom FSDP if True.
        nccl_ub (bool): Use NCCL UBs if True.
        fsdp_double_buffer (bool): Use double buffer if True.
        activation_checkpointing (bool): Enable activation checkpointing for transformer
            MLP layers to save memory.
    """

    megatron_fsdp_unit_modules: List[str] = field(
        default_factory=lambda: ["transformers.models.llama.modeling_llama.LlamaDecoderLayer"]
    )
    zero_dp_strategy: int = 3
    init_fsdp_with_meta_device: bool = False
    grad_reduce_in_fp32: bool = False
    preserve_fp32_weights: bool = False
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    check_for_nan_in_grad: bool = True
    average_in_collective: bool = False
    disable_bucketing: bool = False
    calculate_per_token_loss: bool = False
    keep_fp8_transpose_cache: bool = False
    nccl_ub: bool = False
    fsdp_double_buffer: bool = False
    activation_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (shallow, preserves objects)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class DDPConfig:
    """
    Additional configuration for DDP distributed training.

    Note: DDP does not support tensor parallelism, pipeline parallelism, or expert parallelism.
    Only dp_size is relevant (inferred from world_size).

    Attributes:
        activation_checkpointing (bool | "selective"): Enable activation checkpointing. ``True`` keeps the existing
            full activation checkpointing behavior. ``"selective"`` wraps transformer blocks with PyTorch selective
            activation checkpointing.
        broadcast_buffers (bool): Synchronize module buffers before each forward.
        find_unused_parameters (bool): Traverse the autograd graph to support unused parameters.
        static_graph (bool): Tell DDP the used/unused parameter set is stable.
        bucket_cap_mb (Optional[float]): DDP gradient bucket size in MiB. ``None`` uses PyTorch's default.
        gradient_as_bucket_view (bool): Make gradients views into DDP buckets after the first iteration.
        autocast_dtype (Optional[torch.dtype]): If set, recipes can wrap the forward pass in
            ``torch.autocast(device_type="cuda", dtype=autocast_dtype)``. Set to ``None`` to disable.
            Can be set from YAML as a string (e.g. ``autocast_dtype: bfloat16``).
    """

    activation_checkpointing: ActivationCheckpointingMode = False
    broadcast_buffers: bool = False
    find_unused_parameters: bool = False
    static_graph: bool = False
    bucket_cap_mb: Optional[float] = None
    gradient_as_bucket_view: bool = False
    autocast_dtype: Optional[torch.dtype] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


_StrategyConfigClass = type[FSDP2Config] | type[MegatronFSDPConfig] | type[DDPConfig]
_STRATEGY_MAP: Dict[str, _StrategyConfigClass] = {
    "fsdp2": FSDP2Config,
    "megatron_fsdp": MegatronFSDPConfig,
    "megatron-fsdp": MegatronFSDPConfig,
    "mfsdp": MegatronFSDPConfig,
    "ddp": DDPConfig,
}


def _resolve_strategy_config(
    strategy: str | DistributedStrategyConfig,
    **strategy_kwargs: Any,
) -> DistributedStrategyConfig:
    """Resolve a setup-level strategy name or config object."""
    if isinstance(strategy, (FSDP2Config, MegatronFSDPConfig, DDPConfig)):
        if strategy_kwargs:
            raise ValueError("Strategy kwargs cannot be passed with an instantiated strategy config.")
        return strategy

    if not isinstance(strategy, str):
        raise ValueError(f"Unknown distributed strategy type: {type(strategy)}")

    strategy_name = strategy.lower()
    if strategy_name not in _STRATEGY_MAP:
        valid = sorted(_STRATEGY_MAP)
        raise ValueError(f"Unknown strategy: {strategy}. Valid strategies: {valid}")
    strategy_cls = _STRATEGY_MAP[strategy_name]
    valid_fields = {f.name for f in fields(strategy_cls)}
    unknown = set(strategy_kwargs) - valid_fields
    if unknown:
        raise ValueError(f"Unknown options for strategy '{strategy_name}': {sorted(unknown)}")
    return strategy_cls(**strategy_kwargs)


__all__ = [
    "DDPConfig",
    "DistributedSetup",
    "DistributedStrategyConfig",
    "FSDP2Config",
    "MegatronFSDPConfig",
    "MoEParallelizerConfig",
]
