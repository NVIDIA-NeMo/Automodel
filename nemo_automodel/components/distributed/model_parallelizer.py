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

"""Resolve and execute model-owned parallelization sidecars."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from torch import nn

from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.config import ActivationCheckpointingMode, DistributedStrategyConfig
    from nemo_automodel.components.distributed.mesh import MeshContext
    from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy


@dataclass(frozen=True, slots=True, kw_only=True)
class ParallelizeContext:
    """Runtime topology and policies passed to a model parallelizer."""

    mesh: MeshContext
    strategy: DistributedStrategyConfig | None
    moe: MoEParallelizerConfig | None = None
    activation_checkpointing: ActivationCheckpointingMode = False
    reapply_trainability: Callable[[nn.Module], None] | None = None


class ModelParallelizer(Protocol):
    """Model-owned contract for applying requested parallelisms."""

    def parallelize(self, model: nn.Module, context: ParallelizeContext, /) -> nn.Module:
        """Apply TP, CP, EP, activation checkpointing, and data parallelism."""
        ...


@dataclass(frozen=True, slots=True)
class DefaultModelParallelizer:
    """Infrastructure-owned implementation used when a model has no sidecar."""

    def parallelize(self, model: nn.Module, context: ParallelizeContext, /) -> nn.Module:
        """Apply the standard dense or expert-parallel execution path."""
        if context.mesh.ep_size > 1:
            return _parallelize_moe(model, context)
        if isinstance(context.strategy, FSDP2Config):
            return _parallelize_fsdp2(model, context)
        raise TypeError(
            "DefaultModelParallelizer supports FSDP2 or expert-parallel execution; "
            f"got strategy={type(context.strategy).__name__}."
        )


_DEFAULT_PARALLELIZER = DefaultModelParallelizer()


@dataclass(frozen=True, slots=True)
class FSDP2ModelParallelizer(DefaultModelParallelizer):
    """Adapt an FSDP2 strategy to the model-owned sidecar contract.

    Args:
        strategy: Object implementing the exported FSDP2 strategy API.
    """

    strategy: ParallelizationStrategy

    def parallelize(self, model: nn.Module, context: ParallelizeContext, /) -> nn.Module:
        """Apply MoE execution or the sidecar's specialized FSDP2 strategy."""
        if context.mesh.ep_size > 1:
            return _parallelize_moe(model, context)
        if isinstance(context.strategy, FSDP2Config):
            return _parallelize_fsdp2(model, context, strategy=self.strategy)
        return super().parallelize(model, context)


def get_model_parallelizer(model: nn.Module) -> ModelParallelizer:
    """Return the class-owned sidecar, or the shared default implementation."""
    parallelizer = getattr(type(model), "parallelizer", None)
    if parallelizer is None:
        return _DEFAULT_PARALLELIZER
    parallelize = getattr(parallelizer, "parallelize", None)
    if not callable(parallelize):
        raise TypeError(
            f"{type(model).__name__}.parallelizer must implement parallelize(model, context); "
            f"got {type(parallelizer).__name__}."
        )
    return parallelizer


def parallelize_model(model: nn.Module, context: ParallelizeContext) -> nn.Module:
    """Apply all requested parallelisms through the model-owned contract."""
    return get_model_parallelizer(model).parallelize(model, context)


def _parallelize_fsdp2(
    model: nn.Module,
    context: ParallelizeContext,
    *,
    strategy: ParallelizationStrategy | None = None,
) -> nn.Module:
    """Apply the existing dense FSDP2 executor from a typed context."""
    from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

    config = context.strategy
    if not isinstance(config, FSDP2Config):
        raise TypeError(f"FSDP2 parallelization requires FSDP2Config, got {type(config).__name__}.")
    if context.mesh.device_mesh is None:
        raise ValueError("FSDP2 parallelization requires context.mesh.device_mesh.")

    if strategy is None:
        strategy = DefaultParallelizationStrategy()
    parallelize = getattr(strategy, "parallelize", None)
    if not callable(parallelize):
        raise TypeError(
            "FSDP2ModelParallelizer.strategy must implement parallelize(model, device_mesh, ...); "
            f"got {type(strategy).__name__}."
        )

    return parallelize(
        model=model,
        device_mesh=context.mesh.device_mesh,
        mp_policy=config.mp_policy,
        tp_shard_plan=config.tp_plan,
        offload_policy=config.offload_policy,
        sequence_parallel=config.sequence_parallel,
        activation_checkpointing=config.activation_checkpointing,
        enable_async_tensor_parallel=config.enable_async_tensor_parallel,
        enable_compile=config.enable_compile,
        enable_fsdp2_prefetch=config.enable_fsdp2_prefetch,
        fsdp2_backward_prefetch_depth=config.fsdp2_backward_prefetch_depth,
        fsdp2_forward_prefetch_depth=config.fsdp2_forward_prefetch_depth,
        reshard_after_forward=config.reshard_after_forward,
        activation_checkpointing_scope=config.activation_checkpointing_scope,
        frozen_multimodal_sharding=config.multimodal.frozen_sharding,
        reapply_trainability=context.reapply_trainability,
    )


def _parallelize_moe(model: nn.Module, context: ParallelizeContext) -> nn.Module:
    """Apply the existing TP, CP, EP, activation-checkpointing, and FSDP flow."""
    from nemo_automodel.components.moe.parallelizer import parallelize_model as parallelize_moe_model

    mesh = context.mesh
    if mesh.device_mesh is None or mesh.moe_mesh is None:
        raise ValueError("Expert parallelization requires both device_mesh and moe_mesh.")

    moe = context.moe or MoEParallelizerConfig()
    strategy = context.strategy
    if isinstance(strategy, FSDP2Config):
        mp_policy = moe.mp_policy if moe.mp_policy is not None else strategy.mp_policy
        tp_shard_plan = strategy.tp_plan
        sequence_parallel = strategy.sequence_parallel
        offload_policy = strategy.offload_policy
        reshard_after_forward = (
            strategy.reshard_after_forward if strategy.reshard_after_forward is not None else moe.reshard_after_forward
        )
        enable_async_tensor_parallel = strategy.enable_async_tensor_parallel
        activation_checkpointing_scope = strategy.activation_checkpointing_scope
        frozen_multimodal_sharding = strategy.multimodal.frozen_sharding
    else:
        mp_policy = moe.mp_policy
        tp_shard_plan = None
        sequence_parallel = False
        offload_policy = None
        reshard_after_forward = moe.reshard_after_forward
        enable_async_tensor_parallel = False
        activation_checkpointing_scope = "all"
        frozen_multimodal_sharding = "root"

    parallelize_moe_model(
        model,
        world_mesh=mesh.device_mesh,
        moe_mesh=mesh.moe_mesh,
        activation_checkpointing=context.activation_checkpointing,
        ignore_router_for_ac=moe.ignore_router_for_ac,
        activation_checkpointing_scope=activation_checkpointing_scope,
        reshard_after_forward=reshard_after_forward,
        lm_head_precision=moe.lm_head_precision,
        wrap_outer_model=moe.wrap_outer_model,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        tp_shard_plan=tp_shard_plan,
        sequence_parallel=sequence_parallel,
        enable_async_tensor_parallel=enable_async_tensor_parallel,
        frozen_multimodal_sharding=frozen_multimodal_sharding,
        reapply_trainability=context.reapply_trainability,
        **mesh.parallelize_axis_kwargs(),
    )
    return model


__all__ = [
    "DefaultModelParallelizer",
    "FSDP2ModelParallelizer",
    "ModelParallelizer",
    "ParallelizeContext",
    "get_model_parallelizer",
    "parallelize_model",
]
