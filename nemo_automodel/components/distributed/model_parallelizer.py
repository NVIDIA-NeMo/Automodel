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
from typing import TYPE_CHECKING

from torch import nn

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
    MoEParallelizerConfig,
)

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


@dataclass(frozen=True, slots=True)
class ModelParallelizer:
    """Model-owned sidecar with optional dense and MoE FSDP2 strategies."""

    strategy: ParallelizationStrategy | None = None
    moe_strategy: ParallelizationStrategy | None = None

    def parallelize(self, model: nn.Module, context: ParallelizeContext, /) -> nn.Module:
        """Apply TP, CP, EP, activation checkpointing, and data parallelism."""
        if context.mesh.ep_size > 1:
            return _parallelize_moe(model, context, strategy=self.moe_strategy)
        if isinstance(context.strategy, FSDP2Config):
            return _parallelize_fsdp2(model, context, strategy=self.strategy)
        if isinstance(context.strategy, DDPConfig):
            return _parallelize_ddp(model, context)
        if isinstance(context.strategy, MegatronFSDPConfig):
            return _parallelize_megatron_fsdp(model, context)
        name = type(context.strategy).__name__
        raise TypeError(f"ModelParallelizer does not support strategy={name}.")


_DEFAULT_PARALLELIZER = ModelParallelizer()


def get_model_parallelizer(model: nn.Module) -> ModelParallelizer:
    """Return the class-owned sidecar, or the shared default implementation."""
    parallelizer = getattr(type(model), "parallelizer", None)
    if parallelizer is None:
        return _DEFAULT_PARALLELIZER
    if not callable(getattr(parallelizer, "parallelize", None)):
        name = type(parallelizer).__name__
        raise TypeError(f"{type(model).__name__}.parallelizer must implement parallelize(model, context); got {name}.")
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
    from nemo_automodel.components.distributed.fsdp2 import (
        _patch_is_packed_sequence_for_training,
        fsdp2_sharding_enabled,
    )
    from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

    config = context.strategy
    assert isinstance(config, FSDP2Config)
    if context.mesh.device_mesh is None:
        raise ValueError("FSDP2 parallelization requires context.mesh.device_mesh.")

    if config.patch_is_packed_sequence:
        _patch_is_packed_sequence_for_training()
    if not fsdp2_sharding_enabled(context.mesh.device_mesh):
        return _parallelize_unsharded_fsdp2(model, context)

    if strategy is None:
        strategy = DefaultParallelizationStrategy()
    parallelize = getattr(strategy, "parallelize", None)
    if not callable(parallelize):
        name = type(strategy).__name__
        raise TypeError(f"ModelParallelizer.strategy must implement parallelize(model, device_mesh, ...); got {name}.")

    return parallelize(
        model=model,
        device_mesh=context.mesh.device_mesh,
        mp_policy=config.mp_policy,
        tp_shard_plan=config.tp_plan,
        offload_policy=config.offload_policy,
        sequence_parallel=config.sequence_parallel,
        activation_checkpointing=context.activation_checkpointing,
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


def _parallelize_unsharded_fsdp2(model: nn.Module, context: ParallelizeContext) -> nn.Module:
    from nemo_automodel.components.distributed.activation_checkpointing import (
        apply_submodule_checkpointing,
        detect_kv_sharing_and_maybe_disable_cache,
        is_selective_activation_checkpointing,
    )
    from nemo_automodel.components.distributed.parallelizer import (
        _extract_model_layer_groups,
        _filter_layer_groups_for_activation_checkpointing,
        _should_use_hf_native_gradient_checkpointing,
        apply_selective_activation_checkpointing,
    )

    config = context.strategy
    assert isinstance(config, FSDP2Config)

    if context.activation_checkpointing:
        if is_selective_activation_checkpointing(context.activation_checkpointing):
            apply_selective_activation_checkpointing(
                model,
                enable_compile=config.enable_compile,
                activation_checkpointing_scope=config.activation_checkpointing_scope,
            )
        else:
            layer_groups = _extract_model_layer_groups(model)
            layers, ac_scopes = _filter_layer_groups_for_activation_checkpointing(
                layer_groups,
                config.activation_checkpointing_scope,
            )
            if _should_use_hf_native_gradient_checkpointing(
                model,
                layer_groups,
                ac_scopes,
                enable_compile=config.enable_compile,
            ):
                model.gradient_checkpointing_enable()
            else:
                apply_submodule_checkpointing(layers, detect_kv_sharing_and_maybe_disable_cache(model))
    if context.reapply_trainability is not None:
        context.reapply_trainability(model)
    return model


def _parallelize_ddp(model: nn.Module, context: ParallelizeContext) -> nn.Module:
    from nemo_automodel.components.distributed.ddp import parallelize_ddp

    config = context.strategy
    assert isinstance(config, DDPConfig)
    return parallelize_ddp(model, config, reapply_trainability=context.reapply_trainability)


def _parallelize_megatron_fsdp(model: nn.Module, context: ParallelizeContext) -> nn.Module:
    from nemo_automodel.components.distributed.megatron_fsdp import parallelize_megatron_fsdp

    config = context.strategy
    assert isinstance(config, MegatronFSDPConfig)
    if context.mesh.device_mesh is None:
        raise ValueError("Megatron-FSDP parallelization requires context.mesh.device_mesh.")
    return parallelize_megatron_fsdp(
        model,
        config,
        context.mesh.device_mesh,
        reapply_trainability=context.reapply_trainability,
    )[0]


def compile_parallelized_model(model: nn.Module, context: ParallelizeContext) -> None:
    """Compile FSDP2 layers after parallelization when requested."""
    config = context.strategy
    if not isinstance(config, FSDP2Config) or context.mesh.device_mesh is None:
        return
    if config.enable_compile or (config.enable_async_tensor_parallel and context.mesh.device_mesh["tp"].size() > 1):
        from nemo_automodel.components.distributed.parallelizer import _apply_per_layer_compile

        _apply_per_layer_compile(model)


def _parallelize_moe(
    model: nn.Module,
    context: ParallelizeContext,
    *,
    strategy: ParallelizationStrategy | None = None,
) -> nn.Module:
    from nemo_automodel.components.moe.parallelizer import parallelize_model as parallelize_moe_model

    mesh = context.mesh
    if mesh.device_mesh is None or mesh.moe_mesh is None:
        raise ValueError("Expert parallelization requires both device_mesh and moe_mesh.")

    moe = context.moe or MoEParallelizerConfig()
    strategy_config = context.strategy
    if isinstance(strategy_config, FSDP2Config):
        mp_policy = moe.mp_policy if moe.mp_policy is not None else strategy_config.mp_policy
        tp_shard_plan = strategy_config.tp_plan
        sequence_parallel = strategy_config.sequence_parallel
        offload_policy = strategy_config.offload_policy
        reshard_after_forward = (
            strategy_config.reshard_after_forward
            if strategy_config.reshard_after_forward is not None
            else moe.reshard_after_forward
        )
        enable_async_tensor_parallel = strategy_config.enable_async_tensor_parallel
        activation_checkpointing_scope = strategy_config.activation_checkpointing_scope
        frozen_multimodal_sharding = strategy_config.multimodal.frozen_sharding
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
        parallelization_strategy=strategy,
        **mesh.parallelize_axis_kwargs(),
    )
    return model


__all__ = [
    "ModelParallelizer",
    "ParallelizeContext",
    "compile_parallelized_model",
    "get_model_parallelizer",
    "parallelize_model",
]
