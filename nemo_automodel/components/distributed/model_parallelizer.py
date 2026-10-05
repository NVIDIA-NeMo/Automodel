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

from typing import TYPE_CHECKING

from torch import nn

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
    MoEParallelizerConfig,
)
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.mesh import MeshContext


_DEFAULT_PARALLELIZER = ModelParallelizer()


def get_model_parallelizer(model: nn.Module) -> ModelParallelizer:
    """Return the class-owned sidecar, or the shared default implementation."""
    parallelizer = getattr(type(model), "parallelizer", None)
    if parallelizer is None:
        return _DEFAULT_PARALLELIZER
    if not callable(getattr(parallelizer, "parallelize", None)):
        name = type(parallelizer).__name__
        raise TypeError(
            f"{type(model).__name__}.parallelizer must implement parallelize(model, mesh_context); got {name}."
        )
    return parallelizer


def parallelize_model(model: nn.Module, mesh_context: MeshContext) -> nn.Module:
    """Apply all requested parallelisms through the model-owned contract."""
    return get_model_parallelizer(model).parallelize(model, mesh_context)


def _apply_model_parallelizer(
    parallelizer: ModelParallelizer,
    model: nn.Module,
    mesh_context: MeshContext,
) -> nn.Module:
    """Execute shared strategy dispatch for one model-owned parallelizer."""
    if mesh_context.ep_size > 1:
        return _parallelize_moe(model, mesh_context, parallelizer=parallelizer)
    if isinstance(mesh_context.strategy_config, FSDP2Config):
        return _parallelize_fsdp2(model, mesh_context, parallelizer=parallelizer)
    if isinstance(mesh_context.strategy_config, DDPConfig):
        return _parallelize_ddp(model, mesh_context)
    if isinstance(mesh_context.strategy_config, MegatronFSDPConfig):
        return _parallelize_megatron_fsdp(model, mesh_context)
    name = type(mesh_context.strategy_config).__name__
    raise TypeError(f"ModelParallelizer does not support strategy={name}.")


def _parallelize_fsdp2(
    model: nn.Module,
    mesh_context: MeshContext,
    *,
    parallelizer: ModelParallelizer,
) -> nn.Module:
    from nemo_automodel.components.distributed.fsdp2 import (
        _patch_is_packed_sequence_for_training,
        fsdp2_sharding_enabled,
    )

    config = mesh_context.strategy_config
    assert isinstance(config, FSDP2Config)
    if mesh_context.device_mesh is None:
        raise ValueError("FSDP2 parallelization requires mesh_context.device_mesh.")

    if config.patch_is_packed_sequence:
        _patch_is_packed_sequence_for_training()
    if not fsdp2_sharding_enabled(mesh_context.device_mesh):
        return _parallelize_unsharded_fsdp2(model, mesh_context, parallelizer=parallelizer)

    return parallelizer._apply(
        model=model,
        device_mesh=mesh_context.device_mesh,
        mp_policy=config.mp_policy,
        tp_shard_plan=config.tp_plan,
        offload_policy=config.offload_policy,
        sequence_parallel=config.sequence_parallel,
        activation_checkpointing=mesh_context.activation_checkpointing,
        enable_async_tensor_parallel=config.enable_async_tensor_parallel,
        enable_compile=config.enable_compile,
        enable_fsdp2_prefetch=config.enable_fsdp2_prefetch,
        fsdp2_backward_prefetch_depth=config.fsdp2_backward_prefetch_depth,
        fsdp2_forward_prefetch_depth=config.fsdp2_forward_prefetch_depth,
        reshard_after_forward=config.reshard_after_forward,
        activation_checkpointing_scope=config.activation_checkpointing_scope,
        frozen_multimodal_sharding=config.multimodal.frozen_sharding,
        reapply_trainability=mesh_context.reapply_trainability,
    )


def _parallelize_unsharded_fsdp2(
    model: nn.Module,
    mesh_context: MeshContext,
    *,
    parallelizer: ModelParallelizer,
) -> nn.Module:
    from nemo_automodel.components.distributed.activation_checkpointing import (
        apply_full_layer_checkpointing_to_layers,
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

    config = mesh_context.strategy_config
    assert isinstance(config, FSDP2Config)

    if mesh_context.activation_checkpointing:
        if is_selective_activation_checkpointing(mesh_context.activation_checkpointing):
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
            use_hf_native_checkpointing = _should_use_hf_native_gradient_checkpointing(
                model,
                layer_groups,
                ac_scopes,
                enable_compile=config.enable_compile,
            )
            if use_hf_native_checkpointing:
                model.gradient_checkpointing_enable()
            elif parallelizer._use_full_layer_activation_checkpointing(model):
                apply_full_layer_checkpointing_to_layers(model, layers)
            else:
                apply_submodule_checkpointing(layers, detect_kv_sharing_and_maybe_disable_cache(model))
    if mesh_context.reapply_trainability is not None:
        mesh_context.reapply_trainability(model)
    return model


def _parallelize_ddp(model: nn.Module, mesh_context: MeshContext) -> nn.Module:
    from nemo_automodel.components.distributed.ddp import parallelize_ddp

    config = mesh_context.strategy_config
    assert isinstance(config, DDPConfig)
    return parallelize_ddp(model, config, reapply_trainability=mesh_context.reapply_trainability)


def _parallelize_megatron_fsdp(model: nn.Module, mesh_context: MeshContext) -> nn.Module:
    from nemo_automodel.components.distributed.megatron_fsdp import parallelize_megatron_fsdp

    config = mesh_context.strategy_config
    assert isinstance(config, MegatronFSDPConfig)
    if mesh_context.device_mesh is None:
        raise ValueError("Megatron-FSDP parallelization requires mesh_context.device_mesh.")
    return parallelize_megatron_fsdp(
        model,
        config,
        mesh_context.device_mesh,
        reapply_trainability=mesh_context.reapply_trainability,
    )[0]


def compile_parallelized_model(model: nn.Module, mesh_context: MeshContext) -> None:
    """Compile FSDP2 layers after parallelization when requested."""
    config = mesh_context.strategy_config
    if not isinstance(config, FSDP2Config) or mesh_context.device_mesh is None:
        return
    if config.enable_compile or (config.enable_async_tensor_parallel and mesh_context.device_mesh["tp"].size() > 1):
        from nemo_automodel.components.distributed.parallelizer import _apply_per_layer_compile

        _apply_per_layer_compile(model)


def _parallelize_moe(
    model: nn.Module,
    mesh_context: MeshContext,
    *,
    parallelizer: ModelParallelizer,
) -> nn.Module:
    from nemo_automodel.components.moe.parallelizer import parallelize_model as parallelize_moe_model

    if mesh_context.device_mesh is None or mesh_context.moe_mesh is None:
        raise ValueError("Expert parallelization requires both device_mesh and moe_mesh.")

    moe = mesh_context.moe_parallel_config or MoEParallelizerConfig()
    strategy_config = mesh_context.strategy_config
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
        enable_fsdp2_prefetch = strategy_config.enable_fsdp2_prefetch and mesh_context.pp_size == 1
        fsdp2_backward_prefetch_depth = strategy_config.fsdp2_backward_prefetch_depth
        fsdp2_forward_prefetch_depth = strategy_config.fsdp2_forward_prefetch_depth
        activation_checkpointing_scope = strategy_config.activation_checkpointing_scope
        frozen_multimodal_sharding = strategy_config.multimodal.frozen_sharding
    else:
        mp_policy = moe.mp_policy
        tp_shard_plan = None
        sequence_parallel = False
        offload_policy = None
        reshard_after_forward = moe.reshard_after_forward
        enable_async_tensor_parallel = False
        enable_fsdp2_prefetch = False
        fsdp2_backward_prefetch_depth = 2
        fsdp2_forward_prefetch_depth = 1
        activation_checkpointing_scope = "all"
        frozen_multimodal_sharding = "root"

    parallelize_moe_model(
        model,
        world_mesh=mesh_context.device_mesh,
        moe_mesh=mesh_context.moe_mesh,
        activation_checkpointing=mesh_context.activation_checkpointing,
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
        enable_fsdp2_prefetch=enable_fsdp2_prefetch,
        fsdp2_backward_prefetch_depth=fsdp2_backward_prefetch_depth,
        fsdp2_forward_prefetch_depth=fsdp2_forward_prefetch_depth,
        frozen_multimodal_sharding=frozen_multimodal_sharding,
        reapply_trainability=mesh_context.reapply_trainability,
        model_parallelizer=parallelizer if parallelizer._customizes_moe_fsdp else None,
        **mesh_context.parallelize_axis_kwargs(),
    )
    return model


__all__ = [
    "ModelParallelizer",
    "compile_parallelized_model",
    "get_model_parallelizer",
    "parallelize_model",
]
