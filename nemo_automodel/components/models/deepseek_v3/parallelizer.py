# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import logging

import torch
import torch.nn as nn
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor import Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3Model
from nemo_automodel.components.moe.layers import (
    GroupedExpertsDeepEP,
    MoE,
)

logger = logging.getLogger(__name__)


class ExpertParallel(ParallelStyle):
    """
    ExpertParallel class is used to shard the MoE parameters on the EP mesh.
    Dim `0` of each parameter is sharded since that is the expert dimension.
    """

    def _partition_fn(self, name, module, device_mesh):
        # shard on the expert dimension
        assert device_mesh.ndim == 1

        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, dist_param)

        if isinstance(module, GroupedExpertsDeepEP):
            module.init_token_dispatcher(ep_mesh=device_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


def apply_ep(model: nn.Module, ep_mesh: DeviceMesh):
    """Applies EP to MoE module."""
    assert ep_mesh.size() > 1

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model

    for _, block in _model.layers.named_children():
        if isinstance(block.mlp, MoE):
            parallelize_module(
                module=block.mlp.experts,
                device_mesh=ep_mesh,
                parallelize_plan=ExpertParallel(),
            )


def apply_ac(model: nn.Module, ignore_router: bool = False):
    """Apply activation checkpointing to the model."""

    def _custom_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            if len(args) == 2 and (args[1].shape == (7168, 256) or args[1].shape == (2048, 64)):
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_checkpointing_context_fn():
        return create_selective_checkpoint_contexts(_custom_policy)

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    for layer_id, block in _model.layers.named_children():
        if ignore_router:
            block = ptd_checkpoint_wrapper(
                block, preserve_rng_state=True, context_fn=selective_checkpointing_context_fn
            )
        else:
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)

        _model.layers.register_module(layer_id, block)


def apply_fsdp(
    model: DeepseekV3Model,
    fsdp_mesh: DeviceMesh,
    pp_enabled: bool,
    ep_enabled: bool,
    ep_shard_enabled: bool,
    ep_shard_mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
):
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16
        )

    fully_shard_default = functools.partial(
        fully_shard,
        mesh=fsdp_mesh,
        reshard_after_forward=not pp_enabled,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model

    for _, block in _model.layers.named_children():
        if isinstance(block.mlp, MoE) and ep_shard_enabled:
            # Apply FSDP on dim=1 for grouped experts since we may have more
            # shards than experts (dim=0).
            fully_shard(
                block.mlp.experts,
                mesh=ep_shard_mesh,
                shard_placement_fn=lambda _: Shard(1),
                reshard_after_forward=not pp_enabled,
            )
        # If FSDP is disabled for grouped experts because the parameters are already
        # fully sharded by PP and EP, then we need to explicitly remove the parameters
        # from FSDP for the transformer block.
        # If FSDP is enabled for grouped experts, the parameters are automatically
        # removed from the FSDP for the transformer block due to the rules of the
        # PyTorch FSDP implementation.
        ignored_params = None
        if isinstance(block.mlp, MoE) and ep_enabled:
            ignored_params = set(block.mlp.experts.parameters())

        fully_shard_default(block, ignored_params=ignored_params)

    if hasattr(_model, "embed_tokens") and _model.embed_tokens is not None:
        fully_shard_default(_model.embed_tokens)

    lm_head = getattr(_model, "lm_head", None) or getattr(model, "lm_head", None)
    if lm_head is not None:
        fully_shard_default(lm_head)

    fully_shard_default(_model)


def parallelize_model(
    model: DeepseekV3Model,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    *,
    pp_enabled: bool,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
):
    assert tp_axis_name is None or world_mesh[tp_axis_name].size() == 1, (
        "Tensor parallelism not supported for DeepSeek v3 model"
    )
    assert cp_axis_name is None or world_mesh[cp_axis_name].size() == 1, (
        "Context parallelism not supported for DeepSeek v3 model"
    )

    # TODO: Add support for context parallelism
    # if cp_enabled:
    #     apply_cp(model, meshes["default"]["cp"], parallel_dims)

    ep_enabled = ep_axis_name is not None and moe_mesh is not None and moe_mesh[ep_axis_name].size() > 1
    if ep_enabled:
        assert model.config.n_routed_experts % moe_mesh[ep_axis_name].size() == 0, (
            f"n_routed_experts {model.config.n_routed_experts} must be divisible by "
            f"expert_parallel_degree {moe_mesh[ep_axis_name].size()}"
        )

        apply_ep(model, moe_mesh[ep_axis_name])

    apply_ac(model)

    if ep_shard_axis_names is not None:
        ep_shard_mesh = moe_mesh[ep_shard_axis_names]
    else:
        ep_shard_mesh = None

    fsdp_enabled = (
        dp_axis_names is not None
        and all(map(lambda x: x in world_mesh.mesh_dim_names, dp_axis_names))
        and world_mesh[dp_axis_names].size() > 1
    )
    fsdp_mesh = world_mesh[tuple(dp_axis_names)] if fsdp_enabled else None
    if fsdp_enabled:
        apply_fsdp(
            model,
            fsdp_mesh,
            pp_enabled=pp_enabled,
            ep_enabled=ep_enabled,
            ep_shard_enabled=ep_shard_mesh is not None and ep_shard_mesh.size() > 1,
            ep_shard_mesh=ep_shard_mesh,
        )
