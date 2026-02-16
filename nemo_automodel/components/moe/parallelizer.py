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

from nemo_automodel.components.moe.layers import (
    GroupedExpertsDeepEP,
    MoE,
)
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)
_CP_STREAM = None


def _get_cp_stream() -> torch.cuda.Stream:
    global _CP_STREAM
    if _CP_STREAM is None:
        _CP_STREAM = torch.cuda.Stream()
    return _CP_STREAM


class ExpertParallel(ParallelStyle):
    """
    ExpertParallel class is used to shard the MoE parameters on the EP mesh.
    Dim `0` of each parameter is sharded since that is the expert dimension.
    """

    def _partition_fn(self, name, module, device_mesh):
        # shard on the expert dimension
        assert device_mesh.ndim == 1
        
        logger.info(f"[ExpertParallel._partition_fn] Called with module type: {type(module).__name__}")

        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, dist_param)

        if isinstance(module, GroupedExpertsDeepEP):
            logger.info(f"[ExpertParallel._partition_fn] Calling init_token_dispatcher on GroupedExpertsDeepEP")
            module.init_token_dispatcher(ep_mesh=device_mesh)
            logger.info(f"[ExpertParallel._partition_fn] init_token_dispatcher completed")
        else:
            logger.warning(f"[ExpertParallel._partition_fn] Module is NOT GroupedExpertsDeepEP, it's {type(module)}")

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


def apply_ep(model: nn.Module, ep_mesh: DeviceMesh):
    """Applies EP to MoE module."""
    assert ep_mesh.size() >= 1

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
        logger.info(f"[apply_ep] Resolved inner model via model.model ({type(_model).__name__})")
    elif hasattr(model, "backbone") and model.backbone is not None:
        _model = model.backbone
        logger.info(f"[apply_ep] Resolved inner model via model.backbone ({type(_model).__name__})")
    else:
        _model = model
        logger.info(f"[apply_ep] Using model directly as inner model ({type(_model).__name__})")

    n_moe = 0
    n_total = 0
    for name, block in _model.layers.named_children():
        n_total += 1
        # Support naming conventions: mlp (DeepSeek), block_sparse_moe (Mixtral), mixer (NemotronH)
        moe_module = None
        if hasattr(block, 'mlp') and isinstance(block.mlp, MoE):
            moe_module = block.mlp
            logger.info(f"[apply_ep] Block {name}: found MoE via block.mlp ({type(block.mlp).__name__})")
        elif hasattr(block, 'block_sparse_moe') and hasattr(block.block_sparse_moe, 'moe_layer'):
            if isinstance(block.block_sparse_moe.moe_layer, MoE):
                moe_module = block.block_sparse_moe.moe_layer
                logger.info(f"[apply_ep] Block {name}: found MoE via block.block_sparse_moe.moe_layer")
        elif hasattr(block, 'mixer') and isinstance(block.mixer, MoE):
            moe_module = block.mixer
            logger.info(f"[apply_ep] Block {name}: found MoE via block.mixer ({type(block.mixer).__name__})")
        else:
            block_type = getattr(block, 'block_type', type(block).__name__)
            logger.debug(f"[apply_ep] Block {name}: no MoE found (block_type={block_type})")
        if moe_module is not None:
            n_moe += 1
            if ep_mesh.size() == 1:
                # EP "init-only" mode: we still need DeepEP's token dispatcher and EP group,
                # but we do not need to DTensor-shard parameters when the EP axis is size 1.
                n_inited = 0
                for m in moe_module.experts.modules():
                    if isinstance(m, GroupedExpertsDeepEP):
                        m.init_token_dispatcher(ep_mesh=ep_mesh)
                        n_inited += 1
                logger.info(f"[apply_ep] EP mesh size=1; initialized DeepEP token dispatcher for {n_inited} module(s)")
            else:
                logger.info(f"[apply_ep] Calling parallelize_module on experts, calling init_token_dispatcher...")
                parallelize_module(
                    module=moe_module.experts,
                    device_mesh=ep_mesh,
                    parallelize_plan=ExpertParallel(),
                )
                logger.info(f"[apply_ep] Experts parallelized successfully")


def apply_ac(model: nn.Module, ignore_router: bool = False, hidden_size: int = 7168, num_experts: int = 256):
    """Apply activation checkpointing to the model."""

    def _custom_policy(ctx, func, *args, **kwargs):
        if func == torch.ops.aten.mm.default:
            if len(args) == 2 and (args[1].shape == (hidden_size, num_experts)):
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_checkpointing_context_fn():
        return create_selective_checkpoint_contexts(_custom_policy)

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    elif hasattr(model, "backbone") and model.backbone is not None:
        _model = model.backbone
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
    model: torch.nn.Module,
    fsdp_mesh: DeviceMesh,
    pp_enabled: bool,
    ep_enabled: bool,
    ep_shard_enabled: bool,
    ep_shard_mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
):
    if isinstance(lm_head_precision, str):
        lm_head_precision = dtype_from_str(lm_head_precision, default=None)

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16
        )

    fully_shard_default = functools.partial(
        fully_shard,
        mesh=fsdp_mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
        logger.info(f"[apply_fsdp] Resolved inner model via model.model ({type(_model).__name__})")
    elif hasattr(model, "backbone") and model.backbone is not None:
        _model = model.backbone
        logger.info(f"[apply_fsdp] Resolved inner model via model.backbone ({type(_model).__name__})")
    else:
        _model = model
        logger.info(f"[apply_fsdp] Using model directly as inner model ({type(_model).__name__})")

    n_moe_fsdp = 0
    n_ignored = 0
    for name, block in _model.layers.named_children():
        # Supports naming conventions: mlp (DeepSeek), block_sparse_moe (Mixtral), mixer (Nemotron)
        moe_module = None
        if hasattr(block, 'mlp') and isinstance(block.mlp, MoE):
            moe_module = block.mlp
        elif hasattr(block, 'block_sparse_moe') and hasattr(block.block_sparse_moe, 'moe_layer'):
            if isinstance(block.block_sparse_moe.moe_layer, MoE):
                moe_module = block.block_sparse_moe.moe_layer
        elif hasattr(block, 'mixer') and isinstance(block.mixer, MoE):
            moe_module = block.mixer
        
        if moe_module is not None and ep_shard_enabled:
            # Apply FSDP on dim=1 for grouped experts since we may have more
            # shards than experts (dim=0).
            fully_shard(
                moe_module.experts,
                mesh=ep_shard_mesh,
                shard_placement_fn=lambda _: Shard(1),
                reshard_after_forward=reshard_after_forward,
            )
        # If FSDP is disabled for grouped experts because the parameters are already
        # fully sharded by PP and EP, then we need to explicitly remove the parameters
        # from FSDP for the transformer block.
        # If FSDP is enabled for grouped experts, the parameters are automatically
        # removed from the FSDP for the transformer block due to the rules of the
        # PyTorch FSDP implementation.
        ignored_params = None
        if moe_module is not None and ep_enabled:
            ignored_params = set(moe_module.experts.parameters())

        fully_shard_default(block, ignored_params=ignored_params)

    if hasattr(_model, "embed_tokens") and _model.embed_tokens is not None:
        logger.info(f"[apply_fsdp] FSDP wrapping embed_tokens")
        fully_shard_default(_model.embed_tokens)
    elif hasattr(_model, "embeddings") and _model.embeddings is not None:
        logger.info(f"[apply_fsdp] FSDP wrapping embeddings (NemotronH path)")
        fully_shard_default(_model.embeddings)
    else:
        logger.warning(f"[apply_fsdp] No embeddings found (checked embed_tokens and embeddings)")

    lm_head = getattr(_model, "lm_head", None) or getattr(model, "lm_head", None)
    if lm_head is not None:
        # Use custom mixed precision policy for lm_head if lm_head_precision is specified
        if lm_head_precision == torch.float32:
            lm_head_mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )
            fully_shard(
                lm_head,
                mesh=fsdp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=lm_head_mp_policy,
                offload_policy=offload_policy,
            )
        else:
            fully_shard_default(lm_head)

    # TODO: properly handle all possible multimodal component names
    if hasattr(model, "audio_tower") and model.audio_tower is not None:
        if any(param.requires_grad for param in model.audio_tower.parameters()):
            fully_shard_default(model.audio_tower)
        else:
            logging.info("Skipping FSDP wrap for frozen audio tower")

    if hasattr(model, "visual") and model.visual is not None:
        if any(param.requires_grad for param in model.visual.parameters()):
            fully_shard_default(model.visual)
        else:
            logging.info("Skipping FSDP wrap for frozen visual tower")

    fully_shard_default(_model)

    # If model has a nested structure (outer model wrapping inner _model), wrap the outer model if requested
    if wrap_outer_model and model is not _model:
        fully_shard_default(model)


def apply_cp(model: torch.nn.Module, cp_mesh: DeviceMesh, cp_comm_type: str = "p2p"):
    from transformer_engine.pytorch.attention import DotProductAttention

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    elif hasattr(model, "backbone") and model.backbone is not None:
        _model = model.backbone
    else:
        _model = model

    for _, block in _model.layers.named_children():
        attn_module = block.self_attn.attn_module
        assert isinstance(attn_module, DotProductAttention), (
            "Context parallelism is only supported for TransformerEngine's DotProductAttention"
        )
        attn_module.set_context_parallel_group(
            cp_mesh.get_group(),
            torch.distributed.get_process_group_ranks(cp_mesh.get_group()),
            _get_cp_stream(),
            cp_comm_type=cp_comm_type,
        )


def parallelize_model(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    *,
    pp_enabled: bool,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    activation_checkpointing: bool = False,
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
):
    assert tp_axis_name is None or world_mesh[tp_axis_name].size() == 1, (
        "Tensor parallelism not supported for custom MoE models"
    )

    cp_enabled = cp_axis_name is not None and world_mesh[cp_axis_name].size() > 1
    if cp_enabled:
        apply_cp(model, world_mesh[cp_axis_name])

    ep_mesh = None
    if ep_axis_name is not None and moe_mesh is not None and ep_axis_name in moe_mesh.mesh_dim_names:
        ep_mesh = moe_mesh[ep_axis_name]

    # Always apply EP initialization if an EP mesh is available, even when ep_size == 1.
    # DeepEP requires the EP process group / token dispatcher even in the non-sharded case.
    if ep_mesh is not None:
        if ep_mesh.size() > 1:
            if hasattr(model, "model") and model.model is not None:
                _model = model.model
            elif hasattr(model, "backbone") and model.backbone is not None:
                _model = model.backbone
                logger.info("back bone acquired")
            else:
                _model = model
            assert _model.moe_config.n_routed_experts % ep_mesh.size() == 0, (
            f"n_routed_experts {model.model.moe_config.n_routed_experts} must be divisible by "
            f"expert_parallel_degree {ep_mesh.size()}"
            )
        apply_ep(model, ep_mesh)

    # "Sharded EP" toggle (used for FSDP interaction/ignored_params) remains size>1.
    ep_enabled = ep_mesh is not None and ep_mesh.size() > 1

    if activation_checkpointing:
        apply_ac(model)

    if ep_shard_axis_names is not None:
        ep_shard_mesh = moe_mesh[ep_shard_axis_names]
    else:
        ep_shard_mesh = None

    fsdp_enabled = dp_axis_names is not None and world_mesh[dp_axis_names].size() > 1
    fsdp_mesh = world_mesh[tuple(dp_axis_names)] if fsdp_enabled else None
    if fsdp_enabled:
        apply_fsdp(
            model,
            fsdp_mesh,
            pp_enabled=pp_enabled,
            ep_enabled=ep_enabled,
            ep_shard_enabled=ep_shard_mesh is not None and ep_shard_mesh.size() > 1,
            ep_shard_mesh=ep_shard_mesh,
            reshard_after_forward=reshard_after_forward,
            lm_head_precision=lm_head_precision,
            wrap_outer_model=wrap_outer_model,
        )
