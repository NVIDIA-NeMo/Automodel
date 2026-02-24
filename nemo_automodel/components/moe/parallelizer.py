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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor import Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from nemo_automodel.components.distributed.pipelining.hf_utils import get_text_module
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP, GroupedExpertsTE
from nemo_automodel.components.moe.layers import (
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


def _resolve_text_layer_container(model: nn.Module) -> nn.Module:
    """Return the module that owns transformer `layers` for block-wise iteration."""
    if hasattr(model, "layers") and model.layers is not None:
        return model

    # Try common nested containers first.
    for attr_name in ("backbone", "model", "language_model", "text_model", "text_decoder"):
        if hasattr(model, attr_name):
            nested = getattr(model, attr_name)
            if nested is None:
                continue
            if hasattr(nested, "layers") and nested.layers is not None:
                return nested
            nested_text = get_text_module(nested)
            if hasattr(nested_text, "layers") and nested_text.layers is not None:
                return nested_text

    # Fallback: search any nested submodule exposing `layers`.
    for _, submod in model.named_modules():
        if submod is model:
            continue
        if hasattr(submod, "layers") and submod.layers is not None:
            return submod

    child_names = list(model._modules.keys()) if hasattr(model, "_modules") else []
    has_backbone = hasattr(model, "backbone") and getattr(model, "backbone") is not None
    has_model_attr = hasattr(model, "model") and getattr(model, "model") is not None
    raise AttributeError(
        "Could not find a module with `layers` under "
        f"{type(model).__name__} (children={child_names[:24]}, has_backbone={has_backbone}, has_model={has_model_attr})"
    )


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
            dist_param.requires_grad = param.requires_grad
            module.register_parameter(name, dist_param)

        if isinstance(module, GroupedExpertsDeepEP):
            module.init_token_dispatcher(ep_mesh=device_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


def apply_ep(model: nn.Module, ep_mesh: DeviceMesh, moe_mesh: DeviceMesh | None = None):
    """Applies EP to MoE module."""
    assert ep_mesh.size() > 1

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    # Prefer nested text modules when present
    _model = get_text_module(_model)
    _layer_container = _resolve_text_layer_container(_model)

    for _, block in _layer_container.layers.named_children():
        moe_module = block.moe if hasattr(block, "moe") else block.mlp
        if isinstance(moe_module, MoE):
            # GroupedExpertsTEGroupedLinear uses TE's GroupedLinear which creates
            # local experts directly. It doesn't support DTensor wrapping, so we
            # skip distribute_module entirely and just initialize token dispatcher.
            if isinstance(moe_module.experts, GroupedExpertsTE):
                moe_module.experts.init_token_dispatcher(ep_mesh=ep_mesh, moe_mesh=moe_mesh)
            else:
                parallelize_module(
                    module=moe_module.experts,
                    device_mesh=ep_mesh,
                    parallelize_plan=ExpertParallel(),
                )


def apply_ac(
    model: nn.Module,
    ignore_router: bool = False,
    hidden_size: int | None = None,
    num_experts: int | None = None,
):
    """Apply activation checkpointing to the model.

    Args:
        model: The model to apply activation checkpointing to.
        ignore_router: If True, uses selective checkpointing that saves router outputs.
        hidden_size: Hidden dimension size. If None, derived from model.config.hidden_size.
        num_experts: Number of routed experts. If None, derived from moe_config.n_routed_experts
            first, then falls back to model.config attributes.
    """
    # Derive hidden_size and num_experts from model.config if not provided
    if hidden_size is None:
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
        else:
            raise ValueError("hidden_size must be provided or model must have config.hidden_size attribute")

    if num_experts is None:
        _inner = getattr(model, "model", model)
        if hasattr(_inner, "moe_config") and hasattr(_inner.moe_config, "n_routed_experts"):
            num_experts = _inner.moe_config.n_routed_experts
        else:
            for attr in ["num_experts", "moe_num_experts", "n_routed_experts"]:
                if hasattr(model, "config") and hasattr(model.config, attr):
                    num_experts = getattr(model.config, attr)
                    break
            else:
                raise ValueError("num_experts must be provided or model must have config.num_experts attribute")

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
    else:
        _model = model
    _model = get_text_module(_model)
    _layer_container = _resolve_text_layer_container(_model)
    for layer_id, block in _layer_container.layers.named_children():
        if ignore_router:
            block = ptd_checkpoint_wrapper(
                block, preserve_rng_state=True, context_fn=selective_checkpointing_context_fn
            )
        else:
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)

        _layer_container.layers.register_module(layer_id, block)


def apply_fsdp(
    model: torch.nn.Module,
    fsdp_mesh: DeviceMesh,
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
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
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
    else:
        _model = model
    # handle VLM
    _model = get_text_module(_model)
    _layer_container = _resolve_text_layer_container(_model)

    for _, block in _layer_container.layers.named_children():
        moe_module = block.moe if hasattr(block, "moe") else block.mlp
        if isinstance(moe_module, MoE) and ep_shard_enabled:
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
        if isinstance(moe_module, MoE) and ep_enabled:
            ignored_params = set(moe_module.experts.parameters())

        fully_shard_default(block, ignored_params=ignored_params)

    if hasattr(_layer_container, "embed_tokens") and _layer_container.embed_tokens is not None:
        fully_shard_default(_layer_container.embed_tokens)

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

    fully_shard_default(_layer_container)

    # If model has a nested structure (outer model wrapping inner _model), wrap the outer model if requested
    if wrap_outer_model and model is not _model:
        fully_shard_default(model)


def apply_cp(model: torch.nn.Module, cp_mesh: DeviceMesh, cp_comm_type: str = "p2p"):
    from transformer_engine.pytorch.attention import DotProductAttention

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    _model = get_text_module(_model)
    _layer_container = _resolve_text_layer_container(_model)

    for _, block in _layer_container.layers.named_children():
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
    pp_enabled: bool = False,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    activation_checkpointing: bool = False,
    ignore_router_for_ac: bool = False,
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

    ep_enabled = ep_axis_name is not None and moe_mesh is not None and moe_mesh[ep_axis_name].size() > 1
    if ep_enabled:
        _model = model.model if hasattr(model, "model") and model.model is not None else model
        _model = get_text_module(_model)
        n_routed_experts = None
        if hasattr(_model, "moe_config") and _model.moe_config is not None:
            n_routed_experts = getattr(_model.moe_config, "n_routed_experts", None)
        if n_routed_experts is None and hasattr(_model, "config"):
            for attr in ("n_routed_experts", "moe_num_experts", "num_experts"):
                if hasattr(_model.config, attr):
                    n_routed_experts = getattr(_model.config, attr)
                    break

        if n_routed_experts is not None:
            assert n_routed_experts % moe_mesh[ep_axis_name].size() == 0, (
                f"n_routed_experts {n_routed_experts} must be divisible by "
                f"expert_parallel_degree {moe_mesh[ep_axis_name].size()}"
            )
        else:
            logger.warning("Could not infer n_routed_experts; skipping EP divisibility assertion.")

        apply_ep(model, moe_mesh[ep_axis_name], moe_mesh=moe_mesh)

    if activation_checkpointing:
        apply_ac(model, ignore_router=ignore_router_for_ac)

    ep_shard_mesh = None
    if ep_shard_axis_names is not None:
        if moe_mesh is None:
            logger.warning(
                "ep_shard_axis_names=%s provided but moe_mesh is None; skipping ep_shard mesh setup.",
                ep_shard_axis_names,
            )
        else:
            ep_shard_mesh = moe_mesh[ep_shard_axis_names]

    fsdp_enabled = dp_axis_names is not None and world_mesh[dp_axis_names].size() > 1
    fsdp_mesh = world_mesh[tuple(dp_axis_names)] if fsdp_enabled else None
    if fsdp_enabled:
        apply_fsdp(
            model,
            fsdp_mesh,
            ep_enabled=ep_enabled,
            ep_shard_enabled=ep_shard_mesh is not None and ep_shard_mesh.size() > 1,
            ep_shard_mesh=ep_shard_mesh,
            reshard_after_forward=reshard_after_forward,
            lm_head_precision=lm_head_precision,
            wrap_outer_model=wrap_outer_model,
        )
