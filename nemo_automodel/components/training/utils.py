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

import gc
import math
import re
import weakref
from typing import Iterable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate

from nemo_automodel.components.models.common.utils import set_is_first_microbatch, set_is_optim_step

# Regex pattern to match expert parameters in GroupedExpertsTE.
# Matches FQNs like:
# - model.layers.X.mlp.experts.gate_up_linear.weight0
# - model.layers.X.mlp.experts.gate_up_linear.bias0
# - model.layers.X.mlp.experts.down_linear.weight0
# - model.layers.X.mlp.experts.down_linear.bias0
_TE_EXPERT_PARAM_PATTERN = re.compile(r"(^|\.)mlp\.experts\.(gate_up_linear|down_linear)\.(weight|bias)\d+")

# Model structure and trainability are stable during an optimizer step. Cache the
# expensive module parameter walks while keeping entries lifetime-bound to the module.
_TRAINABLE_PARAMETER_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_NAMED_PARAMETER_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_SHARDING_GROUP_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _parameter_sharding_key(parameter: torch.Tensor):
    if isinstance(parameter, DTensor):
        return id(parameter.device_mesh), tuple(str(placement) for placement in parameter.placements)
    return "regular"


def _get_cached_trainable_parameters(model_parts: list[torch.nn.Module]) -> list[torch.Tensor]:
    parameters = []
    for model_part in model_parts:
        cached = _TRAINABLE_PARAMETER_CACHE.get(model_part)
        if cached is None:
            cached = tuple(p for p in model_part.parameters() if p.requires_grad)
            _TRAINABLE_PARAMETER_CACHE[model_part] = cached
        parameters.extend(cached)
    return parameters


def _get_cached_named_parameters(model_parts: list[torch.nn.Module]) -> list[tuple[str, torch.Tensor]]:
    named_parameters = []
    for model_part in model_parts:
        cached = _NAMED_PARAMETER_CACHE.get(model_part)
        if cached is None:
            cached = tuple(model_part.named_parameters())
            _NAMED_PARAMETER_CACHE[model_part] = cached
        named_parameters.extend(cached)

    return named_parameters


def _get_cached_sharding_groups(model_parts: list[torch.nn.Module]) -> list[tuple[torch.Tensor, ...]]:
    groups = []
    for model_part in model_parts:
        cached = _SHARDING_GROUP_CACHE.get(model_part)
        if cached is None:
            grouped_parameters = {}
            for parameter in model_part.parameters():
                if parameter.requires_grad:
                    grouped_parameters.setdefault(_parameter_sharding_key(parameter), []).append(parameter)
            cached = tuple(tuple(parameters) for parameters in grouped_parameters.values())
            _SHARDING_GROUP_CACHE[model_part] = cached
        groups.extend(cached)
    return groups


def clear_grad_clip_parameter_cache(model_parts: list[torch.nn.Module] | None = None) -> None:
    """Clear cached clipping parameters after changing a model's trainable set."""
    if model_parts is None:
        _TRAINABLE_PARAMETER_CACHE.clear()
        _NAMED_PARAMETER_CACHE.clear()
        _SHARDING_GROUP_CACHE.clear()
        return
    for model_part in model_parts:
        _TRAINABLE_PARAMETER_CACHE.pop(model_part, None)
        _NAMED_PARAMETER_CACHE.pop(model_part, None)
        _SHARDING_GROUP_CACHE.pop(model_part, None)


@torch.no_grad()
def count_tail_padding(labels, ignore_label=-100):
    """Counts the total number of padding token in the tail of labels

    e.g.
        labels = torch.tensor([
            [-100, 1, 1, -100, -100],   # 2 tail -100s
            [-100, -100, 2, 3, 4],      # 0 tail -100s
            [5, 6, -100, -100, -100],   # 3 tail -100s
        ])
        count_tail_padding will return 5. Please do note there's more than 5 ignore labels.
    Args:
        labels (torch.Tensor): the labels
        ignore_label (int, optional): ignore label index. Defaults to -100.

    Returns:
        int: total number of ignored tokens in the `labels` input.
    """
    # Flip along the last dimension (seq_len)
    flipped = labels.flip(dims=[1])
    tail_mask = flipped == ignore_label

    # Compute cumulative product to "break" on first non ignore_label
    prod_mask = torch.cumprod(tail_mask.int(), dim=1)

    # Count tail -100s by summing cumprod mask along the sequence dimension
    return prod_mask.view(-1).sum().item()


@torch.no_grad()
def _clip_grad_norm_impl(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    precomputed_grad_norm: torch.Tensor | None = None,
    sharding_groups: Iterable[Iterable[torch.Tensor]] | None = None,
) -> torch.Tensor:
    # Determine target device for all tensor operations
    # Use current CUDA device if available, otherwise use CPU
    if torch.cuda.is_available():
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        target_device = torch.device("cpu")

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    if sharding_groups is None:
        # Group parameters by their sharding pattern when called directly. The main
        # clip_grad_norm path supplies cached groups to avoid this walk every step.
        grouped_parameters = {}
        for parameter in parameters:
            if parameter.grad is not None:
                grouped_parameters.setdefault(_parameter_sharding_key(parameter), []).append(parameter)
        sharding_groups = tuple(grouped_parameters.values())
    else:
        sharding_groups = tuple(sharding_groups)
    sharding_groups = tuple(
        tuple(parameter for parameter in group if parameter.grad is not None) for group in sharding_groups
    )
    sharding_groups = tuple(group for group in sharding_groups if group)

    if precomputed_grad_norm is not None:
        total_norm = precomputed_grad_norm.float().to(target_device)
    else:
        # Compute norm for each sharding group using a scalar-first reduction:
        # sum(|g_local|^p) locally → single-scalar allreduce per Shard mesh dim.
        # Going through torch.nn.utils.get_total_norm on DTensor grads would stack
        # per-param scalar DTensors into a 1-D DTensor whose local length equals
        # the number of local param tensors in the group. Under EP, that length
        # can differ across ranks, and the vector_norm redistribute (Partial →
        # Replicate) then allreduces with mismatched numel and hangs.
        is_inf = math.isinf(norm_type)
        group_norms = []
        for group_params in sharding_groups:
            first = group_params[0]
            is_dtensor = isinstance(first, DTensor)
            # Partial placements can't be reduced via sum-of-local-norms; reduce
            # the flattened local values before calculating the norm.
            has_partial = is_dtensor and any(isinstance(pl, Partial) for pl in first.placements)

            if has_partial:
                # A Partial DTensor needs its values reduced before its norm is
                # known. Flatten the whole placement group so this is one
                # collective per mesh dimension instead of one full_tensor()
                # collective per parameter.
                local_values = [p.grad.to_local().detach().float().reshape(-1) for p in group_params]
                flat_values = torch.cat(local_values)
                for dim_idx, placement in enumerate(first.placements):
                    if isinstance(placement, Partial):
                        torch.distributed.all_reduce(
                            flat_values,
                            op=torch.distributed.ReduceOp.SUM,
                            group=first.device_mesh.get_group(mesh_dim=dim_idx),
                        )
                if is_inf:
                    local_val = flat_values.abs().max()
                else:
                    local_val = flat_values.abs().pow(norm_type).sum()
            else:
                local_val = torch.zeros((), dtype=torch.float32, device=target_device)
                for p in group_params:
                    g = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                    g = g.detach().float()
                    if is_inf:
                        local_val = torch.maximum(local_val, g.abs().max())
                    else:
                        local_val = local_val + g.abs().pow(norm_type).sum()

            if is_dtensor:
                mesh = first.device_mesh
                op = torch.distributed.ReduceOp.MAX if is_inf else torch.distributed.ReduceOp.SUM
                for dim_idx, pl in enumerate(first.placements):
                    if isinstance(pl, (Replicate, Partial)):
                        continue
                    torch.distributed.all_reduce(local_val, op=op, group=mesh.get_group(mesh_dim=dim_idx))

            group_norms.append(local_val if is_inf else local_val.pow(1.0 / norm_type))

        # Combine norms across groups (all rank-identical scalars, no comm)
        if len(group_norms) == 0:
            total_norm = torch.tensor(0.0, device=target_device)
        elif len(group_norms) == 1:
            total_norm = group_norms[0]
        elif is_inf:
            total_norm = torch.stack(group_norms).max()
        else:
            total_norm = torch.zeros((), dtype=torch.float32, device=target_device)
            for gn in group_norms:
                total_norm = total_norm + gn.pow(norm_type)
            total_norm = total_norm.pow(1.0 / norm_type)

        total_norm = total_norm.float().to(target_device)
        # Reduce across pipeline parallel mesh if provided
        if pp_mesh is not None:
            if math.isinf(norm_type):
                torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm = total_norm**norm_type
                torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    # Scale DTensor gradients through their local tensors. The clip coefficient
    # is rank-identical, so this preserves Partial/Shard semantics without
    # triggering one redistribution collective per gradient.
    for group_params in sharding_groups:
        dtensor_grads = [parameter.grad.to_local() for parameter in group_params if isinstance(parameter.grad, DTensor)]
        if not dtensor_grads:
            torch.nn.utils.clip_grads_with_norm_(group_params, max_norm, total_norm, foreach)
            continue

        regular_params = [
            parameter
            for parameter in group_params
            if parameter.grad is not None and not isinstance(parameter.grad, DTensor)
        ]
        if regular_params:
            torch.nn.utils.clip_grads_with_norm_(regular_params, max_norm, total_norm, foreach)
        if foreach:
            grouped_grads = {}
            for grad in dtensor_grads:
                grouped_grads.setdefault((grad.device, grad.dtype), []).append(grad)
            for grads in grouped_grads.values():
                torch._foreach_mul_(grads, clip_coef.to(grads[0].device))
        else:
            for grad in dtensor_grads:
                grad.mul_(clip_coef.to(grad.device))

    return total_norm


@torch.no_grad()
def clip_grad_norm(
    max_grad_norm: float | None,
    model_parts: list[torch.nn.Module],
    *,
    norm_type: float = 2.0,
    pp_enabled: bool = False,
    device_mesh: DeviceMesh | None = None,
    pp_axis_name: str | None = None,
    foreach: bool = True,
    use_torch_clip_grad_norm: bool = False,
    precomputed_grad_norm: torch.Tensor | None = None,
):
    """Common gradient clipping helper.

    Handles all parallelism strategies (TP, PP, EP/MoE) with automatic sharding-aware grouping.
    Returns the gradient norm as a float, or 0.0 if clipping is skipped.

    This function automatically:
    - Groups parameters by sharding pattern (device mesh + placements)
    - Computes norms correctly across different sharding strategies
    - Handles MoE with separate DP/EP meshes
    - Reduces norms across pipeline parallel stages when enabled

    Args:
        max_grad_norm: Maximum gradient norm. If None, skips clipping.
        model_parts: List of model modules to clip.
        norm_type: Type of norm to use (default: 2.0 for L2).
        pp_enabled: Whether pipeline parallelism is enabled.
        device_mesh: Device mesh for parallelism.
        moe_mesh: MoE-specific device mesh (unused, kept for API compatibility).
        ep_axis_name: Expert parallel axis name (unused, kept for API compatibility).
        pp_axis_name: Pipeline parallel axis name.
        foreach: Whether to use foreach implementation for clipping.
        use_torch_clip_grad_norm: Use PyTorch's optimized regular-tensor clipping path when possible.
        precomputed_grad_norm: Optional total norm supplied by a distributed gradient reducer.

    Returns:
        Total gradient norm as a float.
    """
    if max_grad_norm is None:
        for model_part in model_parts:
            state = getattr(model_part, "_nemo_fused_grad_norm_state", None)
            if state is not None:
                state.reset()
        return 0.0

    # Reuse the stable parameter set; this avoids walking the full module tree every step.
    parameters = _get_cached_trainable_parameters(model_parts)
    fused_norm_states = [getattr(model_part, "_nemo_fused_grad_norm_state", None) for model_part in model_parts]
    fused_norm_states = [state for state in fused_norm_states if state is not None]
    if fused_norm_states:
        if precomputed_grad_norm is not None:
            raise ValueError("precomputed_grad_norm cannot be combined with an active DDP norm state")
        for state in fused_norm_states:
            state.wait()
        norm_sq = torch.zeros_like(fused_norm_states[0].norm_sq)
        for state in fused_norm_states:
            norm_sq.add_(state.norm_sq)
        precomputed_grad_norm = norm_sq.sqrt()

    # Determine pp_mesh if PP is enabled
    pp_mesh = None
    if pp_enabled:
        assert pp_axis_name is not None, "pp_axis_name must be provided when pp_enabled is True"
        pp_mesh = device_mesh[pp_axis_name] if device_mesh is not None else None

    can_use_torch_clip = (use_torch_clip_grad_norm or precomputed_grad_norm is not None) and pp_mesh is None
    if can_use_torch_clip:
        for p in parameters:
            if isinstance(p, DTensor) or isinstance(p.grad, DTensor):
                can_use_torch_clip = False
                break

    try:
        if precomputed_grad_norm is not None and can_use_torch_clip:
            grad_norm = precomputed_grad_norm
            torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, grad_norm, foreach)
        elif can_use_torch_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_grad_norm,
                norm_type=norm_type,
                error_if_nonfinite=False,
                foreach=foreach,
            )
        else:
            # Use the sharding-aware implementation for DTensor, PP, EP, and mixed placement cases.
            grad_norm = _clip_grad_norm_impl(
                parameters=parameters,
                max_norm=max_grad_norm,
                norm_type=norm_type,
                error_if_nonfinite=False,
                foreach=foreach,
                pp_mesh=pp_mesh,
                precomputed_grad_norm=precomputed_grad_norm,
                sharding_groups=_get_cached_sharding_groups(model_parts),
            )

        # Convert to float for API compatibility
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item() if grad_norm.numel() == 1 else grad_norm
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()

        return grad_norm
    finally:
        for state in fused_norm_states:
            state.reset()


def prepare_for_grad_accumulation(model_parts: list[torch.nn.Module], pp_enabled: bool = False):
    """Prepare model parts before starting gradient accumulation.

    This is typically called once at the start of gradient accumulation to prepare
    FSDP states for the upcoming forward and backward passes.

    Args:
        model_parts: List of model parts (modules) to prepare.
        pp_enabled: Whether pipeline parallelism is enabled.
    """
    set_is_optim_step(False)
    set_is_first_microbatch(True)
    if pp_enabled:
        return

    for mp in model_parts:
        if hasattr(mp, "prepare_for_grad_accumulation"):
            mp.prepare_for_grad_accumulation(pp_enabled=pp_enabled)


def prepare_after_first_microbatch():
    """Disable first-microbatch flag after the first forward-backward pass.

    Called after the first microbatch in gradient accumulation so that
    subsequent microbatches reuse cached FP8 weights instead of re-quantizing.
    """
    set_is_first_microbatch(False)


def prepare_for_final_backward(model_parts: list[torch.nn.Module], pp_enabled: bool = False):
    """Prepare model parts before the final backward pass.

    This is typically called before the final gradient accumulation step to prepare
    FSDP states for gradient synchronization and resharding.

    Args:
        model_parts: List of model parts (modules) to prepare.
        pp_enabled: Whether pipeline parallelism is enabled.
    """
    set_is_optim_step(True)
    if pp_enabled:
        return

    for mp in model_parts:
        if hasattr(mp, "prepare_for_final_backward"):
            mp.prepare_for_final_backward(pp_enabled=pp_enabled)


@torch.no_grad()
def scale_grads_and_clip_grad_norm(
    max_grad_norm: float | None,
    model_parts: list[torch.nn.Module],
    *,
    norm_type: float = 2.0,
    pp_enabled: bool = False,
    device_mesh: DeviceMesh | None = None,
    moe_mesh: DeviceMesh | None = None,
    ep_axis_name: str | None = None,
    pp_axis_name: str | None = None,
    foreach: bool = True,
    num_label_tokens: int | None = None,
    dp_group_size: int | None = None,
    use_torch_clip_grad_norm: bool = False,
    precomputed_grad_norm: torch.Tensor | None = None,
):
    """Scale gradients for PP/EP in a single pass, then clip.

    - PP scaling: divide all local grads by (num_label_tokens / dp_group_size).
    - EP scaling: for parameters on the expert axis, divide grads by (dp_group_size / ep_shard_size).
    - Finally, perform grad clipping with PP/EP-aware reductions.
    """

    # Precompute scale factors
    pp_divisor: float | None = None
    if pp_enabled and num_label_tokens is not None and dp_group_size is not None:
        if dp_group_size != 0:
            candidate = num_label_tokens / dp_group_size
            pp_divisor = float(candidate) if candidate != 0 else None

    ep_ratio: float | None = None
    if moe_mesh is not None and dp_group_size is not None:
        ep_shard_size = moe_mesh["ep_shard"].size() if "ep_shard" in moe_mesh.mesh_dim_names else 1
        if ep_shard_size > 0:
            ep_ratio = float(dp_group_size) / float(ep_shard_size)

    # Single pass over parameters to apply both scalings where applicable
    if pp_divisor is not None or ep_ratio is not None:
        for name, p in _get_cached_named_parameters(model_parts):
            if p.grad is None:
                continue
            if pp_divisor is not None:
                p.grad.div_(pp_divisor)
            if ep_ratio is not None:
                # Scale expert gradients by EP ratio.
                # DTensor experts: check device mesh for EP sharding axis
                # Non-DTensor experts (e.g., DeepEP): check param name
                is_ep_sharded_dtensor = (
                    isinstance(p, DTensor)
                    and isinstance(p.grad, DTensor)
                    and ep_axis_name
                    and ep_axis_name in p.device_mesh.mesh_dim_names
                )
                is_expert_param = (
                    isinstance(p, torch.Tensor)
                    and isinstance(p.grad, torch.Tensor)
                    and _TE_EXPERT_PARAM_PATTERN.search(name) is not None
                )
                if is_ep_sharded_dtensor or is_expert_param:
                    p.grad.div_(ep_ratio)

    # Clip with the existing PP/EP-aware helper
    return clip_grad_norm(
        max_grad_norm,
        model_parts,
        norm_type=norm_type,
        pp_enabled=pp_enabled,
        device_mesh=device_mesh,
        pp_axis_name=pp_axis_name,
        foreach=foreach,
        use_torch_clip_grad_norm=use_torch_clip_grad_norm,
        precomputed_grad_norm=precomputed_grad_norm,
    )


def move_to_device(model, device):
    """Move a model and its buffers to a device and release stale CUDA cache."""
    # FSDP modules do not move buffers to the device automatically
    for v in model.buffers():
        v.data = v.data.to(device)
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()


class ScopedModuleOffloading:
    """Context manager that temporarily moves a module between CPU and CUDA."""

    def __init__(self, model, enabled=False):
        self.model = model
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            move_to_device(self.model, "cuda")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            move_to_device(self.model, "cpu")
        return False  # Re-raise exceptions by default
