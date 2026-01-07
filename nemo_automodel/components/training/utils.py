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
from typing import Iterable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.fsdp_mixin import set_is_optim_step


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

    # Group parameters by their sharding pattern
    # Key: (device_mesh_id, tuple of placements)
    sharding_groups = {}

    for p in parameters:
        if p.grad is None:
            continue

        if isinstance(p, DTensor):
            # Create a hashable key from device_mesh and placements
            mesh_id = id(p.device_mesh)
            placements_tuple = tuple(str(placement) for placement in p.placements)
            key = (mesh_id, placements_tuple)
        else:
            # Regular tensor - group separately
            key = ("regular", "regular")

        if key not in sharding_groups:
            sharding_groups[key] = []
        sharding_groups[key].append(p)

    # Compute norm for each sharding group
    group_norms = []
    for group_params in sharding_groups.values():
        grads = [p.grad for p in group_params]
        group_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
        group_norm = group_norm.float().to(target_device)

        # Convert DTensor norms to regular tensors and ensure they're on the same device
        if isinstance(group_norm, DTensor):
            group_norm = group_norm.full_tensor()

        # Ensure the norm is a regular tensor by cloning and detaching
        # This removes any DTensor metadata that might cause issues
        group_norm = group_norm.clone().detach()

        group_norms.append(group_norm)

    # Combine norms across groups
    if len(group_norms) == 0:
        total_norm = torch.tensor(0.0, device=target_device)
    elif len(group_norms) == 1:
        total_norm = group_norms[0]
    else:
        # Ensure all group norms are on the same device (use the first one's device)
        group_norms = [gn.to(target_device) if gn.device != target_device else gn for gn in group_norms]

        if math.isinf(norm_type):
            # For inf norm, take the maximum across groups
            total_norm = torch.stack(group_norms).max()
        else:
            # For p-norm, combine as (sum of p-th powers)^(1/p)
            total_norm = torch.tensor(0.0, device=target_device)
            for gn in group_norms:
                total_norm += gn**norm_type
            total_norm = total_norm ** (1.0 / norm_type)

    total_norm = total_norm.float().to(target_device)
    # Reduce across pipeline parallel mesh if provided
    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm = total_norm**norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm = total_norm ** (1.0 / norm_type)

    # Clip gradients for each sharding group separately
    # This is necessary because clip_grads_with_norm_ doesn't support mixing tensors from different device meshes
    for group_params in sharding_groups.values():
        torch.nn.utils.clip_grads_with_norm_(group_params, max_norm, total_norm, foreach)

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

    Returns:
        Total gradient norm as a float.
    """
    if max_grad_norm is None:
        return 0.0

    # Collect all parameters
    parameters = [p for m in model_parts for p in m.parameters() if p.requires_grad]

    # Determine pp_mesh if PP is enabled
    pp_mesh = None
    if pp_enabled:
        assert pp_axis_name is not None, "pp_axis_name must be provided when pp_enabled is True"
        pp_mesh = device_mesh[pp_axis_name] if device_mesh is not None else None

    # Use the new sharding-aware implementation
    grad_norm = _clip_grad_norm_impl(
        parameters=parameters,
        max_norm=max_grad_norm,
        norm_type=norm_type,
        error_if_nonfinite=False,
        foreach=foreach,
        pp_mesh=pp_mesh,
    )

    # Convert to float for API compatibility
    if isinstance(grad_norm, torch.Tensor):
        grad_norm = grad_norm.item() if grad_norm.numel() == 1 else grad_norm
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

    return grad_norm


def prepare_for_grad_accumulation(model_parts: list[torch.nn.Module], pp_enabled: bool = False):
    """Prepare model parts before starting gradient accumulation.

    This is typically called once at the start of gradient accumulation to prepare
    FSDP states for the upcoming forward and backward passes.

    Args:
        model_parts: List of model parts (modules) to prepare.
        pp_enabled: Whether pipeline parallelism is enabled.
    """
    set_is_optim_step(False)
    if pp_enabled:
        return

    for mp in model_parts:
        if hasattr(mp, "prepare_for_grad_accumulation"):
            mp.prepare_for_grad_accumulation(pp_enabled=pp_enabled)


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
        for mp in model_parts:
            for p in mp.parameters():
                if p.grad is None:
                    continue
                if pp_divisor is not None:
                    p.grad.div_(pp_divisor)
                if ep_ratio is not None:
                    # Grad and param must be DTensors for EP-aware scaling
                    if isinstance(p, DTensor) and isinstance(p.grad, DTensor):
                        if ep_axis_name and ep_axis_name in p.device_mesh.mesh_dim_names:
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
    )


def move_to_device(model, device):
    # FSDP modules do not move buffers to the device automatically
    for v in model.buffers():
        v.data = v.data.to(device)
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()


class ScopedModuleOffloading:
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
