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

import math
from typing import Iterable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


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
def clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
    ep_axis_name: str,
) -> torch.Tensor:
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
        if ep_axis_name not in p.device_mesh.mesh_dim_names:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)
        else:
            ep_params.append(p)
            ep_grads.append(p.grad)
    ep_grads_total_norm = torch.nn.utils.get_total_norm(ep_grads, norm_type, error_if_nonfinite, foreach).full_tensor()
    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    ).full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        total_norm **= 1.0 / norm_type

    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    return total_norm


@torch.no_grad()
def clip_grad_norm_with_pp(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_axis_name: str | None = None,
) -> torch.Tensor:
    if ep_axis_name:
        return clip_grad_norm_with_ep(
            parameters, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh, ep_axis_name
        )

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def clip_grad_norm(
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
):
    """Common gradient clipping helper.

    Handles both pipeline-parallel and single-model clipping paths. Returns a float grad norm when available,
    otherwise 0.0 if clipping is skipped due to constraints.
    """
    grad_norm = 0
    if max_grad_norm is None:
        return grad_norm

    if pp_enabled:
        assert pp_axis_name is not None, "pp_axis_name must be provided when pp_enabled is True"
        pp_mesh = device_mesh[pp_axis_name]
        grad_norm = clip_grad_norm_with_pp(
            [p for m in model_parts for p in m.parameters()],
            max_norm=max_grad_norm,
            norm_type=norm_type,
            error_if_nonfinite=False,
            foreach=foreach,
            pp_mesh=pp_mesh,
            ep_axis_name=ep_axis_name,
        )
    else:
        # MoE present without PP: handle expert-parallel aware clipping
        if moe_mesh is not None:
            assert ep_axis_name is not None, "ep_axis_name must be provided when moe_mesh is not None"
            grad_norm = clip_grad_norm_with_ep(
                [p for p in model_parts[0].parameters() if p.requires_grad],
                max_norm=max_grad_norm,
                norm_type=norm_type,
                error_if_nonfinite=False,
                foreach=foreach,
                pp_mesh=None,
                ep_axis_name=ep_axis_name,
            )
        # Only clip locally if no MoE and either no device_mesh or TP size is 1
        elif moe_mesh is None and (not device_mesh or device_mesh["tp"].size() == 1):
            params = [p for p in model_parts[0].parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()  # collect the summed grad norm across ranks
        else:
            return 0

    if isinstance(grad_norm, torch.Tensor):
        grad_norm = grad_norm.item()

    return grad_norm


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
        moe_mesh=moe_mesh,
        ep_axis_name=ep_axis_name,
        pp_axis_name=pp_axis_name,
        foreach=foreach,
    )
