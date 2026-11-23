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

from collections.abc import Iterable

import torch
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.tensor_utils import to_local_if_dtensor


def clip_grad_by_total_norm_(
    parameters: Iterable[torch.Tensor | DTensor] | torch.Tensor | DTensor,
    max_grad_norm: int | float,
    total_norm: float,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Clips gradient of an iterable of parameters by total norm.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L138

    Note that the gradients are modified in place.

    Args:
        parameters (Iterable[torch.Tensor | DTensor] | torch.Tensor | DTensor):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradients normalized.
        max_grad_norm (float | int): Maximum norm of the gradients.
        total_norm (float): The pre-computed total norm of the gradients to use for scaling.
        dtype (torch.dtype, optional): Data type to cast gradients for scaling. Defaults to torch.float32.
    """
    params: Iterable[torch.Tensor | DTensor]
    if isinstance(parameters, (torch.Tensor, DTensor)):
        params = [parameters]
    else:
        params = parameters

    # Grads.
    grads = [to_local_if_dtensor(p.grad.detach()).to(dtype) for p in params if p.grad is not None]

    # Scale.
    clip_coeff = max_grad_norm / (total_norm + 1.0e-6)

    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)


def get_grad_norm(
    parameters: Iterable[torch.Tensor | DTensor] | torch.Tensor | DTensor,
    dp_cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    norm_type: int | float = 2,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Calculate the norm of gradients.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L51

    Args:
        parameters (Iterable[torch.Tensor | DTensor] | torch.Tensor | DTensor):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradient norm calculated.
        dp_cp_group (torch.distributed.ProcessGroup): Process group for data and context parallel communication.
        tp_group (torch.distributed.ProcessGroup): Process group for tensor parallel communication.
        norm_type (int | float): Type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        dtype (torch.dtype, optional): Data type for gradient conversion. Defaults to torch.float32.

    Returns:
        float: Total norm of the gradients (viewed as a single vector)
    """
    params: Iterable[torch.Tensor | DTensor]
    if isinstance(parameters, (torch.Tensor, DTensor)):
        params = [parameters]
    else:
        params = parameters

    # Grads.
    grads_for_norm = [to_local_if_dtensor(p.grad.detach()).to(dtype) for p in params if p.grad is not None]

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm: float = 0.0

    # Calculate norm.
    if norm_type == torch.inf:
        total_norm = max(grad.abs().max().item() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device="cuda")
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_cp_group)

        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        total_norm = total_norm_cuda[0].item()

    else:
        total_norm_accum = torch.zeros(1, dtype=torch.float32)
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm_accum += grad_norm**norm_type

        total_norm_cuda = total_norm_accum.cuda()
        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group)

        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        total_norm = float(total_norm_cuda.item() ** (1.0 / norm_type))

    return total_norm
