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

"""
Tensor utilities for device transfers and memory management in distributed settings.

This module provides utilities for handling tensor operations across different devices
and distributed tensor types, with optimizations for performance in distributed training scenarios.
"""

from typing import Iterable, Union

import torch
from torch.distributed.tensor import DTensor


@torch.no_grad()
def get_cpu_state_dict(
    state_generator: Iterable[tuple[str, Union[torch.Tensor, DTensor]]],
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Copy the state dict generator to CPU memory.

    Args:
        state_generator (Iterable[tuple[str, Union[torch.Tensor, DTensor]]]):
            An iterable that yields (key, tensor) pairs from a model state.
        pin_memory (bool, optional):
            Whether to allocate the CPU tensors in pinned memory for faster GPU transfer.
            Defaults to False.

    Returns:
        dict[str, torch.Tensor]: A dictionary mapping parameter names to CPU tensors.
    """
    new_state_dict = {}
    for k, v in state_generator:
        val = to_local_if_dtensor(v)

        if len(val.shape) == 0:
            new_state_dict[k] = val.cpu()
        else:
            cpu_tensor = torch.empty(*val.shape, device="cpu", pin_memory=pin_memory, dtype=val.dtype)
            cpu_tensor.copy_(val, non_blocking=True)
            new_state_dict[k] = cpu_tensor

    torch.cuda.synchronize()
    return new_state_dict


def to_cpu(v):
    """
    Move a tensor or distributed tensor to the CPU.

    This function takes an input tensor, which can be either a `DTensor` (distributed tensor)
    or a standard `Tensor`, and ensures that it is moved to the CPU.

    Args:
        v (DTensor | Tensor | any): The input value, which can be a `DTensor`, `Tensor`, or
                                    any other object. If `DTensor`, it checks the device and
                                    moves the tensor accordingly.

    Returns:
        Tensor | any: The corresponding CPU tensor if `v` is a `DTensor` or `Tensor`,
                    otherwise returns `v` unchanged.

    Raises:
        ValueError: If `v` is a `DTensor` but its device is neither 'cuda' nor 'cpu'.

    Example:
        >>> t = torch.tensor([1, 2, 3], device='cuda')
        >>> to_cpu(t)  # Moves tensor to CPU
        tensor([1, 2, 3])

        >>> dt = DTensor(torch.tensor([4, 5, 6], device='cuda'))
        >>> to_cpu(dt)  # Moves DTensor to CPU
        tensor([4, 5, 6])
    """
    if isinstance(v, DTensor):
        if v.device.type == "cuda":
            return v.full_tensor().cpu()
        elif v.device.type == "cpu":
            return v._local_tensor
        else:
            raise ValueError("Unknown device " + str(v.device))
    elif isinstance(v, torch.Tensor):
        return v.cpu()
    else:
        return v


def to_local_if_dtensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    """Returns the local shard of the given tensor if it is a DTensor.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/605f618f237cda8fa80132bc2ccff933512d5a0d/megatron/core/utils.py#L746
    """
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
