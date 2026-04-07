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
Deferred ReduceScatter for gradient accumulation.

Replaces GA× ReduceScatter operations with local shard accumulation +
1× AllReduce on the final microbatch, reducing RS communication cost.

Mathematical basis: RS is a linear operation, so
    sum_i(RS(grad_i)) = RS(sum_i(grad_i))

We exploit this by accumulating only the local shard contribution
(no communication) across microbatches and firing AllReduce once.
This is equivalent to Megatron-FSDP's no_sync() + deferred RS pattern.

Works correctly with HSDP, non-Shard(0) placements, TP, CP — the
reordering is transparent to any subsequent collective operations in
foreach_reduce (e.g. HSDP's outer AllReduce still fires on our output).
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter

logger = logging.getLogger(__name__)


class GAState:
    """Shared mutable flag indicating whether the current backward is the final
    microbatch in a gradient accumulation window. Set from the training loop via
    prepare_for_grad_accumulation() / prepare_for_final_backward()."""

    def __init__(self):
        self.is_final: bool = False


class DeferredShardedReduceScatter(ReduceScatter):
    """Custom ReduceScatter that defers communication to the final GA microbatch.

    On non-final microbatches:
        - Accumulates the full input_tensor locally (no communication)
        - Writes zeros to output_tensor (no effect on sharded .grad)

    On the final microbatch:
        - Performs a single ReduceScatter on the accumulated input
        - foreach_reduce then accumulates the result into sharded_param.grad normally

    Correctness: RS is linear, so sum_i(RS(grad_i)) = RS(sum_i(grad_i)).
    We accumulate sum_i(grad_i) locally and do one RS instead of GA× RS ops.

    Each FSDPParamGroup must have its own instance (different param shapes).
    All instances within one model share the same GAState for synchronized
    final-step signaling.

    Args:
        ga_state: Shared GAState object; set ga_state.is_final=True before the
            final backward call.
    """

    def __init__(self, ga_state: GAState):
        self.ga_state = ga_state
        self._accum: torch.Tensor | None = None

    def allocate(self, size, *, dtype, device):
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op,
        async_op: bool = False,
    ):
        # Accumulate the full input across microbatches.
        # predivide_factor has already been applied by foreach_reduce.
        if self._accum is None:
            self._accum = input_tensor.detach().clone()
        else:
            self._accum.add_(input_tensor)

        if not self.ga_state.is_final:
            # Non-final microbatch: suppress communication, write zeros.
            # foreach_reduce will set sharded_param.grad = 0 (or accumulate 0),
            # which is overwritten on the final microbatch.
            output_tensor.zero_()
            return None

        # Final microbatch: single RS on the accumulated input.
        # By linearity: RS(sum_i grad_i) = sum_i RS(grad_i), so this produces
        # the same result as GA× individual RS ops.
        accum = self._accum
        self._accum = None

        if output_tensor.numel() == 0:
            return None

        return dist.reduce_scatter_tensor(output_tensor, accum, op=op, group=group, async_op=async_op)
