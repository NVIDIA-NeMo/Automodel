# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Private torch context-parallel transport primitives."""

import contextlib
from collections.abc import Callable
from contextlib import AbstractContextManager

import torch
from torch.distributed.device_mesh import DeviceMesh

ContextFactory = Callable[[], AbstractContextManager[None]]


# Based on https://github.com/pytorch/torchtitan/blob/0b44d4c437c424b6bf719661c0eb4283dc4068bc/torchtitan/distributed/utils.py#L180
def get_train_context(
    enable_loss_parallel: bool,
    enable_compiled_autograd: bool,
    cp_context: AbstractContextManager[None] | None = None,
) -> ContextFactory:
    """Build the context entered around a context-parallel forward."""

    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(cp_context)

            yield

    return context


# Based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str | None = None,
) -> AbstractContextManager[None]:
    """Create the torch context-parallel buffer context.

    Args:
        cp_mesh: One-dimensional context-parallel device mesh.
        cp_buffers: Tensors of arbitrary rank with sequence axes selected by
            ``cp_seq_dims``. The context shards these tensors in place while active.
        cp_seq_dims: Sequence-axis index for each tensor in ``cp_buffers``.
        cp_no_restore_buffers: Members of ``cp_buffers`` that remain in their
            per-rank local layout after the context exits.
        cp_rotate_method: Optional Q/K/V rotation implementation.

    Returns:
        Context manager that applies torch context-parallel transport.
    """
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import set_rotate_method

    if cp_rotate_method is not None:
        set_rotate_method(cp_rotate_method)

    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def shard_grad_buffer_for_cp(buffer: torch.Tensor, seq_dim: int, cp_mesh: DeviceMesh) -> torch.Tensor:
    """Shard a gradient-bearing tensor in CP's head-tail order.

    Args:
        buffer: Tensor of arbitrary rank containing the global padded sequence.
        seq_dim: Sequence axis of ``buffer``; its extent must be divisible by
            twice the CP mesh size.
        cp_mesh: One-dimensional context-parallel device mesh.

    Returns:
        Per-rank tensor with the head and tail sequence chunks concatenated on
        ``seq_dim``. The result remains connected to ``buffer`` by autograd.
    """
    cp_size = cp_mesh.size()
    num_chunks = 2 * cp_size
    seq_len = buffer.shape[seq_dim]
    if seq_len % num_chunks != 0:
        raise ValueError(f"CP sequence length {seq_len} must be divisible by {num_chunks}")

    chunk_size = seq_len // num_chunks
    cp_rank = cp_mesh.get_local_rank()
    head_chunk = buffer.narrow(seq_dim, cp_rank * chunk_size, chunk_size)
    tail_chunk = buffer.narrow(seq_dim, (num_chunks - cp_rank - 1) * chunk_size, chunk_size)
    return torch.cat((head_chunk, tail_chunk), dim=seq_dim)
