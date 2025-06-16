# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import contextlib
from typing import Dict, Generator, List, Optional, Set, Union

def _build_position_ids(batch, device):
    """Add position_ids to the batch only if they are missing."""
    # TODO(@boxiangw): Refractor. Needed for SP support
    # If 'position_ids' does not exist in batch already then override it.
    # In case of Packed sequence contains 'position_ids' and we don't want to override it.
    if 'position_ids' not in batch:
        seq_len = batch['input_ids'].shape[1]
        batch['position_ids'] = torch.arange(seq_len, device=device).unsqueeze(0)
    return batch

# based on https://github.com/pytorch/torchtitan/blob/0b44d4c437c424b6bf719661c0eb4283dc4068bc/torchtitan/distributed/utils.py#L180
def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool, cp_context=None):
    """
    Create a train context.

    Args:
        enable_loss_parallel (bool): Whether to enable loss parallelism.
        enable_compiled_autograd (bool): Whether to enable compiled autograd.
    """

    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # currently we only support these two SDP backends.
                # SDPBackend.MATH is not currently compatible with DTensor
                stack.enter_context(
                    sdpa_kernel(
                        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: str,
):
    """
    Create a context parallel context.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        cp_buffers (List[torch.Tensor]): The buffers for context parallel.
        cp_seq_dims (List[int]): The sequence dimensions for context parallel.
        cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
        cp_rotate_method (str): The rotation method for context parallel, such as "allgather" or "addtoall".
    """
    from torch.distributed.tensor.experimental import context_parallel

    # TODO: uncomment this when torch.distributed.tensor.experimental._attention.set_rotate_method is available
    # from torch.distributed.tensor.experimental._attention import set_rotate_method
    # set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )

def make_cp_batch_and_ctx(device_mesh, batch, enable_loss_parallel: bool = False, enable_compiled_autograd: bool = False):
    from contextlib import nullcontext
    if device_mesh is None:
        cp_mesh = None
    else:
        cp_mesh = device_mesh["context_parallel"]

    # return nullcontext()
    if cp_mesh is None or cp_mesh.size() == 1:
        return nullcontext(), batch
    # CP > 0
    # add positions_ids to batch
    batch = _build_position_ids(batch, device=torch.cuda.current_device())
    # prepare buffers
    cp_buffers = list(batch.values())
    cp_seq_dims = [1] * len(cp_buffers)
    no_restore = {
        v
        for k, v in batch.items()
        if v != 'position_ids'
    }
    cp_ctx = create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=no_restore,
        cp_rotate_method="allgather",   # TODO: expose through cfg
    )
    return get_train_context(enable_loss_parallel, enable_compiled_autograd, cp_ctx), batch
