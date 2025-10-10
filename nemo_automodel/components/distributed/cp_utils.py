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

import contextlib
from typing import List, Optional, Set

import torch
from torch.distributed.device_mesh import DeviceMesh


def _build_position_ids(batch, device):
    """Add position_ids to the batch only if they are missing."""
    # TODO(@boxiangw): Refractor. Needed for SP support
    # If 'position_ids' does not exist in batch already then override it.
    # In case of Packed sequence contains 'position_ids' and we don't want to override it.
    if "position_ids" not in batch:
        seq_len = batch["input_ids"].shape[1]
        batch["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0)
    return batch


# based on https://github.com/pytorch/torchtitan/blob/0b44d4c437c424b6bf719661c0eb4283dc4068bc/torchtitan/distributed/utils.py#L180  # pylint: disable=C0301
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
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # currently we only support these two SDP backends.
                # SDPBackend.MATH is not currently compatible with DTensor
                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(cp_context)

            yield

    return context


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: Optional[str] = None,
):
    """
    Create a context parallel context.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        cp_buffers (List[torch.Tensor]): The buffers for context parallel.
        cp_seq_dims (List[int]): The sequence dimensions for context parallel.
        cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
        cp_rotate_method (str): The rotation method for context parallel,
            such as "allgather" or "addtoall".
    """
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import set_rotate_method

    if cp_rotate_method is not None:
        set_rotate_method(cp_rotate_method)

    # TODO: uncomment this when torch.distributed.tensor.experimental._attention.set_rotate_method
    # is available
    # from torch.distributed.tensor.experimental._attention import set_rotate_method
    # set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def make_cp_batch_and_ctx(
    device_mesh,
    batch,
    loss_mask=None,
    use_te: bool = False,
    padding_token_id: int = 0,
    padding_label_id: int = -100,
):
    """
    Build a CP context manager and shards a batch. If the input device_mesh is None or the size
    of the context_parallel submesh is 1, this function is effectively a no-op.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        batch (Dict[str, torch.Tensor]): The input batch containing (string, torch.Tensor)

    Returns:
        tuple (contextmanager, dict[str, torch.Tensor]): Returns a tuple with a context manager
        and a new batch. The context manager is either nullcontext (no CP) or CP context manager as
        returned by `create_context_parallel_ctx`. The batch has also been passed to
        `create_context_parallel_ctx` and is accordingly sharded.
    """
    from contextlib import nullcontext

    def _get_submesh(device_mesh, name):
        if name in getattr(device_mesh, "mesh_dim_names", {}):
            return device_mesh[name]
        return None

    def _get_mesh_size(mesh):
        if mesh is None:
            return 0
        return mesh.size()

    cp_mesh = _get_submesh(device_mesh, "cp")
    tp_mesh = _get_submesh(device_mesh, "tp")

    if _get_mesh_size(cp_mesh) <= 1:
        return nullcontext, batch

    # CP doesn't support packed sequence currently. Let torch SDPA handle attention mask.
    batch.pop("attention_mask", None)

    if use_te:
        return nullcontext, make_cp_batch_for_te(
            cp_mesh, batch, padding_token_id=padding_token_id, padding_label_id=padding_label_id, qvk_format="thd"
        )

    if "position_ids" not in batch and (_get_mesh_size(cp_mesh) > 1 or _get_mesh_size(tp_mesh) > 1):
        batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(batch["input_ids"].device)

    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]

    labels = batch["labels"]
    if loss_mask is not None:
        cp_buffers = [input_ids, labels, position_ids, loss_mask]
        cp_seq_dims = [1, 1, 1, 1]
        cp_no_restore_buffers = {input_ids, labels, loss_mask}
    else:
        cp_buffers = [input_ids, labels, position_ids]
        cp_seq_dims = [1, 1, 1]
        cp_no_restore_buffers = {input_ids, labels}

    cp_ctx = create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=cp_no_restore_buffers,
        cp_rotate_method="allgather",  # TODO: expose through cfg
    )
    # TODO(@akoumparouli): surface these in the future.
    enable_loss_parallel: bool = False
    enable_compiled_autograd: bool = False
    return get_train_context(enable_loss_parallel, enable_compiled_autograd, cp_ctx), batch


def make_cp_batch_for_te(
    cp_mesh,
    batch,
    padding_token_id=0,
    padding_label_id=-100,
    qvk_format="thd",
    max_seq_len=4096,
):
    """
    Build a CP batch for Transformer Engine using THD format.

    This function pads sequences to be divisible by the CP size and shards the batch
    across context parallel ranks using the utilities from te_cp_utils.

    Currently only supports THD format with seq_lens and seq_lens_padded in the batch.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        batch (Dict[str, torch.Tensor]): The input batch containing:
            - input_ids: Input token IDs
            - labels: Label token IDs
            - seq_lens: Sequence lengths (required)
            - seq_lens_padded: Padded sequence lengths (required)
            - loss_mask (optional): Loss mask tensor
        padding_token_id (int): Token ID to use for padding (default: 0)
        padding_label_id (int): Label ID to use for padding (default: -100)
        qvk_format (str): Format for QKV tensors (currently only "thd" is supported)

    Returns:
        dict: Processed batch with sharded tensors for this CP rank. Contains:
            - input_ids: Padded and sharded input_ids
            - labels: Padded and sharded labels
            - position_ids: Generated and sharded position_ids
            - cu_seqlens: Padded cumulative sequence lengths (not sharded)
            - loss_mask (optional): Padded and sharded loss_mask if provided
    """
    from nemo_automodel.components.distributed.te_cp_utils import (
        generate_positional_ids_for_cp,
        get_batch_on_this_cp_rank,
        pad_thd_sequences_for_cp,
    )

    if qvk_format != "thd":
        raise ValueError(f"Currently only 'thd' format is supported, got: {qvk_format}")

    # Extract fields from batch
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    cu_seqlens = batch.get("cu_seqlens", None)
    cu_seqlens_padded = batch.get("cu_seqlens_padded", None)

    # Check for required fields - BSHD format is not supported
    if cu_seqlens is None or cu_seqlens_padded is None:
        raise ValueError(
            "BSHD format is not supported. Both 'seq_lens' and 'seq_lens_padded' must be present in the batch. "
            "Please use packed sequence format with seq_lens and seq_lens_padded."
        )

    cp_size = cp_mesh.size()

    # Filter padded tokens based on seq_lens and seq_lens_padded
    # Extract only the valid tokens (excluding tail padding) for each sequence
    filtered_input_ids = [input_ids[cu_seqlens[i] : cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)]
    filtered_labels = [labels[cu_seqlens[i] : cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)]
    input_ids = torch.cat(filtered_input_ids)
    labels = torch.cat(filtered_labels)

    # CP requires each sequence to be divisible by 2 * cp_size
    divisibility_factor = 2 * cp_size

    # Pad sequences
    input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
        input_ids, labels, cu_seqlens, divisibility_factor, padding_token_id, padding_label_id, max_seq_len=max_seq_len
    )

    # Generate position IDs for padded sequences
    position_ids_padded = generate_positional_ids_for_cp(cu_seqlens_padded, divisibility_factor=1, dtype=torch.long).to(
        input_ids_padded.device
    )

    # Get CP rank and size
    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group()) if cp_mesh is not None else 0

    # Shard the batch across CP ranks
    input_ids_sharded, labels_sharded, position_ids_sharded = get_batch_on_this_cp_rank(
        cu_seqlens_padded,
        input_ids_padded,
        labels_padded,
        position_ids_padded,
        cp_size=cp_size,
        cp_rank=cp_rank,
        qvk_format=qvk_format,
    )

    # Build the output batch
    max_seqlen = (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).max().item()
    # max_seqlen = max_seqlen // cp_size
    output_batch = {
        "input_ids": input_ids_sharded.to(torch.int64).contiguous(),
        "labels": labels_sharded.to(torch.int64).contiguous(),
        "position_ids": position_ids_sharded.to(torch.int64).contiguous(),
        "cu_seqlens_q": cu_seqlens_padded.to(torch.int32).contiguous(),
        "cu_seqlens_kv": cu_seqlens_padded.to(torch.int32).contiguous(),
        "cu_seqlens_q_padded": cu_seqlens_padded.to(torch.int32).contiguous(),
        "cu_seqlens_kv_padded": cu_seqlens_padded.to(torch.int32).contiguous(),
        "max_seqlen_q": max_seqlen,
        "max_seqlen_kv": max_seqlen,
        "qkv_format": qvk_format,
    }
    return output_batch
