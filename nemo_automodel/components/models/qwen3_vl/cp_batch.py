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

"""Qwen3-VL DeepStack batch sharding for context parallelism."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx


def _pad_sequence_axis(tensor: torch.Tensor, *, seq_dim: int, pad_len: int, value: bool | float) -> torch.Tensor:
    """Pad one sequence axis without changing the other tensor axes.

    Args:
        tensor: Tensor of shape ``[..., sequence, ...]`` whose sequence axis is
            selected by ``seq_dim``.
        seq_dim: Axis containing the sequence extent.
        pad_len: Number of positions to append to the sequence axis.
        value: Scalar fill value for the appended positions.

    Returns:
        Tensor of shape ``[..., sequence + pad_len, ...]``. When ``pad_len`` is
        zero, the input tensor is returned without copying.
    """
    if pad_len == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    padding = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=seq_dim)


def _shard_load_balanced_sequence(tensor: torch.Tensor, *, seq_dim: int, cp_mesh: DeviceMesh) -> torch.Tensor:
    """Select this rank's head-tail context-parallel sequence shard.

    Args:
        tensor: Tensor of shape ``[..., sequence, ...]`` with ``sequence``
            divisible by ``2 * cp_size``.
        seq_dim: Axis containing the global sequence extent.
        cp_mesh: One-dimensional context-parallel mesh. Its local rank selects
            one head chunk and the mirrored tail chunk.

    Returns:
        Tensor of shape ``[..., local_sequence, ...]`` where
        ``local_sequence = sequence / cp_size``. The output retains autograd
        connections to ``tensor``.
    """
    cp_size = cp_mesh.size()
    chunk_size = tensor.shape[seq_dim] // (2 * cp_size)
    cp_rank = cp_mesh.get_local_rank()
    head = tensor.narrow(seq_dim, cp_rank * chunk_size, chunk_size)
    tail = tensor.narrow(seq_dim, (2 * cp_size - cp_rank - 1) * chunk_size, chunk_size)
    return torch.cat((head, tail), dim=seq_dim)


def make_qwen3_vl_cp_batch(
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
) -> tuple[Callable[[], AbstractContextManager[Any]], dict[str, Any]]:
    """Shard Qwen3-VL's sequence and ragged DeepStack features in matching order.

    Qwen3-VL represents DeepStack inputs as a visual-position mask of shape
    ``[batch, sequence]`` plus one ragged tensor of shape ``[visual_tokens,
    hidden]`` per decoder layer. The standard CP sharder handles tensors with a
    sequence axis, so this function first expands each ragged tensor to
    ``[batch, sequence, hidden]``, selects the same head-tail shard as the token
    embeddings, and converts it back to the local ragged representation.

    Args:
        cp_mesh: One-dimensional context-parallel mesh, or ``None`` when CP is inactive.
        tp_mesh: Optional tensor-parallel mesh. Qwen3-VL's DeepStack transform is
            sequence-only, so this mesh does not change the transform.
        batch: Batch mapping containing ``inputs_embeds`` with shape ``[batch,
            sequence, hidden]``, ``labels`` with shape ``[batch, sequence]``,
            ``visual_pos_masks`` with shape ``[batch, sequence]``, and
            ``_deepstack_visual_embeds`` as a list of tensors with shape
            ``[visual_tokens, hidden]``.
        loss_mask: Optional tensor of shape ``[batch, sequence]`` sharded with labels.
        padding_token_id: Token id used if the shared CP sharder pads token ids.

    Returns:
        The ``(context_factory, batch)`` pair returned by
        :func:`make_cp_batch_and_ctx`. ``batch["visual_pos_masks"]`` has shape
        ``[batch, local_sequence]`` and each DeepStack tensor has shape
        ``[local_visual_tokens, hidden]`` in the same head-tail order.

    Raises:
        TypeError: If ``visual_pos_masks`` is not boolean.
        ValueError: If the mask does not match ``inputs_embeds``, or a DeepStack
            tensor is not rank two or its visual-token count does not match the mask.
    """
    del tp_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        return make_cp_batch_and_ctx(
            cp_mesh,
            batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
        )

    visual_pos_masks = batch["visual_pos_masks"]
    if visual_pos_masks.dtype != torch.bool:
        raise TypeError("Qwen3-VL visual_pos_masks must be a boolean tensor")
    inputs_embeds = batch["inputs_embeds"]
    if inputs_embeds.ndim != 3 or visual_pos_masks.ndim != 2:
        raise ValueError(
            "Qwen3-VL CP expects inputs_embeds [batch, sequence, hidden] and "
            f"visual_pos_masks [batch, sequence], got {tuple(inputs_embeds.shape)} and "
            f"{tuple(visual_pos_masks.shape)}"
        )
    if visual_pos_masks.shape != inputs_embeds.shape[:2]:
        raise ValueError(
            "Qwen3-VL visual_pos_masks must match inputs_embeds on batch and sequence axes, got "
            f"{tuple(visual_pos_masks.shape)} and {tuple(inputs_embeds.shape[:2])}"
        )

    deepstack_visual_embeds = batch["_deepstack_visual_embeds"]
    num_visual_tokens = int(visual_pos_masks.sum().item())
    for layer_idx, embeds in enumerate(deepstack_visual_embeds):
        if embeds.ndim != 2:
            raise ValueError(
                "Qwen3-VL DeepStack tensor "
                f"{layer_idx} must have shape [visual_tokens, hidden], got {tuple(embeds.shape)}"
            )
        if embeds.shape[0] != num_visual_tokens:
            raise ValueError(
                "Qwen3-VL DeepStack tensor "
                f"{layer_idx} has {embeds.shape[0]} visual tokens but visual_pos_masks selects {num_visual_tokens}"
            )

    cp_divisor = 2 * cp_mesh.size()
    seq_len = visual_pos_masks.shape[1]
    pad_len = (-seq_len) % cp_divisor
    padded_mask = _pad_sequence_axis(visual_pos_masks, seq_dim=1, pad_len=pad_len, value=False)
    local_mask = _shard_load_balanced_sequence(padded_mask, seq_dim=1, cp_mesh=cp_mesh)

    local_deepstack: list[torch.Tensor] = []
    for embeds in deepstack_visual_embeds:
        sequence_aligned = embeds.new_zeros(*visual_pos_masks.shape, embeds.shape[-1])
        sequence_aligned[visual_pos_masks] = embeds
        sequence_aligned = _pad_sequence_axis(sequence_aligned, seq_dim=1, pad_len=pad_len, value=0.0)
        local_sequence_aligned = _shard_load_balanced_sequence(sequence_aligned, seq_dim=1, cp_mesh=cp_mesh)
        local_deepstack.append(local_sequence_aligned[local_mask])

    batch["visual_pos_masks"] = local_mask
    batch["_deepstack_visual_embeds"] = local_deepstack
    return make_cp_batch_and_ctx(
        cp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
    )
