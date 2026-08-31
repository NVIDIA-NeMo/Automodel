# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Context-parallel helpers for HY V4's packed cuDNN DSA attention."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ShardLayout,
    contiguous_local_indices,
)
from nemo_automodel.components.distributed.thd_utils import (
    split_batch_into_thd_chunks,
    thd_padding_mask_from_token_ids,
)


def hy_v4_cp_enabled(cp_group: Any) -> bool:
    """Return whether a real HY V4 DSA CP process group is active."""
    return (
        cp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=cp_group) > 1
    )


def hy_v4_cp_all_gather(tensor: torch.Tensor, *, dim: int, cp_group: Any) -> torch.Tensor:
    """All-gather an activation axis across CP ranks with autograd.

    Args:
        tensor: Rank-local activation with arbitrary leading/trailing axes.
        dim: Axis along which rank-ordered shards are concatenated.
        cp_group: Context-parallel process group, or ``None`` for CP1.

    Returns:
        The concatenated global activation. Under CP1 the exact input object is
        returned; under CP>1 the result is differentiably gathered and does not
        alias the local input.
    """
    if not hy_v4_cp_enabled(cp_group):
        return tensor

    parts = all_gather(tensor.contiguous(), group=cp_group)
    return torch.cat(tuple(parts), dim=dim)


def _slice_thd_chunk_for_cp(
    chunk: dict[str, torch.Tensor],
    *,
    cp_mesh: Any,
    cp_group: Any,
    cp_size: int,
    cp_rank: int,
    padding_token_id: int,
) -> dict[str, Any]:
    """Slice one global packed THD stream into a contiguous CP query shard.

    Args:
        chunk: Global THD batch. Token fields use ``[T_global]`` and cumulative
            sequence fields use ``[documents + 1]``.
        cp_mesh: Context-parallel mesh used to calculate the local token range.
        cp_group: Process group later used for differentiable K/V gathers.
        cp_size: Number of CP ranks.
        cp_rank: Rank within the CP group.
        padding_token_id: Token ID used only when no pack-derived mask exists.

    Returns:
        A new mapping whose token fields have shape ``[T_global / cp_size]``;
        cumulative sequence metadata remains global and tensors do not alias
        the corresponding input fields.
    """
    total_tokens = int(chunk["input_ids"].shape[0])
    query_indices = contiguous_local_indices(cp_mesh, total_tokens, chunk["input_ids"].device)

    out: dict[str, Any] = {
        "input_ids": chunk["input_ids"].index_select(0, query_indices).to(torch.int64).contiguous(),
        "labels": chunk["labels"].index_select(0, query_indices).to(torch.int64).contiguous(),
        "position_ids": chunk["position_ids"].index_select(0, query_indices).to(torch.int64).contiguous(),
        "cu_seqlens": chunk["cu_seqlens"].to(torch.int32).contiguous(),
        "qkv_format": "thd",
        "cp_size": cp_size,
        "cp_rank": cp_rank,
        "_hy_v4_cp_group": cp_group,
        "hy_v4_cp_query_indices": query_indices.to(torch.int32).contiguous(),
    }
    if "max_seqlen" in chunk:
        out["max_seqlen"] = chunk["max_seqlen"].to(torch.int32).contiguous()
    if "cu_seqlens_padded" in chunk:
        out["cu_seqlens_padded"] = chunk["cu_seqlens_padded"].to(torch.int32).contiguous()
    # Preserve the packer's authoritative padding decisions. The public
    # checkpoint uses distinct pad/eos IDs (120002/120025); deriving a new mask
    # is only a compatibility fallback for batches that omit this field.
    if "padding_mask" in chunk:
        out["padding_mask"] = chunk["padding_mask"].index_select(0, query_indices).bool().contiguous()
    else:
        out["padding_mask"] = thd_padding_mask_from_token_ids(out["input_ids"], padding_token_id).contiguous()
    return out


def _packed_cp_layout(batch: dict[str, Any], *, num_chunks: int) -> ShardLayout | None:
    """Describe the caller's unflattened layout for a single packed stream."""
    # The BSHD->THD flatten is a pure reshape: the pre-flatten rows are the
    # caller's coordinate system and the stream length is rows x cols. Chunked
    # streams (num_chunks > 1) are per-chunk token spaces and report no layout.
    input_ids = batch.get("input_ids")
    if num_chunks <= 1 and input_ids is not None and input_ids.dim() >= 2:
        return ShardLayout(
            padded_seq_len=input_ids.shape[0] * input_ids.shape[1],
            input_row_shape=tuple(input_ids.shape[:2]),
        )
    return None


def make_hy_v4_packed_cp_batch_and_ctx(
    cp_mesh: Any,
    tp_mesh: Any,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
) -> tuple[Callable[[], AbstractContextManager], dict[str, Any]]:
    """Convert packed HY V4 DSA batches to THD and keep a contiguous query shard per CP rank.

    HY V4 DSA sparse attention gathers K/V activations inside the model. The batch
    side only slices local query tokens and carries the full packed-sequence
    ``cu_seqlens`` plus per-query global token indices for cuDNN's causal
    top-k window.

    Args:
        cp_mesh: Context-parallel mesh that owns contiguous query shards.
        tp_mesh: Unused tensor-parallel mesh accepted by the common sharder API.
        batch: Packed or pre-packed batch. Before THD conversion, token fields
            conventionally have shape ``[batch, sequence]``.
        loss_mask: Unused optional mask accepted by the common sharder API.
        padding_token_id: Tokenizer padding ID for the legacy mask fallback.
        num_chunks: Number of pipeline microbatch chunks represented in batch.
        seq_lens_padding_value: Sentinel used in padded sequence-length metadata.

    Returns:
        A null context factory and a fresh THD mapping. With one chunk, token
        fields are ``[T_local]``; multiple chunks use ``[chunks, T_local]``.
    """
    del tp_mesh, loss_mask

    thd_batch = split_batch_into_thd_chunks(
        batch,
        num_chunks=num_chunks,
        seq_lens_padding_value=seq_lens_padding_value,
        padding_token_id=padding_token_id,
    )
    cp_group = cp_mesh.get_group()
    cp_size = cp_mesh.size()
    cp_rank = dist.get_rank(group=cp_group) if dist.is_available() and dist.is_initialized() else 0

    if num_chunks <= 1:
        sliced = _slice_thd_chunk_for_cp(
            thd_batch,
            cp_mesh=cp_mesh,
            cp_group=cp_group,
            cp_size=cp_size,
            cp_rank=cp_rank,
            padding_token_id=padding_token_id,
        )
        return contextlib.nullcontext, sliced

    chunks = []
    for idx in range(num_chunks):
        chunk = {key: value[idx] if isinstance(value, torch.Tensor) else value for key, value in thd_batch.items()}
        chunks.append(
            _slice_thd_chunk_for_cp(
                chunk,
                cp_mesh=cp_mesh,
                cp_group=cp_group,
                cp_size=cp_size,
                cp_rank=cp_rank,
                padding_token_id=padding_token_id,
            )
        )

    stacked: dict[str, Any] = {}
    for key, value in chunks[0].items():
        if isinstance(value, torch.Tensor):
            tensor_values = [chunk[key] for chunk in chunks]
            if not all(isinstance(item, torch.Tensor) for item in tensor_values):
                raise TypeError(f"HY4 CP chunk field {key!r} changed type across chunks")
            stacked[key] = torch.stack(tensor_values)
        else:
            stacked[key] = value
    return contextlib.nullcontext, stacked


def shard_hy_v4_packed_cp_batch(
    cp_mesh: Any,
    tp_mesh: Any,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
) -> tuple[Callable[[], AbstractContextManager], dict[str, Any], ShardLayout | None]:
    """Shard an HY4 packed batch and report the original row layout.

    Args:
        cp_mesh: Context-parallel mesh that owns contiguous query shards.
        tp_mesh: Tensor-parallel mesh accepted by the common sharder interface.
        batch: Global packed batch with token fields shaped ``[batch, sequence]``.
        loss_mask: Optional mask accepted by the common sharder interface.
        padding_token_id: Tokenizer padding ID for a missing-mask fallback.
        num_chunks: Number of pipeline microbatch chunks in ``batch``.
        seq_lens_padding_value: Sentinel used for padded sequence-length entries.

    Returns:
        Null context factory, local THD mapping, and the single-stream source
        layout used to restore loss/logit coordinates. Outputs do not mutate
        or alias the batch's token tensors.
    """
    layout = _packed_cp_layout(batch, num_chunks=num_chunks)
    ctx_factory, sharded_batch = make_hy_v4_packed_cp_batch_and_ctx(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        num_chunks=num_chunks,
        seq_lens_padding_value=seq_lens_padding_value,
    )
    return ctx_factory, sharded_batch, layout
