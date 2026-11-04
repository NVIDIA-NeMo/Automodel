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

"""MiMo adapters for the framework Transformer Engine THD sharder."""

from __future__ import annotations

import contextlib
from functools import partial
from typing import Any

import torch

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
)
from nemo_automodel.components.distributed.context_parallel.utils import make_cp_batch_for_te
from nemo_automodel.components.models.mimo_v2_flash.parallelization import ensure_mimo_te_context_parallel

_MIMO_GLOBAL_IMAGE_MASK = "_mimo_global_image_mask"
_MIMO_GLOBAL_VIDEO_MASK = "_mimo_global_video_mask"
_MIMO_THD_LOCAL_INDICES = "_mimo_thd_local_indices"
_VLM_PP_MEDIA_KEY = "_vlm_pp_media_chunks"
_SEQ_LENS_PADDING_VALUE = -1000


def _flatten_chunks(tensor: torch.Tensor, num_chunks: int) -> torch.Tensor:
    """Flatten batch rows into one token stream per pipeline chunk.

    Args:
        tensor: Token-aligned tensor of shape [batch, sequence].
        num_chunks: Number of equal groups along the batch axis.

    Returns:
        Tensor of shape [num_chunks, batch * sequence / num_chunks], or a
        one-dimensional [batch * sequence] tensor when ``num_chunks == 1``.
    """
    if tensor.ndim != 2:
        raise ValueError(f"MiMo THD token metadata must have shape [batch, sequence], got {tuple(tensor.shape)}")
    if num_chunks <= 0 or tensor.shape[0] % num_chunks:
        raise ValueError(f"MiMo THD num_chunks={num_chunks} must evenly divide batch size {tensor.shape[0]}")
    if num_chunks == 1:
        return tensor.reshape(-1).contiguous()
    rows_per_chunk = tensor.shape[0] // num_chunks
    return tensor.reshape(num_chunks, rows_per_chunk * tensor.shape[1]).contiguous()


def _media_mask(input_ids: torch.Tensor, token_id: int | None, num_chunks: int) -> torch.Tensor | None:
    """Build a global placeholder mask before Transformer Engine shards tokens.

    Args:
        input_ids: Unsharded token IDs of shape [batch, sequence].
        token_id: Image or video placeholder ID, or ``None`` when absent.
        num_chunks: Number of pipeline token streams.

    Returns:
        Boolean mask in global THD stream order, with shape [tokens] for one
        chunk or [chunks, tokens_per_chunk] for pipeline execution. Returns
        ``None`` when the modality has no configured placeholder ID.
    """
    if token_id is None:
        return None
    return _flatten_chunks(input_ids.eq(int(token_id)), num_chunks)


def _chunk_partition_indices(
    batch: dict[str, Any],
    *,
    cp_mesh,
    num_chunks: int,
    seq_lens_padding_value: int,
) -> torch.Tensor:
    """Reproduce TE's data-dependent THD token partition for every chunk.

    Args:
        batch: Unsharded packed batch. ``input_ids`` has shape [batch,
            sequence] and ``seq_lens_padded`` has shape [batch, documents].
        cp_mesh: Optional one-dimensional context-parallel mesh.
        num_chunks: Number of equal pipeline chunks along the batch axis.
        seq_lens_padding_value: Sentinel used in ragged length rows.

    Returns:
        Global token indices with shape [local_tokens] for one chunk or
        [chunks, local_tokens_per_chunk] for pipeline execution. Indices are
        relative to the corresponding chunk's global flattened token stream.
    """
    input_ids = batch.get("input_ids")
    padded_lengths = batch.get("seq_lens_padded")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("MiMo TE THD sharding requires input_ids [batch, sequence]")
    if not isinstance(padded_lengths, torch.Tensor) or padded_lengths.ndim != 2:
        raise ValueError("MiMo TE THD sharding requires seq_lens_padded [batch, documents]")
    if input_ids.shape[0] % num_chunks:
        raise ValueError(f"MiMo THD num_chunks={num_chunks} must evenly divide batch size {input_ids.shape[0]}")

    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    if cp_mesh is None:
        cp_rank = 0
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group())
    else:
        cp_rank = getattr(cp_mesh, "get_local_rank", lambda: 0)()
    rows_per_chunk = input_ids.shape[0] // num_chunks
    tokens_per_chunk = rows_per_chunk * input_ids.shape[1]
    chunk_indices = []
    for chunk_idx in range(num_chunks):
        if cp_size == 1:
            indices = torch.arange(tokens_per_chunk, device=input_ids.device, dtype=torch.long)
        else:
            rows = padded_lengths[chunk_idx * rows_per_chunk : (chunk_idx + 1) * rows_per_chunk]
            lengths = rows.reshape(-1)
            lengths = lengths[lengths != seq_lens_padding_value].to(torch.int32)
            if lengths.numel() == 0 or bool((lengths <= 0).any().item()):
                raise ValueError("MiMo TE THD padded document lengths must be positive")
            if int(lengths.sum().item()) != tokens_per_chunk:
                raise ValueError(
                    "MiMo TE THD seq_lens_padded must cover each pipeline chunk: "
                    f"got {int(lengths.sum().item())} slots for {tokens_per_chunk} tokens"
                )
            divisor = 2 * cp_size
            if bool((lengths.remainder(divisor) != 0).any().item()):
                raise ValueError(
                    "MiMo TE context parallelism requires every padded document length "
                    f"to be divisible by 2 * cp_size ({divisor}); got {lengths.tolist()}"
                )
            cu_seqlens_padded = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=lengths.device),
                    lengths.cumsum(0, dtype=torch.int32),
                )
            )
            import transformer_engine_torch as tex

            indices = tex.thd_get_partitioned_indices(
                cu_seqlens_padded,
                tokens_per_chunk,
                cp_size,
                cp_rank,
            ).to(torch.long)
        chunk_indices.append(indices)
    return chunk_indices[0] if num_chunks == 1 else torch.stack(chunk_indices)


def shard_batch_for_mimo_te(
    cp_mesh,
    tp_mesh,
    batch: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
):
    """Delegate MiMo packed CP to the framework TE THD sharder.

    The wrapper records global VLM placeholder masks and TE's local-token index
    map before the framework mutates the packed batch. Those tensors let MiMo
    select exactly the image/video features owned by each DualChunkSwap shard.
    It also preserves the PP media side channel, which is intentionally not a
    token-aligned tensor and therefore must not be split by the THD helper.

    Args:
        cp_mesh: Optional one-dimensional context-parallel mesh.
        tp_mesh: Unused tensor-parallel mesh required by the sharder protocol.
        batch: Packed batch whose token tensors have shape [batch, sequence].
        model: MiMo model or pipeline-local part whose TE attention is configured
            from the runtime CP mesh before the first forward.
        loss_mask: Optional loss mask passed by the sharder protocol. Labels
            already carry the loss ignore value, so this is unsupported here.
        padding_token_id: Token ID used for physical THD padding.
        num_chunks: Number of pipeline microbatch streams.
        image_token_id: Optional image placeholder token ID.
        video_token_id: Optional video placeholder token ID.

    Returns:
        A null transport context, the TE-prepared batch, and its
        :class:`ShardLayout`. Token tensors are true THD: [local_tokens] for one
        stream or [chunks, local_tokens_per_chunk] for PP.
    """
    del tp_mesh
    if loss_mask is not None:
        raise ValueError("MiMo TE THD sharding does not support an external loss_mask; encode it in labels")
    if batch.get("qkv_format") != "thd":
        raise ValueError(
            "MiMo packed context parallelism requires qkv_format='thd'; "
            "packed NEAT masks cannot be represented by TE context parallelism"
        )
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("MiMo TE THD sharding requires input_ids [batch, sequence]")
    position_ids = batch.get("position_ids")
    if num_chunks > 1 and isinstance(position_ids, torch.Tensor) and position_ids.ndim != 2:
        raise ValueError("MiMo THD pipeline parallelism currently requires one-dimensional position_ids")
    if model is not None:
        ensure_mimo_te_context_parallel(model, cp_mesh)

    original_row_shape = tuple(input_ids.shape)
    global_image_mask = _media_mask(input_ids, image_token_id, num_chunks)
    global_video_mask = _media_mask(input_ids, video_token_id, num_chunks)
    local_indices = _chunk_partition_indices(
        batch,
        cp_mesh=cp_mesh,
        num_chunks=num_chunks,
        seq_lens_padding_value=_SEQ_LENS_PADDING_VALUE,
    )
    pp_media = batch.get(_VLM_PP_MEDIA_KEY)

    prepared = make_cp_batch_for_te(
        cp_mesh,
        batch,
        qkv_format="thd",
        padding_token_id=padding_token_id,
        num_chunks=num_chunks,
        seq_lens_padding_value=_SEQ_LENS_PADDING_VALUE,
        return_local_indices=False,
    )
    prepared[_MIMO_THD_LOCAL_INDICES] = local_indices
    if global_image_mask is not None:
        prepared[_MIMO_GLOBAL_IMAGE_MASK] = global_image_mask
    if global_video_mask is not None:
        prepared[_MIMO_GLOBAL_VIDEO_MASK] = global_video_mask
    if pp_media is not None:
        prepared[_VLM_PP_MEDIA_KEY] = pp_media

    layout = None
    if num_chunks == 1:
        layout = ShardLayout(
            local_token_global_indices=local_indices,
            padded_seq_len=original_row_shape[0] * original_row_shape[1],
            input_row_shape=original_row_shape,
        )
    return contextlib.nullcontext, prepared, layout


def make_mimo_te_cp_sharder(
    *,
    model: torch.nn.Module,
    num_chunks: int,
    image_token_id: int | None,
    video_token_id: int | None,
) -> ContextParallelSharder:
    """Create MiMo's thin adapter around the framework TE THD sharder.

    Args:
        model: MiMo model or pipeline-local part to configure from the runtime CP mesh.
        num_chunks: Number of pipeline microbatch streams.
        image_token_id: Optional image placeholder token ID.
        video_token_id: Optional video placeholder token ID.

    Returns:
        An unresolved :class:`ContextParallelSharder` configured by the caller's
        device mesh before its first :meth:`~ContextParallelSharder.shard` call.
    """
    return ContextParallelSharder(
        shard_batch=partial(
            shard_batch_for_mimo_te,
            model=model,
            num_chunks=num_chunks,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        ),
        local_token_global_indices=None,
    )


__all__ = [
    "_MIMO_GLOBAL_IMAGE_MASK",
    "_MIMO_GLOBAL_VIDEO_MASK",
    "_MIMO_THD_LOCAL_INDICES",
    "make_mimo_te_cp_sharder",
    "shard_batch_for_mimo_te",
]
