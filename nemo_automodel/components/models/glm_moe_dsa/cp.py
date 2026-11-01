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

"""Context-parallel helpers for GLM MoE DSA TileLang attention."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ShardLayout,
    contiguous_local_indices,
)
from nemo_automodel.components.distributed.thd_utils import (
    split_batch_into_thd_chunks,
    thd_padding_mask_from_token_ids,
)
from nemo_automodel.shared.recompute_replay import RecomputeReplay

# Record/replay channel for the gathered latent K/V of each layer across an activation-checkpoint
# recompute (bound per checkpointed block call by ``Block.nemo_checkpoint_context_fn``): the recompute
# reuses the forward's gathered tensor instead of communicating again, while its backward still
# reduce-scatters the gradient to the owning ranks.
glm_dsa_gather_replay: RecomputeReplay[torch.Tensor] = RecomputeReplay("GLM DSA CP gathered latent")


def glm_dsa_cp_enabled(cp_group) -> bool:
    """Return whether a real GLM DSA CP process group is active."""
    return (
        cp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=cp_group) > 1
    )


def _wait(tensor: torch.Tensor) -> torch.Tensor:
    """Complete an asynchronous functional collective result."""
    if isinstance(tensor, funcol.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


class _ReplayedAllGather(torch.autograd.Function):
    """Return a recorded all-gather result while keeping the gradient path of the gather."""

    @staticmethod
    def forward(ctx: Any, local: torch.Tensor, gathered: torch.Tensor, dim: int, cp_group) -> torch.Tensor:
        """Hand back ``gathered`` (the forward's all-gather of ``local``) without communicating.

        Args:
            ctx: Autograd context.
            local: This rank's shard, only used for its gradient shape and dtype.
            gathered: The recorded all-gather of ``local`` across the CP group along ``dim``.
            dim: Gather dimension.
            cp_group: CP process group.

        Returns:
            ``gathered`` as a fresh view.
        """
        del local
        ctx.dim = dim
        ctx.cp_group = cp_group
        return gathered.view_as(gathered)

    @staticmethod
    def backward(ctx: Any, grad_gathered: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Reduce-scatter the gathered gradient back to each rank's shard, as the gather's backward does."""
        grad_local = _wait(funcol.reduce_scatter_tensor(grad_gathered.contiguous(), "sum", ctx.dim, ctx.cp_group))
        return grad_local, None, None, None


def glm_dsa_cp_all_gather(
    tensor: torch.Tensor,
    *,
    dim: int,
    cp_group,
    replay: RecomputeReplay[torch.Tensor] | None = None,
) -> torch.Tensor:
    """All-gather an activation across CP ranks along ``dim`` while preserving autograd.

    One functional collective produces the concatenated result directly (no per-rank list and
    concat copy), and its backward reduce-scatters the gradient to the owning ranks. With
    ``replay`` bound in record mode the gathered tensor is recorded; in replay mode (the
    activation-checkpoint recompute) the recorded tensor is returned without communication,
    with the same gradient path.

    Args:
        tensor: This rank's shard; ``[local_tokens, ...]`` for ``dim=0``.
        dim: Concatenation dimension.
        cp_group: CP process group; a size-one or missing group returns ``tensor`` unchanged.
        replay: Optional record/replay channel consulted inside a checkpointed block.

    Returns:
        Tensor whose ``dim`` holds every rank's shard in rank order.
    """
    if not glm_dsa_cp_enabled(cp_group):
        return tensor
    local = tensor.contiguous()
    state = replay.current() if replay is not None else None
    if state is not None and state[1] == "replay":
        recorded = state[0].take()
        if recorded is not None:
            return _ReplayedAllGather.apply(local, recorded, dim, cp_group)
    gathered = _wait(funcol.all_gather_tensor_autograd(local, dim, cp_group))
    if state is not None and state[1] == "record":
        state[0].record(gathered.detach())
    return gathered


def _slice_thd_chunk_for_cp(
    chunk: dict[str, torch.Tensor],
    *,
    cp_mesh,
    cp_group,
    cp_size: int,
    cp_rank: int,
    padding_token_id: int,
) -> dict[str, torch.Tensor]:
    total_tokens = int(chunk["input_ids"].shape[0])
    query_indices = contiguous_local_indices(cp_mesh, total_tokens, chunk["input_ids"].device)

    out: dict[str, torch.Tensor | int | str | object] = {
        "input_ids": chunk["input_ids"].index_select(0, query_indices).to(torch.int64).contiguous(),
        "labels": chunk["labels"].index_select(0, query_indices).to(torch.int64).contiguous(),
        "position_ids": chunk["position_ids"].index_select(0, query_indices).to(torch.int64).contiguous(),
        "cu_seqlens": chunk["cu_seqlens"].to(torch.int32).contiguous(),
        "qkv_format": "thd",
        "cp_size": cp_size,
        "cp_rank": cp_rank,
        "_glm_dsa_cp_group": cp_group,
        "glm_dsa_cp_query_indices": query_indices.to(torch.int32).contiguous(),
    }
    if "max_seqlen" in chunk:
        out["max_seqlen"] = chunk["max_seqlen"].to(torch.int32).contiguous()
    if "cu_seqlens_padded" in chunk:
        out["cu_seqlens_padded"] = chunk["cu_seqlens_padded"].to(torch.int32).contiguous()
    # Slice the pack-derived mask rather than re-deriving it from token values:
    # GLM's pad id is <|endoftext|>, which is also its first eos_token_id, so a
    # value comparison would drop every document-final eos from the MoE experts.
    if "padding_mask" in chunk:
        out["padding_mask"] = chunk["padding_mask"].index_select(0, query_indices).bool().contiguous()
    else:
        out["padding_mask"] = thd_padding_mask_from_token_ids(out["input_ids"], padding_token_id).contiguous()
    return out  # type: ignore[return-value]


def _packed_cp_layout(batch, *, num_chunks: int) -> ShardLayout | None:
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


def make_glm_dsa_packed_cp_batch_and_ctx(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
):
    """Convert packed GLM DSA batches to THD and keep a contiguous query shard per CP rank.

    GLM DSA sparse attention gathers K/V activations inside the model. The batch
    side only slices local query tokens and carries the full packed-sequence
    ``cu_seqlens`` plus per-query global token indices for TileLang's causal
    top-k window.
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

    stacked: dict[str, torch.Tensor | int | str | object] = {}
    for key, value in chunks[0].items():
        if isinstance(value, torch.Tensor):
            stacked[key] = torch.stack([chunk[key] for chunk in chunks])  # type: ignore[list-item]
        else:
            stacked[key] = value
    return contextlib.nullcontext, stacked


def shard_glm_dsa_packed_cp_batch(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
):
    """``ContextParallelSharder.shard_batch`` wrapper for GLM DSA packed CP."""
    layout = _packed_cp_layout(batch, num_chunks=num_chunks)
    ctx_factory, sharded_batch = make_glm_dsa_packed_cp_batch_and_ctx(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        num_chunks=num_chunks,
        seq_lens_padding_value=seq_lens_padding_value,
    )
    return ctx_factory, sharded_batch, layout
