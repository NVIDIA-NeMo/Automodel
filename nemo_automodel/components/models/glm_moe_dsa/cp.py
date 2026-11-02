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
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ShardLayout,
    contiguous_local_indices,
    round_robin_local_indices,
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

# Which global query positions a CP rank owns. "contiguous": rank r keeps tokens
# [r * T/cp, (r + 1) * T/cp). "head_tail": the sequence is cut into 2 * cp chunks and rank r keeps
# chunks r and 2 * cp - 1 - r, so the causal (quadratic) indexer work is the same on every rank;
# the top-k sparse attention itself is uniform per query and does not care.
QueryLayout = Literal["contiguous", "head_tail"]
QUERY_LAYOUTS: tuple[QueryLayout, ...] = ("contiguous", "head_tail")

# ``(from_global, to_global)`` row maps between the rank-order all-gather of a layout's shards and
# global token order, both int64 ``[total_tokens]``; see :func:`gather_order`.
GatherOrder = tuple[torch.Tensor, torch.Tensor]


def layout_indices_for_rank(
    layout: QueryLayout, rank: int, cp_size: int, total_tokens: int, device: torch.device
) -> torch.Tensor:
    """Global query positions rank ``rank`` of ``cp_size`` owns under ``layout``, in its local order.

    Mesh-free form of :func:`contiguous_local_indices` / :func:`round_robin_local_indices`, which the
    batch slicer uses for the calling rank; this one serves every rank so :func:`gather_order` can
    describe the whole gathered tensor.

    Args:
        layout: Query layout, see :data:`QueryLayout`.
        rank: Context-parallel rank in ``[0, cp_size)``.
        cp_size: Context-parallel world size.
        total_tokens: Global padded token count; a multiple of ``cp_size`` (contiguous) or
            ``2 * cp_size`` (head/tail).
        device: Device of the returned index tensor.

    Returns:
        Int64 tensor of shape ``[total_tokens / cp_size]`` with the rank's global token positions in
        local shard order (contiguous run, or head chunk then tail chunk).
    """
    if layout not in QUERY_LAYOUTS:
        raise ValueError(f"Unknown GLM DSA CP query layout {layout!r}; expected one of {QUERY_LAYOUTS}")
    if not 0 <= rank < cp_size:
        raise ValueError(f"rank must lie in [0, cp_size), got {rank=} {cp_size=}")
    if layout == "contiguous":
        if total_tokens % cp_size != 0:
            raise ValueError(f"total_tokens must be divisible by cp_size, got {total_tokens=} {cp_size=}")
        local_tokens = total_tokens // cp_size
        return torch.arange(rank * local_tokens, (rank + 1) * local_tokens, device=device, dtype=torch.long)
    if total_tokens % (2 * cp_size) != 0:
        raise ValueError(f"total_tokens must be divisible by 2 * cp_size, got {total_tokens=} {cp_size=}")
    chunk_len = total_tokens // (2 * cp_size)
    head = torch.arange(rank * chunk_len, (rank + 1) * chunk_len, device=device, dtype=torch.long)
    tail_start = (2 * cp_size - 1 - rank) * chunk_len
    tail = torch.arange(tail_start, tail_start + chunk_len, device=device, dtype=torch.long)
    return torch.cat((head, tail))


def local_query_indices(layout: QueryLayout, cp_mesh, total_tokens: int, device: torch.device) -> torch.Tensor:
    """Return this rank's global query positions for ``layout`` (see :data:`QueryLayout`)."""
    if layout == "head_tail":
        return round_robin_local_indices(cp_mesh, total_tokens, device)
    if layout == "contiguous":
        return contiguous_local_indices(cp_mesh, total_tokens, device)
    raise ValueError(f"Unknown GLM DSA CP query layout {layout!r}; expected one of {QUERY_LAYOUTS}")


def gather_order(layout: QueryLayout, cp_size: int, total_tokens: int, device: torch.device) -> GatherOrder | None:
    """Row maps between the rank-order all-gather of ``layout`` shards and global token order.

    ``all_gather_tensor_autograd`` concatenates the ranks' shards in rank order, so gathered row
    ``r * T/cp + j`` holds the ``j``-th local token of rank ``r``. Every consumer of the gathered K/V
    (the cuDNN packed metadata's key sources and causal windows, the top-k coordinates FlashMLA and
    cuDNN read ``kv_latent`` with, the TileLang causal windows) addresses rows by global token
    position, which coincides with rank order only for the contiguous layout.

    Args:
        layout: Query layout the batch was sliced with.
        cp_size: Context-parallel world size.
        total_tokens: Global padded token count (local tokens times ``cp_size``).
        device: Device of the returned index tensors.

    Returns:
        ``None`` when rank order already is global order (contiguous layout, or ``cp_size <= 1``);
        otherwise ``(from_global, to_global)``: ``from_global[j]`` is the global token in gathered row
        ``j`` and ``to_global[i]`` is the gathered row holding global token ``i``.
    """
    if layout not in QUERY_LAYOUTS:
        raise ValueError(f"Unknown GLM DSA CP query layout {layout!r}; expected one of {QUERY_LAYOUTS}")
    if cp_size <= 1 or layout == "contiguous":
        return None
    from_global = torch.cat(
        [layout_indices_for_rank(layout, rank, cp_size, total_tokens, device) for rank in range(cp_size)]
    )
    to_global = torch.empty_like(from_global)
    to_global[from_global] = torch.arange(total_tokens, device=device, dtype=torch.long)
    return from_global, to_global


def validate_gather_order(order: GatherOrder, cp_rank: int, local_tokens: int, query_indices: torch.Tensor) -> None:
    """Check that ``order`` describes the shard this rank actually received.

    Args:
        order: Result of :func:`gather_order` for the model's configured layout.
        cp_rank: This rank's index in the context-parallel group.
        local_tokens: Number of local query rows.
        query_indices: The batch's ``glm_dsa_cp_query_indices`` (global positions of the local rows).

    Raises:
        ValueError: If the batch was sliced with a different layout than the model's gather order
            assumes, which would feed the kernels wrong key rows.
    """
    from_global, _ = order
    expected = from_global[cp_rank * local_tokens : (cp_rank + 1) * local_tokens]
    actual = query_indices.flatten().to(device=expected.device, dtype=expected.dtype)
    if actual.numel() != local_tokens or not torch.equal(expected, actual):
        raise ValueError(
            "glm_dsa_cp_query_indices do not match BackendConfig.cp_query_layout: the batch was sliced with "
            "a different context-parallel query layout than the one the model's K/V gather assumes."
        )


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


class _GatherToGlobal(torch.autograd.Function):
    """Permute a rank-order gather into global token order; the backward permutes the gradient back."""

    @staticmethod
    def forward(ctx: Any, gathered: torch.Tensor, from_global: torch.Tensor, to_global: torch.Tensor, dim: int):
        """Reorder ``gathered`` so that index ``i`` along ``dim`` holds global token ``i``.

        Args:
            ctx: Autograd context.
            gathered: Rank-order all-gather result.
            from_global: Global token held by each gathered row (see :func:`gather_order`).
            to_global: Gathered row holding each global token.
            dim: Gather dimension.

        Returns:
            The gathered tensor in global token order.
        """
        ctx.dim = dim
        ctx.save_for_backward(from_global)
        return gathered.index_select(dim, to_global)

    @staticmethod
    def backward(ctx: Any, grad_global: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Map the global-order gradient back onto the rank-order gather."""
        (from_global,) = ctx.saved_tensors
        return grad_global.index_select(ctx.dim, from_global), None, None, None


class _ReplayedAllGather(torch.autograd.Function):
    """Return a recorded all-gather result while keeping the gradient path of the gather."""

    @staticmethod
    def forward(
        ctx: Any,
        local: torch.Tensor,
        gathered: torch.Tensor,
        dim: int,
        cp_group,
        from_global: torch.Tensor | None,
    ) -> torch.Tensor:
        """Hand back ``gathered`` (the forward's all-gather of ``local``) without communicating.

        Args:
            ctx: Autograd context.
            local: This rank's shard, only used for its gradient shape and dtype.
            gathered: The recorded all-gather of ``local`` across the CP group along ``dim``, in global
                token order when ``from_global`` is given and in rank order otherwise.
            dim: Gather dimension.
            cp_group: CP process group.
            from_global: Global token held by each rank-order gathered row when the recorded tensor
                was reordered into global order (see :func:`gather_order`); ``None`` for rank order.

        Returns:
            ``gathered`` as a fresh view.
        """
        del local
        ctx.dim = dim
        ctx.cp_group = cp_group
        ctx.reordered = from_global is not None
        if from_global is not None:
            ctx.save_for_backward(from_global)
        return gathered.view_as(gathered)

    @staticmethod
    def backward(ctx: Any, grad_gathered: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        """Reduce-scatter the gathered gradient back to each rank's shard, as the gather's backward does."""
        if ctx.reordered:
            (from_global,) = ctx.saved_tensors
            grad_gathered = grad_gathered.index_select(ctx.dim, from_global)
        grad_local = _wait(funcol.reduce_scatter_tensor(grad_gathered.contiguous(), "sum", ctx.dim, ctx.cp_group))
        return grad_local, None, None, None, None


def glm_dsa_cp_all_gather(
    tensor: torch.Tensor,
    *,
    dim: int,
    cp_group,
    replay: RecomputeReplay[torch.Tensor] | None = None,
    order: GatherOrder | None = None,
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
        order: Optional :func:`gather_order` maps; when given, the result is permuted from rank
            order into global token order (and the backward permutes the gradient back). Required
            for every layout whose rank order is not global order, such as head/tail.

    Returns:
        Tensor whose ``dim`` holds every rank's shard in rank order, or in global token order when
        ``order`` is given.
    """
    if not glm_dsa_cp_enabled(cp_group):
        return tensor
    local = tensor.contiguous()
    from_global = order[0] if order is not None else None
    state = replay.current() if replay is not None else None
    if state is not None and state[1] == "replay":
        recorded = state[0].take()
        if recorded is not None:
            return _ReplayedAllGather.apply(local, recorded, dim, cp_group, from_global)
    gathered = _wait(funcol.all_gather_tensor_autograd(local, dim, cp_group))
    if order is not None:
        gathered = _GatherToGlobal.apply(gathered, order[0], order[1], dim)
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
    query_layout: QueryLayout = "contiguous",
) -> dict[str, torch.Tensor]:
    total_tokens = int(chunk["input_ids"].shape[0])
    query_indices = local_query_indices(query_layout, cp_mesh, total_tokens, chunk["input_ids"].device)

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
    query_layout: QueryLayout = "contiguous",
):
    """Convert packed GLM DSA batches to THD and keep one query shard per CP rank.

    GLM DSA sparse attention gathers K/V activations inside the model. The batch
    side only slices local query tokens (``query_layout`` decides which, see
    :data:`QueryLayout`) and carries the full packed-sequence ``cu_seqlens`` plus
    per-query global token indices for the kernels' causal top-k windows.
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
            query_layout=query_layout,
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
                query_layout=query_layout,
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
    query_layout: QueryLayout = "contiguous",
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
        query_layout=query_layout,
    )
    return ctx_factory, sharded_batch, layout
