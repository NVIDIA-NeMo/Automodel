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

"""Context-parallel batch-sharding contract.

Every CP backend is a :class:`CPShardStrategy`. A model that owns its CP batch
sharding and attention transport returns one directly from
``prepare_model_inputs_for_cp``;
the framework constructs its own for the remaining backends (torch
``context_parallel`` round-robin, TE/THD, MagiAttention) so the public sharder
reduces to resolving a sharder and calling ``shard_batch``. This replaces the retired private batch keys
(``_cp_make_batch_fn``, ``_cp_metadata_seq_dims``, ``_cp_metadata_pad_values``,
``_cp_full_logits_grad_touch``).

The contract is a closed verb set with open implementations: the dataclass
slots are plain callables filled with functions from the owning model's
directory or from the framework. ``local_token_global_indices`` — the global
position of every local token — is the universal layout coordinate system: the
default token-tensor shard/gather are synthesized from it, so a sharder only
overrides them when it has a cheaper communication pattern. Layouts that are a
pure function of ``(cp_mesh, padded_seq_len)`` (contiguous, round-robin)
provide it at construction; data-dependent layouts (THD ``cu_seqlens``
partitioning, magi's dispatch solver) report the partition they computed as
:class:`CPShardResult`. The public sharder combines that result with the mesh
into an immutable :class:`CPForward`, so backend strategy never carries mutable
per-batch state.
The sharder carries no backend tag: nothing may branch on which backend
produced it.

This module also hosts the framework's ``shard_batch`` implementations: the
shared contiguous-shard batch prep used by models whose CP ranks own contiguous
sequence slices (Gemma4, DeepSeek V4), and the torch ``context_parallel``
round-robin load-balanced prep with its index map. The TE/THD and magi preps
live with their dependencies (``utils``, ``magi``); the
dispatcher wraps them into sharders at resolution time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


@contextlib.contextmanager
def _sdpa_context(cp_context: AbstractContextManager[None]):
    """Enter the SDPA backend and context-parallel contexts for one forward."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]), cp_context:
        yield


def _create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str | None = None,
) -> AbstractContextManager[None]:
    """Create the PyTorch SDPA context used by the generic CP sharder.

    Args:
        cp_mesh: One-dimensional context-parallel device mesh.
        cp_buffers: Tensors of arbitrary rank with sequence axes selected by
            ``cp_seq_dims``. The context shards these tensors in place while active.
        cp_seq_dims: Sequence-axis index for each tensor in ``cp_buffers``.
        cp_no_restore_buffers: Members of ``cp_buffers`` that remain in their
            per-rank local layout after the context exits.
        cp_rotate_method: Optional Q/K/V rotation implementation.

    Returns:
        Context manager that applies generic SDPA context parallelism.
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


def _cp_rank(cp_mesh) -> int:
    """Resolve this rank's index within the CP submesh (0 without distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=cp_mesh.get_group())
    return getattr(cp_mesh, "get_local_rank", lambda: 0)()


def _contiguous_local_indices(cp_mesh, padded_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Global positions owned by this rank under contiguous rank-ordered sharding.

    Rank ``r`` owns ``[r * L, (r + 1) * L)`` with ``L = padded_seq_len // cp_size``.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        padded_seq_len: Global sequence length after CP padding; must be
            divisible by the CP size.
        device: Device for the returned index tensor.

    Returns:
        A 1-D int64 tensor of this rank's global token positions.
    """
    cp_size = cp_mesh.size()
    if padded_seq_len % cp_size != 0:
        raise ValueError(f"padded_seq_len must be divisible by cp_size, got {padded_seq_len=} {cp_size=}")
    local_len = padded_seq_len // cp_size
    start = _cp_rank(cp_mesh) * local_len
    return torch.arange(start, start + local_len, device=device, dtype=torch.long)


def _identity_local_indices(cp_mesh, padded_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Global positions owned by this rank when CP is inactive: all of them.

    Index map of the identity sharder, so consumers run the same
    token-verb code path at cp_size <= 1 (shard/gather become identities).
    """
    del cp_mesh
    return torch.arange(padded_seq_len, device=device, dtype=torch.long)


def _shard_batch_identity(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id: int = 0):
    """``shard_batch`` of the identity sharder: a no-op with a trivial token layout.

    Returned by the dispatch when no CP prep applies, so callers hold working
    token verbs at every cp_size (nothing was padded: original == padded).
    """
    del cp_mesh, tp_mesh, loss_mask, padding_token_id
    primary = batch.get("inputs_embeds", batch.get("input_ids"))
    if primary is not None and primary.dim() >= 2:
        return CPShardResult(
            contextlib.nullcontext(),
            batch,
            original_seq_len=primary.shape[1],
            padded_seq_len=primary.shape[1],
        )
    return CPShardResult(contextlib.nullcontext(), batch)


def _round_robin_local_indices(cp_mesh, padded_seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Global positions owned by this rank under torch ``context_parallel`` load balancing.

    The sequence splits into ``2 * cp_size`` chunks and rank ``r`` owns chunks
    ``r`` and ``2 * cp_size - 1 - r`` (head-tail pairing, so causal-attention
    work stays even across ranks).

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        padded_seq_len: Global sequence length after CP padding; must be
            divisible by ``2 * cp_size``.
        device: Device for the returned index tensor.

    Returns:
        A 1-D int64 tensor of this rank's global token positions, in local
        shard order (head chunk then tail chunk).
    """
    cp_size = cp_mesh.size()
    if padded_seq_len % (2 * cp_size) != 0:
        raise ValueError(f"padded_seq_len must be divisible by 2 * cp_size, got {padded_seq_len=} {cp_size=}")
    chunk_len = padded_seq_len // (2 * cp_size)
    rank = _cp_rank(cp_mesh)
    head = torch.arange(rank * chunk_len, (rank + 1) * chunk_len, device=device, dtype=torch.long)
    tail_start = (2 * cp_size - 1 - rank) * chunk_len
    tail = torch.arange(tail_start, tail_start + chunk_len, device=device, dtype=torch.long)
    return torch.cat((head, tail))


def _shard_token_tensor_by_indices(tensor: torch.Tensor, local_indices: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
    """Keep this rank's slice of a full-length token-aligned tensor.

    Args:
        tensor: Full (padded) sequence tensor, e.g. ``[B, S]`` advantages.
        local_indices: This rank's global token positions.
        seq_dim: The sequence dimension of ``tensor``.

    Returns:
        The local shard, laid out identically to the model inputs.
    """
    return tensor.index_select(seq_dim, local_indices.to(tensor.device)).contiguous()


def _reorder_gathered_token_tensor(
    parts: list[torch.Tensor], index_parts: list[torch.Tensor], seq_dim: int = 1
) -> torch.Tensor:
    """Reassemble rank-order gathered shards into global sequence order.

    Args:
        parts: Per-rank local shards, in rank order.
        index_parts: Per-rank global token positions, in the same order.
        seq_dim: The sequence dimension of the shards.

    Returns:
        The full-sequence tensor with position ``i`` holding the token whose
        global index is ``i``.
    """
    full = torch.cat(parts, dim=seq_dim)
    full_indices = torch.cat([p.to(full.device) for p in index_parts])
    return full.index_select(seq_dim, torch.argsort(full_indices)).contiguous()


def _gather_token_tensor_by_indices(
    cp_mesh, tensor: torch.Tensor, local_indices: torch.Tensor, seq_dim: int = 1
) -> torch.Tensor:
    """Differentiably gather a token-aligned local shard back to the full sequence.

    Uses ``torch.distributed.nn.functional.all_gather`` so gradients route back
    to the owning ranks, then restores global sequence order from the gathered
    per-rank indices. Identity when CP is inactive.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        tensor: This rank's local shard, e.g. ``[B, S/cp]`` token logprobs.
        local_indices: This rank's global token positions.
        seq_dim: The sequence dimension of ``tensor``.

    Returns:
        The full-sequence tensor in global order (padding not trimmed).
    """
    if cp_mesh is None or cp_mesh.size() <= 1 or not (dist.is_available() and dist.is_initialized()):
        return tensor

    from torch.distributed.nn.functional import all_gather  # noqa: PLC0415

    group = cp_mesh.get_group()
    parts = list(all_gather(tensor.contiguous(), group=group))
    local_indices = local_indices.to(tensor.device)
    index_parts = [torch.empty_like(local_indices) for _ in range(cp_mesh.size())]
    dist.all_gather(index_parts, local_indices.contiguous(), group=group)
    return _reorder_gathered_token_tensor(parts, index_parts, seq_dim=seq_dim)


@dataclass(frozen=True)
class CPShardResult:
    """Prepared batch, forward context, and captured token layout.

    Attributes:
        context: Context manager entered around the model forward.
        batch: Prepared mapping whose token-aligned tensors use backend-specific
            per-rank layouts such as [batch, local_sequence, ...] or packed THD
            [local_tokens, ...].
        local_token_global_indices: Tensor of arbitrary shape containing this
            rank's global token positions for data-dependent layouts; flattened
            to [local_tokens] when consumed.
        original_seq_len: Global sequence length before CP padding.
        padded_seq_len: Global sequence length after CP padding.
        input_row_shape: Original leading [batch, sequence] shape when the
            prepared token stream was flattened to THD.
        input_token_stream_positions: Tensor of shape [batch, input_sequence]
            mapping input positions to padded output columns, with -1 for
            dropped input padding.
    """

    context: AbstractContextManager[None]
    batch: dict[str, Any]
    local_token_global_indices: torch.Tensor | None = None
    original_seq_len: int | None = None
    padded_seq_len: int | None = None
    input_row_shape: tuple[int, ...] | None = None
    input_token_stream_positions: torch.Tensor | None = None


class _ShardBatch(Protocol):
    """Callable contract implemented by context-parallel batch strategies."""

    def __call__(
        self,
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
        batch: dict[str, Any],
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
    ) -> CPShardResult:
        """Shard sequence-bearing tensors in ``batch`` and report their layout."""


_LocalIndexFn = Callable[[DeviceMesh | None, int, torch.device | None], torch.Tensor]


@dataclass(frozen=True)
class CPShardStrategy:
    """Immutable CP backend strategy used while preparing one batch.

    Attributes:
        shard_batch: ``(cp_mesh, tp_mesh, batch, *, loss_mask=None,
            padding_token_id=0) -> CPShardResult``.
            Pads and shards the batch, installs any backend-owned attention
            transport, and reports the shard layout it computed; the dispatch
            binds it into the returned :class:`CPForward`.
        local_token_global_indices: ``(cp_mesh, padded_seq_len, device) ->
            LongTensor`` with the global position of each local token —
            closed-form for contiguous/round-robin layouts; None for
            data-dependent layouts, whose partition arrives in the returned
            :class:`CPShardResult`.
    """

    shard_batch: _ShardBatch
    local_token_global_indices: _LocalIndexFn | None

    @classmethod
    def identity(cls) -> "CPShardStrategy":
        """Build the no-op strategy used when context parallelism is inactive."""
        return cls(shard_batch=_shard_batch_identity, local_token_global_indices=_identity_local_indices)

    @classmethod
    def sdpa(cls) -> "CPShardStrategy":
        """Build the generic load-balanced SDPA context-parallel strategy.

        Returns:
            Sharder using head-tail round-robin token ownership and PyTorch's
            SDPA context-parallel attention implementation.
        """
        return cls(
            shard_batch=_shard_batch_load_balanced,
            local_token_global_indices=_round_robin_local_indices,
        )

    @classmethod
    def sdpa_aux(cls) -> "CPShardStrategy":
        """Build SDPA CP that shards auxiliary streams while models shard embeddings."""
        return cls(shard_batch=_shard_batch_aux_only, local_token_global_indices=_round_robin_local_indices)

    @classmethod
    def contiguous(cls, shard_batch: _ShardBatch) -> "CPShardStrategy":
        """Bind a model-owned batch preparation to contiguous token ownership."""
        return cls(shard_batch=shard_batch, local_token_global_indices=_contiguous_local_indices)


@dataclass(frozen=True)
class CPForward:
    """Prepared context-parallel state for one model forward.

    Token operations use the exact layout captured while preparing ``batch``.
    This is not a ``DTensor``: round-robin, packed THD, and Magi layouts cannot
    be represented by a standard ``Shard(dim)`` placement.

    Attributes:
        context: Context manager entered around the model forward.
        batch: Prepared mapping whose token-aligned tensors use backend-specific
            per-rank layouts such as [batch, local_sequence, ...] or packed THD
            [local_tokens, ...].
    """

    _result: CPShardResult
    _cp_mesh: DeviceMesh | None
    _local_token_global_indices: _LocalIndexFn | None

    @property
    def context(self) -> AbstractContextManager[None]:
        """Context manager entered around the model forward."""
        return self._result.context

    @property
    def batch(self) -> dict[str, Any]:
        """Prepared batch containing this rank's local token tensors."""
        return self._result.batch

    def _indices(self, padded_seq_len: int, device: torch.device) -> torch.Tensor:
        captured = self._result.local_token_global_indices
        if captured is not None:
            # Data-dependent layout: use the partition the shard reported, and
            # validate the requested length against it so a mismatched tensor
            # cannot be silently mis-sharded.
            cp_size = self._cp_mesh.size() if self._cp_mesh is not None else 1
            expected = captured.numel() * cp_size
            if padded_seq_len != expected:
                raise ValueError(
                    f"the reported CP indices cover a padded stream of {expected} tokens "
                    f"({captured.numel()} local x cp_size {cp_size}), got {padded_seq_len=}. "
                    "The tensor does not match the batch used to create this token layout."
                )
            return captured.reshape(-1).to(device=device, dtype=torch.long)
        if self._local_token_global_indices is None:
            raise RuntimeError("The prepared CP token layout does not contain a local-token index map")
        return self._local_token_global_indices(self._cp_mesh, padded_seq_len, device)

    def shard(self, tensor: torch.Tensor, *, seq_dim: int = 1, fill: float | int | None = None) -> torch.Tensor:
        """Shard a full-length token-aligned tensor exactly like the model inputs.

        The caller may pass tensors in its own coordinates and this method
        applies the same transform as the prepared model inputs:

        - ``[B, S_in]`` tensors on a repositioned-row layout (reported position
          map, e.g. DSV4 packed repad) are scattered into the padded rows,
          ``fill`` filling the pad slots;
        - tensors matching the reported pre-flatten ``input_row_shape`` on a
          flat-stream (THD) layout are flattened first (the returned shard is
          in the model's local stream coordinate);
        - tensors of ``original_seq_len`` are right-padded to
          ``padded_seq_len`` with the explicit ``fill`` value;
        - tensors already at ``padded_seq_len`` shard directly.

        Any other length raises instead of silently sharding the wrong slice.

        Args:
            tensor: Token-aligned tensor of shape ``[..., sequence, ...]`` in
                caller coordinates. ``seq_dim`` selects its sequence axis.
            seq_dim: Sequence-axis index in ``tensor``.
            fill: Value used when caller-coordinate tensors require CP padding.

        Returns:
            Per-rank local tensor with the same non-sequence axes and token order
            used by the prepared model inputs.
        """
        result = self._result
        if result.input_token_stream_positions is not None and tuple(tensor.shape) == tuple(
            result.input_token_stream_positions.shape
        ):
            if fill is None:
                raise ValueError("sharding an input-coordinate tensor on a repositioned layout requires `fill`")
            positions = result.input_token_stream_positions.to(tensor.device)
            valid = positions >= 0
            padded = torch.full(
                (tensor.shape[0], result.padded_seq_len), fill, dtype=tensor.dtype, device=tensor.device
            )
            padded[valid.nonzero(as_tuple=True)[0], positions[valid]] = tensor[valid]
            tensor, seq_dim = padded, 1
        elif result.input_row_shape is not None and tuple(tensor.shape[: len(result.input_row_shape)]) == tuple(
            result.input_row_shape
        ):
            tensor = tensor.reshape(-1, *tensor.shape[len(result.input_row_shape) :])
            seq_dim = 0

        length = tensor.shape[seq_dim]
        if result.padded_seq_len is not None and length != result.padded_seq_len:
            if fill is not None and result.original_seq_len is not None and length == result.original_seq_len:
                tensor = _pad_tensor_seq_dim_(tensor, seq_dim, result.padded_seq_len - length, fill)
            else:
                raise ValueError(
                    f"This CP token layout has padded_seq_len={result.padded_seq_len} "
                    f"(original_seq_len={result.original_seq_len}), got a tensor of length {length} on dim {seq_dim}. "
                    "Pass the original-length tensor with an explicit `fill`, or pre-pad it yourself."
                )
        indices = self._indices(tensor.shape[seq_dim], tensor.device)
        return _shard_token_tensor_by_indices(tensor, indices, seq_dim=seq_dim)

    def gather(
        self,
        tensor: torch.Tensor,
        *,
        seq_dim: int = 1,
        fill: float | int | None = None,
    ) -> torch.Tensor:
        """Differentiably gather a token-aligned local shard to global order.

        The result is returned in the caller's original coordinates using the
        shard layout: sliced back to ``original_seq_len``,
        un-flattened to ``input_row_shape`` (THD), or mapped through the
        reported position map (``fill`` for input positions whose tokens were
        dropped, e.g. re-padded pack slots). Without captured layout facts, the
        gathered global tensor is returned unchanged.

        Args:
            tensor: Per-rank local tensor of shape ``[..., local_sequence, ...]``
                in the prepared CP token order. ``seq_dim`` selects its sequence
                axis.
            seq_dim: Sequence-axis index in ``tensor``.
            fill: Value used for caller positions dropped by a repositioning
                layout.

        Returns:
            Replicated tensor restored to the caller's original coordinates.
        """
        result = self._result
        padded_seq_len = tensor.shape[seq_dim] * (self._cp_mesh.size() if self._cp_mesh is not None else 1)
        indices = self._indices(padded_seq_len, tensor.device)
        full = _gather_token_tensor_by_indices(self._cp_mesh, tensor, indices, seq_dim=seq_dim)
        if result.padded_seq_len is not None and full.shape[seq_dim] != result.padded_seq_len:
            raise ValueError(
                f"gathered length {full.shape[seq_dim]} on dim {seq_dim} != reported "
                f"padded_seq_len {result.padded_seq_len}; the local shard does not match "
                "the batch used to create this token layout (or no collective ran)."
            )
        if result.input_token_stream_positions is not None:
            if fill is None:
                raise ValueError("trimming to input coordinates on a repositioned layout requires `fill`")
            positions = result.input_token_stream_positions.to(full.device)
            out = full.gather(1, positions.clamp(min=0).to(torch.long))
            return out.masked_fill(positions < 0, fill)
        if result.input_row_shape is not None:
            return full.reshape(*result.input_row_shape, *full.shape[seq_dim + 1 :])
        if result.original_seq_len is not None:
            return full.narrow(seq_dim, 0, result.original_seq_len)
        return full


# ---------------------------------------------------------------------------
# Shared contiguous-shard batch implementation.
#
# Each CP rank keeps one contiguous ``seq_start:seq_end`` slice; no collective
# happens here — the transport lives in the owning model's attention (Gemma4
# ring FlexAttention, DSV4 K/V all-gather). This is the non-load-balanced peer
# of the ``context_parallel`` and TE/THD batch shardings.
# ---------------------------------------------------------------------------


def _pad_tensor_seq_dim_(tensor: torch.Tensor, seq_dim: int, pad_len: int, value: float | int = 0) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=seq_dim)


def _pad_position_ids_seq_dim_(position_ids: torch.Tensor, seq_dim: int, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return position_ids
    last_position = position_ids.select(seq_dim, position_ids.shape[seq_dim] - 1).unsqueeze(seq_dim)
    increment_shape = [1] * position_ids.ndim
    increment_shape[seq_dim] = pad_len
    increments = torch.arange(1, pad_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(
        increment_shape
    )
    return torch.cat((position_ids, last_position + increments), dim=seq_dim)


def convert_attention_mask_to_padding_mask(batch: dict) -> None:
    """Pop ``attention_mask`` and derive ``padding_mask`` (True == pad) in place.

    Preserves padding semantics for modules such as MoE routers after the
    attention mask is stripped for CP. Idempotent: a no-op when
    ``padding_mask`` already exists or there is no ``attention_mask``.
    """
    attention_mask = batch.pop("attention_mask", None)
    if attention_mask is not None and "padding_mask" not in batch:
        if attention_mask.ndim == 4:
            diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
            batch["padding_mask"] = diagonal.logical_not() if attention_mask.dtype == torch.bool else diagonal != 0
        else:
            batch["padding_mask"] = attention_mask.bool().logical_not()


def shard_batch_contiguous(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    pad_multiple: int = 1,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, Any] | None = None,
    shard_primary: bool = True,
):
    """Prepare and contiguously shard a batch for model-owned CP.

    Normalizes the batch (mask conversion, position_ids, labels), pads the
    sequence to ``cp_size * max(pad_multiple, 2)``, then keeps one contiguous
    ``seq_start:seq_end`` slice per CP rank.

    ``shard_primary`` is the only behavioral switch. When True (default) the
    primary stream (``input_ids`` / ``inputs_embeds``) is padded and sliced with
    the aux streams — the dispatch-level shard (e.g. DSV4). When False the primary
    and pixel streams are left FULL-length for a model that embeds and slices them
    inside its own forward per microbatch (e.g. Gemma4; see
    :func:`shard_sequence_for_cp_contiguous`). The padded layout is identical
    either way, so the two paths stay bit-for-bit slice-equivalent.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        tp_mesh: The tensor-parallel device (sub)mesh (or None).
        batch: The full-sequence batch; mutated and sharded in place.
        loss_mask: Optional per-token loss mask; used as labels when the batch
            has none, otherwise sharded alongside the batch.
        padding_token_id: Pad sentinel for ``input_ids``.
        pad_multiple: Required per-CP-rank shard length multiple (e.g. DSV4's
            compress-ratio LCM). The effective divisor is
            ``cp_size * max(pad_multiple, 2)``.
        extra_seq_keys: Model-specific per-token batch keys to pad and shard,
            mapped to their sequence dim (e.g. Gemma4 vision group ids).
        extra_pad_values: Pad sentinels for ``extra_seq_keys`` (default 0).
        shard_primary: When False, leave the primary/pixel streams full-length for
            in-forward slicing (aux-only shard); ``CPShardResult.padded_seq_len``
            is then the length the model must pad its primary to before slicing.

    Returns:
        Prepared batch and layout with a null context; transport lives in the
        model's own attention.
    """
    # --- normalize: mask -> padding_mask, primary tensor, position_ids, labels
    convert_attention_mask_to_padding_mask(batch)

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "the CP dispatch requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary_seq_tensor = batch[primary_key]
    seq_len = primary_seq_tensor.shape[1]

    batch_size = primary_seq_tensor.shape[0]
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(0, seq_len, device=primary_seq_tensor.device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    else:
        position_ids = batch["position_ids"]
        if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
            batch["position_ids"] = position_ids.expand(batch_size, -1).contiguous()

    position_ids = batch["position_ids"]
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels = batch.get("labels")
    if labels is None and loss_mask is not None:
        labels = loss_mask
        loss_mask = None
    if labels is None:
        raise KeyError("Context parallelism requires `labels` in the batch, or labels passed as `loss_mask`.")

    # --- pad to the CP divisor
    cp_size = cp_mesh.size()
    # Batch-resident sequence tensors, each sharded on seq dim 1 with its own pad
    # sentinel. ``position_ids``/``labels``/``loss_mask`` are sharded too but handled
    # separately below (special pad logic, or carried as locals rather than in the
    # batch dict during padding). This single table drives padding, slicing, and the
    # known-key set so a new field is declared in one place. The primary stream is
    # excluded on the aux-only path (``shard_primary=False``): the model embeds and
    # slices it in forward, so it must stay full-length here.
    seq_pad_values = {"padding_mask": True}
    if shard_primary:
        seq_pad_values = {"input_ids": padding_token_id, "inputs_embeds": 0, "padding_mask": True}
    known_sequence_keys = set(seq_pad_values) | {"labels", "position_ids", "loss_mask"}

    # Extra per-token metadata (e.g. Gemma4 vision group ids) is sharded like the
    # known sequence tensors, using model-provided seq dims / pad values.
    metadata_seq_dims = extra_seq_keys or {}
    metadata_pad_values = extra_pad_values or {}
    extra_metadata_keys = [key for key in metadata_seq_dims if key in batch and key not in known_sequence_keys]

    divisor = cp_size * max(int(pad_multiple or 1), 2)
    pad_len = (-seq_len) % divisor
    if pad_len:
        for key, pad_val in seq_pad_values.items():
            if key in batch:
                batch[key] = _pad_tensor_seq_dim_(batch[key], 1, pad_len, pad_val)
        labels = _pad_tensor_seq_dim_(labels, 1, pad_len, -100)
        position_ids = _pad_position_ids_seq_dim_(position_ids, pos_seq_dim, pad_len)
        batch["position_ids"] = position_ids
        if loss_mask is not None:
            loss_mask = _pad_tensor_seq_dim_(loss_mask, 1, pad_len, 0)
        for key in extra_metadata_keys:
            batch[key] = _pad_tensor_seq_dim_(
                batch[key],
                metadata_seq_dims[key],
                pad_len,
                metadata_pad_values.get(key, 0),
            )

    # --- contiguous sequence slicing. Every CP rank in the same CP group starts
    # from the same full batch, then keeps one contiguous sequence shard. The
    # padded length equals the primary's post-pad length whether or not the
    # primary itself was padded here, so the aux-only slice matches the primary
    # slice the model computes with :func:`shard_sequence_for_cp_contiguous`.
    batch["labels"] = labels
    cp_rank = _cp_rank(cp_mesh)

    # The primary-inclusive shard just padded the primary, so its actual length is
    # authoritative and keeps the divisibility guard meaningful; the aux-only shard
    # leaves the primary full-length, so use the intended padded length the model
    # will pad its own primary to.
    padded_seq_len = batch[primary_key].shape[1] if shard_primary else seq_len + pad_len
    if padded_seq_len % cp_size != 0:
        raise ValueError(
            f"CP sequence length must be divisible by cp_size after padding, got {padded_seq_len=} {cp_size=}"
        )
    local_seq_len = padded_seq_len // cp_size
    seq_start = cp_rank * local_seq_len
    seq_end = seq_start + local_seq_len

    def _slice_seq(key: str, seq_dim: int = 1) -> None:
        if key not in batch:
            return
        slices = [slice(None)] * batch[key].ndim
        slices[seq_dim] = slice(seq_start, seq_end)
        batch[key] = batch[key][tuple(slices)].contiguous()

    for key in seq_pad_values:
        _slice_seq(key, 1)
    _slice_seq("labels", 1)
    _slice_seq("position_ids", pos_seq_dim)
    for key in extra_metadata_keys:
        _slice_seq(key, metadata_seq_dims[key])
    if loss_mask is not None:
        batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

    return CPShardResult(
        contextlib.nullcontext(),
        batch,
        original_seq_len=seq_len,
        padded_seq_len=padded_seq_len,
    )


# ---------------------------------------------------------------------------
# Round-robin load-balanced shard batch implementation.
#
# The framework's default CP path: torch ``context_parallel`` shards every
# buffer into 2*cp head-tail-paired chunks (see ``round_robin_local_indices``)
# and the transport is ring SDPA installed by the returned context.
# ---------------------------------------------------------------------------


# Per-buffer pad sentinels for the round-robin CP shard: each tensor's "ignore"
# value is semantic, not dtype-derived. ``labels``/``padding_mask``/
# ``attention_mask`` are all int/bool but have different ignore conventions.
# Falling through to 0 for ``padding_mask`` (== False == "real token") would tell
# the MoE router to route the cp-pad slots to experts -- silently wasting
# capacity and skewing load-balance loss. Everything else (input_ids,
# position_ids, ...) falls through to 0.
_ROUND_ROBIN_PAD_FILL = {
    "labels": -100,  # CE ignore_index
    "padding_mask": True,  # bool: True == "this position is pad, ignore"
    "attention_mask": False,  # HF: 0 == "this position is pad, ignore"
}


def _normalize_cp_primary_and_positions(batch) -> tuple[str, torch.Tensor, int, int, torch.Tensor, int]:
    """Resolve the primary stream and normalize ``position_ids`` for a CP shard.

    Pops nothing; determines whether the batch carries ``inputs_embeds``
    (``[batch, sequence, hidden]``) or ``input_ids`` (``[batch, sequence]``),
    injects a 1-D ``position_ids`` arange when absent, and expands a shared
    ``[1, sequence]`` position row to the batch size.

    Args:
        batch: The full-sequence batch; ``position_ids`` is added/expanded in place.

    Returns:
        ``(primary_key, primary_seq_tensor, seq_len, batch_size, position_ids,
        pos_seq_dim)`` where ``pos_seq_dim`` is 2 for mRoPE ``[3, batch,
        sequence]`` position ids and 1 otherwise.
    """
    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    assert has_inputs_embeds ^ has_input_ids, (
        "the CP dispatch requires exactly one of 'inputs_embeds' or 'input_ids' in batch"
    )
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary_seq_tensor = batch[primary_key]
    seq_len = primary_seq_tensor.shape[1]
    batch_size = primary_seq_tensor.shape[0]

    # Skip 1D injection if position_ids already in batch (e.g. mRoPE pre-computed)
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(0, seq_len, device=primary_seq_tensor.device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    else:
        position_ids = batch["position_ids"]
        if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
            batch["position_ids"] = position_ids.expand(batch_size, -1).contiguous()

    position_ids = batch["position_ids"]
    # mRoPE: [3, B, S] → shard on dim 2; standard: [B, S] → shard on dim 1
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1
    return primary_key, primary_seq_tensor, seq_len, batch_size, position_ids, pos_seq_dim


def _pad_cp_buffers_to_divisor(
    cp_buffers, cp_seq_dims, cp_no_restore_buffers, batch_buffer_keys, seq_len, cp_divisor, batch
):
    """Right-pad every CP buffer on its sequence dim to ``cp_divisor`` in place.

    Pads sequence length to be divisible by ``2 * cp_size`` (required by
    ``context_parallel`` load balancing); the inputs_embeds path can hit
    arbitrary seq lengths from the VLM collator, so padding happens here rather
    than relying on dataset-side padding. Each padded, batch-sourced buffer is
    mirrored back into ``batch`` so downstream dict readers see the padded shape.

    Args:
        cp_buffers: Sequence-aligned tensors, each ``[..., sequence, ...]``,
            padded and replaced in place.
        cp_seq_dims: Per-buffer sequence axis.
        cp_no_restore_buffers: Set of buffers the CP context must not restore;
            rebuilt to reference the padded tensors.
        batch_buffer_keys: Map from buffer index to its batch key (buffers not
            sourced from the batch are omitted).
        seq_len: The pre-pad sequence length.
        cp_divisor: ``2 * cp_size``.
        batch: The batch dict; padded batch-sourced buffers are mirrored back.

    Returns:
        ``(cp_buffers, cp_no_restore_buffers)`` referencing the padded tensors.
    """
    if seq_len % cp_divisor == 0:
        return cp_buffers, cp_no_restore_buffers
    pad_len = cp_divisor - (seq_len % cp_divisor)
    new_no_restore = set()
    for i, (buf, dim) in enumerate(zip(cp_buffers, cp_seq_dims)):
        pad_shape = list(buf.shape)
        pad_shape[dim] = pad_len
        if buf.dtype.is_floating_point:
            pad_val = torch.zeros(pad_shape, dtype=buf.dtype, device=buf.device)
        else:
            fill_val = _ROUND_ROBIN_PAD_FILL.get(batch_buffer_keys.get(i), 0)
            pad_val = torch.full(pad_shape, fill_val, dtype=buf.dtype, device=buf.device)
        old_buf = buf
        cp_buffers[i] = torch.cat([buf, pad_val], dim=dim)
        if old_buf in cp_no_restore_buffers:
            new_no_restore.add(cp_buffers[i])
    # Mirror every batch-sourced cp_buffer back into ``batch`` so any downstream
    # consumer reading from the dict sees the padded shape.
    for idx, key in batch_buffer_keys.items():
        batch[key] = cp_buffers[idx]
    return cp_buffers, new_no_restore


def _shard_batch_load_balanced(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
):
    """Shard a batch with torch ``context_parallel`` round-robin load balancing.

    ``CPShardStrategy.shard_batch`` implementation for the default framework-owned CP
    path (layout ``"round_robin"``, indices from
    :func:`_round_robin_local_indices`). Assumes an active CP mesh (size > 1).
    ``padding_token_id`` is accepted per the contract but unused: CP-pad slots
    are zero-filled, sit after every real token under the causal mask, and
    carry -100 labels.

    Returns:
        Prepared batch, token layout, and SDPA context for the forward.
    """
    # Remove attention_mask from the batch so the model does not attempt to
    # build a 4D causal mask (which would have mismatched shapes with
    # DTensor-sharded Q/K/V).  Each self_attn module's forward_pre_hook
    # (registered by attach_context_parallel_hooks) will set is_causal=True
    # so that SDPA handles causal masking internally.
    batch.pop("attention_mask", None)

    # Determine the primary sequence tensor: inputs_embeds (VLM with CP, where
    # multimodal token replacement happened pre-shard) or input_ids (standard LLM).
    primary_key, primary_seq_tensor, seq_len, batch_size, position_ids, pos_seq_dim = (
        _normalize_cp_primary_and_positions(batch)
    )

    labels = batch["labels"]

    # Collect all available tensors for context parallel.  We track each
    # cp_buffer's batch key (when sourced from ``batch``) so the padding pass
    # below can pick the semantically-correct fill sentinel and mirror the
    # padded tensor back into ``batch``.  ``loss_mask`` is passed as an arg
    # (not in batch) so it has no key.
    cp_buffers = [primary_seq_tensor, labels, position_ids]
    # inputs_embeds is [B, S, H] → seq_dim=1; input_ids is [B, S] → seq_dim=1
    cp_seq_dims = [1, 1, pos_seq_dim]
    cp_no_restore_buffers = {primary_seq_tensor, labels}
    batch_buffer_keys: dict[int, str] = {0: primary_key, 1: "labels", 2: "position_ids"}

    # Add loss_mask if available (passed as arg, not in batch -> no key)
    if loss_mask is not None:
        cp_buffers.append(loss_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(loss_mask)

    # Add padding_mask if available in batch
    if "padding_mask" in batch:
        padding_mask = batch["padding_mask"]
        batch_buffer_keys[len(cp_buffers)] = "padding_mask"
        cp_buffers.append(padding_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(padding_mask)

    cp_divisor = cp_mesh.size() * 2
    cp_buffers, cp_no_restore_buffers = _pad_cp_buffers_to_divisor(
        cp_buffers, cp_seq_dims, cp_no_restore_buffers, batch_buffer_keys, seq_len, cp_divisor, batch
    )

    # PyTorch's legacy context_parallel buffers API shards in place with
    # ``resize_``/``copy_``. ``resize_`` rejects tensors that require gradients,
    # and detaching inputs_embeds here would silently stop gradients to trainable
    # embeddings and multimodal towers. Apply the same default head-tail shard
    # out of place so autograd remains connected, then let context_parallel
    # mutate only the integer/mask buffers.
    primary_seq_tensor = cp_buffers[0]
    if primary_seq_tensor.requires_grad:
        local_indices = _round_robin_local_indices(
            cp_mesh,
            primary_seq_tensor.shape[cp_seq_dims[0]],
            device=primary_seq_tensor.device,
        )
        batch[primary_key] = _shard_token_tensor_by_indices(
            primary_seq_tensor,
            local_indices,
            seq_dim=cp_seq_dims[0],
        )
        cp_no_restore_buffers.remove(primary_seq_tensor)
        cp_buffers = cp_buffers[1:]
        cp_seq_dims = cp_seq_dims[1:]

    cp_ctx = _create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=cp_no_restore_buffers,
        cp_rotate_method="allgather",  # TODO: expose through cfg
    )
    return CPShardResult(
        _sdpa_context(cp_ctx),
        batch,
        original_seq_len=seq_len,
        padded_seq_len=seq_len + (-seq_len) % cp_divisor,
    )


def _shard_batch_aux_only(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
):
    """Round-robin CP shard of the no-grad AUXILIARY streams only.

    The framework-owned peer of :func:`_shard_batch_load_balanced` for models that
    embed and sequence-shard their primary stream (``input_ids`` /
    ``inputs_embeds`` and any ``pixel_*`` keys) inside their own forward
    (Megatron-style per-microbatch CP). Pads and round-robin-shards
    ``labels``/``position_ids``/``loss_mask``/``padding_mask`` exactly like the
    load-balanced path, installs the same
    ring-SDPA ``context_parallel`` context, and leaves the primary stream
    full-length in the batch for the model to embed and shard via
    :func:`shard_sequence_for_cp_round_robin`.

    Because the primary stream never enters the ``context_parallel`` buffer list,
    the grad-carrying-buffer constraint (``resize_`` rejects tensors requiring
    grad) does not apply: the model shards its embeddings with a differentiable
    ``index_select`` and gradients route to the embeddings / vision tower.

    Args:
        cp_mesh: The context-parallel device (sub)mesh (size > 1).
        tp_mesh: The tensor-parallel device (sub)mesh (or None); unused here.
        batch: The full-sequence batch; aux tensors are padded/mirrored in place
            and the primary stream (``input_ids``/``inputs_embeds``) is left
            full-length. ``labels`` is required.
        loss_mask: Optional per-token loss mask ``[batch, sequence]``, sharded
            alongside the batch.
        padding_token_id: Accepted per the contract but unused (the primary
            stream is not sharded here).
    Returns:
        Prepared batch, token layout, and SDPA context for the forward.
        The layout's ``padded_seq_len`` is what the model must pad its primary
        stream to before sharding.
    """
    batch.pop("attention_mask", None)

    _, _, seq_len, _, position_ids, pos_seq_dim = _normalize_cp_primary_and_positions(batch)

    labels = batch["labels"]

    # The primary stream is deliberately excluded: the model embeds and shards
    # it inside its forward. Only the no-grad aux streams enter the CP buffers.
    cp_buffers = [labels, position_ids]
    cp_seq_dims = [1, pos_seq_dim]
    cp_no_restore_buffers = {labels}
    batch_buffer_keys: dict[int, str] = {0: "labels", 1: "position_ids"}

    if loss_mask is not None:
        cp_buffers.append(loss_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(loss_mask)

    if "padding_mask" in batch:
        padding_mask = batch["padding_mask"]
        batch_buffer_keys[len(cp_buffers)] = "padding_mask"
        cp_buffers.append(padding_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(padding_mask)

    cp_divisor = cp_mesh.size() * 2
    cp_buffers, cp_no_restore_buffers = _pad_cp_buffers_to_divisor(
        cp_buffers, cp_seq_dims, cp_no_restore_buffers, batch_buffer_keys, seq_len, cp_divisor, batch
    )

    cp_ctx = _create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=cp_no_restore_buffers,
        cp_rotate_method="allgather",  # TODO: expose through cfg
    )
    return CPShardResult(
        _sdpa_context(cp_ctx),
        batch,
        original_seq_len=seq_len,
        padded_seq_len=seq_len + (-seq_len) % cp_divisor,
    )


def shard_sequence_for_cp_round_robin(
    cp_mesh, tensor: torch.Tensor, *, seq_dim: int = 1, pad_value: float | int = 0
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad and round-robin shard a full-length sequence tensor inside a model forward.

    The in-forward peer of :func:`_shard_batch_load_balanced`: a model that leaves
    its primary stream full-length (see :func:`_shard_batch_aux_only`) calls this
    on its embedded, spliced hidden states to keep this CP rank's head-tail chunk
    pair, matching the layout the ring-SDPA context and the aux streams were
    sharded to. Differentiable (``index_select``), so gradients route to the
    embeddings / vision tower.

    Args:
        cp_mesh: The context-parallel device (sub)mesh; None or size <= 1 makes
            this an identity (no pad, arange indices).
        tensor: Full-length sequence tensor, e.g. ``inputs_embeds`` of shape
            ``[batch, sequence, hidden]``; the axis given by ``seq_dim`` is padded
            to ``2 * cp_size`` and sharded.
        seq_dim: The sequence axis of ``tensor``.
        pad_value: Fill for the CP-padding slots appended on ``seq_dim`` (0 for
            embeddings, matching the zero-padding the load-balanced path applies
            to floating buffers).

    Returns:
        ``(local, local_indices, padded_seq_len)``: this rank's shard laid out
        head-chunk-then-tail-chunk (``[..., padded_sequence / cp_size, ...]`` on
        ``seq_dim``), its global token positions, and the padded global sequence
        length.
    """
    seq_len = tensor.shape[seq_dim]
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor, torch.arange(seq_len, device=tensor.device, dtype=torch.long), seq_len
    cp_divisor = 2 * cp_mesh.size()
    pad_len = (-seq_len) % cp_divisor
    if pad_len:
        tensor = _pad_tensor_seq_dim_(tensor, seq_dim, pad_len, pad_value)
    padded_seq_len = tensor.shape[seq_dim]
    local_indices = _round_robin_local_indices(cp_mesh, padded_seq_len, device=tensor.device)
    local = _shard_token_tensor_by_indices(tensor, local_indices, seq_dim=seq_dim)
    return local, local_indices, padded_seq_len


def shard_sequence_for_cp_contiguous(
    cp_mesh, tensor: torch.Tensor, *, seq_dim: int = 1, pad_value: float | int = 0, pad_multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad and contiguously shard a full-length sequence tensor inside a model forward.

    The contiguous peer of :func:`shard_sequence_for_cp_round_robin`: a model that keeps one
    contiguous ``seq_start:seq_end`` slice per CP rank (model-owned p2p ring, e.g.
    Gemma4) and leaves its primary stream full-length (see
    :func:`shard_batch_contiguous` with ``shard_primary=False``) calls this on
    its embedded, spliced hidden states (and its 4-D ``per_layer_inputs``) to keep this rank's
    contiguous shard. It pads the sequence axis to ``cp_size * max(pad_multiple, 2)``
    — Gemma4's ``pad_multiple`` divisor, NOT the round-robin ``2 * cp_size`` — so
    the slice aligns with the aux streams the contiguous aux-only sharder sliced.
    Differentiable (``index_select``), so gradients route to the embeddings /
    vision tower.

    Args:
        cp_mesh: The context-parallel (sub)mesh; None or size <= 1 is an identity.
        tensor: Full-length sequence tensor, e.g. ``inputs_embeds`` ``[B, S, H]``
            or ``per_layer_inputs`` ``[B, S, L, H]`` (seq axis 1).
        seq_dim: The sequence axis of ``tensor``.
        pad_value: Fill for the CP-padding slots appended on ``seq_dim``.
        pad_multiple: Per-CP-rank shard length multiple (the effective divisor is
            ``cp_size * max(pad_multiple, 2)``), matching
            :func:`shard_batch_contiguous`.

    Returns:
        ``(local, local_indices, padded_seq_len)``: this rank's contiguous shard
        (``[..., padded_sequence / cp_size, ...]`` on ``seq_dim``), its global
        token positions, and the padded global sequence length.
    """
    seq_len = tensor.shape[seq_dim]
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor, torch.arange(seq_len, device=tensor.device, dtype=torch.long), seq_len
    cp_divisor = cp_mesh.size() * max(int(pad_multiple or 1), 2)
    pad_len = (-seq_len) % cp_divisor
    if pad_len:
        tensor = _pad_tensor_seq_dim_(tensor, seq_dim, pad_len, pad_value)
    padded_seq_len = tensor.shape[seq_dim]
    local_indices = _contiguous_local_indices(cp_mesh, padded_seq_len, device=tensor.device)
    local = _shard_token_tensor_by_indices(tensor, local_indices, seq_dim=seq_dim)
    return local, local_indices, padded_seq_len
