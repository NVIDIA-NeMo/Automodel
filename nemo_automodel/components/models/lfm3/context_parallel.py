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

"""Packed-document boundaries and context parallelism for LFM3.

Each CP rank owns one contiguous shard of the sequence. A convolution or attention layer that
needs tokens before the shard asks the earlier ranks for them: only the part of the document
that the shard starts in, and no more than the layer can reach (the kernel width minus one for a
convolution, the window for sliding attention, the whole document for full attention).
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import TypeAlias

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
    contiguous_local_indices,
    shard_batch_contiguous,
)
from nemo_automodel.components.models.common.packed_seq import packed_seq_params_from_doc_ids


@dataclass(frozen=True)
class DocumentHalo:
    """For every CP rank, the number of tokens before its shard that belong to the document the shard starts in."""

    group: dist.ProcessGroup
    lengths: tuple[int, ...]


@dataclass(frozen=True)
class PackedDocuments:
    """Document boundaries of the local tokens, rows flattened in order.

    ``cu_seq_lens`` is an int32 Tensor of shape [documents + 1] and ``max_length`` the longest
    document, as FlashAttention's varlen interface takes them. ``halo`` is set under context
    parallelism.
    """

    cu_seq_lens: torch.Tensor
    max_length: int
    halo: DocumentHalo | None = None

    def document_index(self, tokens: int) -> torch.Tensor:
        """Return the document of each token as a long Tensor of shape [tokens]."""
        positions = torch.arange(tokens, device=self.cu_seq_lens.device)
        return torch.bucketize(positions, self.cu_seq_lens[1:-1].long(), right=True)


# Recipe batch entries: token tensors, the packed-format tag, lengths and the CP halo.
CPBatch: TypeAlias = dict[str, torch.Tensor | DocumentHalo | str | int | None]


def _halo_sources(lengths: tuple[int, ...], shard_length: int, rank: int, limit: int | None) -> list[tuple[int, int]]:
    """Return the (rank, token count) pairs that hold a rank's halo, in sequence order."""
    needed = lengths[rank] if limit is None else min(lengths[rank], limit)
    sources = []
    while needed > 0:
        sources.append((rank - len(sources) - 1, min(needed, shard_length)))
        needed -= shard_length
    return sources[::-1]


@functools.cache
def _join_group(group: dist.ProcessGroup, device: torch.device) -> None:
    """Run one collective on the CP group, once per process.

    A batch of point-to-point operations may be the first operation on a process group only if
    every rank takes part in it, and a rank whose shard starts at a document boundary can have
    nothing to send or receive.
    """
    dist.all_reduce(torch.zeros(1, device=device), group=group)


def _transfer(
    group: dist.ProcessGroup, sends: list[tuple[int, torch.Tensor]], receives: list[tuple[int, torch.Tensor]]
) -> None:
    """Send and receive Tensors between CP ranks in one batch of point-to-point operations.

    Args:
        group: Context-parallel process group; the ranks below are ranks within it.
        sends: Pairs of destination rank and the Tensor to send, of shape
            [batch, channels, piece_tokens, ...]. Each piece is copied before it is sent.
        receives: Pairs of source rank and the buffer that receives its piece, of shape
            [batch, channels, piece_tokens, ...] with the sender's dtype, device and trailing
            dimensions. The buffers are filled in place.
    """
    operations = [
        dist.P2POp(dist.isend, piece.contiguous(), dist.get_global_rank(group, rank), group) for rank, piece in sends
    ]
    operations += [dist.P2POp(dist.irecv, piece, dist.get_global_rank(group, rank), group) for rank, piece in receives]
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()


class _HaloExchange(torch.autograd.Function):
    """Send shard suffixes to later ranks and receive this rank's halo; backward returns the gradients."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        receives: list[tuple[int, int]],
        sends: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Exchange shard suffixes.

        Args:
            ctx: Autograd context that records the exchange plan.
            tensor: This rank's shard, a Tensor of shape [batch, channels, sequence, ...];
                axis 2 is this rank's contiguous interval of the global sequence.
            group: Context-parallel process group.
            receives: Pairs of source rank and token count, in sequence order.
            sends: Pairs of destination rank and the number of trailing tokens it reads.

        Returns:
            The received pieces in sequence order, a Tensor of shape
            [batch, channels, halo_tokens, ...] where ``halo_tokens`` is the sum of the receive
            counts. It never aliases ``tensor``.
        """
        ctx.group, ctx.receives, ctx.sends, ctx.shape = group, receives, sends, tensor.shape
        pieces = [tensor.new_empty(*tensor.shape[:2], count, *tensor.shape[3:]) for _, count in receives]
        _transfer(
            group,
            [(rank, tensor[:, :, tensor.shape[2] - count :]) for rank, count in sends],
            [(rank, piece) for (rank, _), piece in zip(receives, pieces)],
        )
        return torch.cat(pieces, dim=2) if pieces else tensor.new_empty(*tensor.shape[:2], 0, *tensor.shape[3:])

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Return the halo's gradient to the ranks that own the tokens.

        Args:
            ctx: Autograd context holding the exchange plan of the forward.
            grad: Gradient of the halo, a Tensor of shape [batch, channels, halo_tokens, ...].

        Returns:
            Gradient of the local shard, a Tensor of shape [batch, channels, sequence, ...] that
            sums the gradients every later reader returns for the shard's trailing tokens, and
            ``None`` for the three non-tensor arguments.
        """
        pieces = grad.split([count for _, count in ctx.receives], dim=2)
        returned = [grad.new_empty(*ctx.shape[:2], count, *ctx.shape[3:]) for _, count in ctx.sends]
        _transfer(
            ctx.group,
            [(rank, piece) for (rank, _), piece in zip(ctx.receives, pieces)],
            [(rank, piece) for (rank, _), piece in zip(ctx.sends, returned)],
        )
        total = grad.new_zeros(ctx.shape)
        for piece in returned:
            total[:, :, ctx.shape[2] - piece.shape[2] :] += piece
        return total, None, None, None


def exchange_halo(tensor: torch.Tensor, halo: DocumentHalo, limit: int | None = None) -> torch.Tensor:
    """Fetch the tokens before this rank's shard that belong to its first document.

    Args:
        tensor: This rank's shard, a Tensor of shape [batch, channels, sequence, ...].
        halo: Halo length of every CP rank.
        limit: Largest number of tokens a layer can use; ``None`` fetches the whole halo.

    Returns:
        Tensor of shape [batch, channels, halo, ...] in sequence order. Gradients flow back
        to the ranks that own the tokens.
    """
    rank, shard_length = dist.get_rank(halo.group), tensor.shape[2]
    _join_group(halo.group, tensor.device)
    receives = _halo_sources(halo.lengths, shard_length, rank, limit)
    sends = [
        (later, count)
        for later in range(rank + 1, len(halo.lengths))
        for source, count in _halo_sources(halo.lengths, shard_length, later, limit)
        if source == rank
    ]
    return _HaloExchange.apply(tensor, halo.group, receives, sends)


def convolve_documents(conv: nn.Conv1d, gated: torch.Tensor, documents: PackedDocuments) -> torch.Tensor:
    """Run the causal convolution without reading across documents.

    Documents are laid out ``kernel_size - 1`` zeros apart, so each one gets the left padding
    it has when it runs alone.

    Args:
        conv: Depthwise convolution padded by ``kernel_size - 1`` on both ends.
        gated: Tensor of shape [batch, channels, sequence].
        documents: Boundaries of the flattened [batch * sequence] tokens.

    Returns:
        Tensor of shape [batch, channels, sequence].
    """
    batch, channels, sequence = gated.shape
    reach = conv.kernel_size[0] - 1
    stream = gated.transpose(0, 1).reshape(1, channels, batch * sequence)
    slots = torch.arange(batch * sequence, device=gated.device) + reach * documents.document_index(batch * sequence)
    if documents.halo is not None:
        # The halo continues the first document, so it sits right before it.
        halo = exchange_halo(stream, documents.halo, reach)
        slots = slots + halo.shape[2]
        stream = torch.cat((halo, stream), dim=2)
    halo_slots = torch.arange(stream.shape[2] - slots.numel(), device=gated.device)
    spaced = stream.new_zeros(1, channels, stream.shape[2] + reach * (documents.cu_seq_lens.numel() - 2))
    spaced.index_copy_(2, torch.cat((halo_slots, slots)), stream)
    mixed = conv(spaced.to(conv.weight.dtype)).index_select(2, slots).to(gated.dtype)
    return mixed.reshape(channels, batch, sequence).transpose(0, 1)


def document_attention_inputs(
    documents: PackedDocuments,
    key: torch.Tensor,
    value: torch.Tensor,
    sliding_window: int | None,
    attn_implementation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor | int]]:
    """Add this rank's halo to the keys and values and describe the documents to the attention backend.

    Args:
        documents: Boundaries of the flattened [batch * sequence] tokens.
        key: Tensor of shape [batch, heads, sequence, head_dim].
        value: Tensor of shape [batch, heads, sequence, head_dim].
        sliding_window: Number of keys a query sees, itself included; ``None`` for full attention.
        attn_implementation: Hugging Face attention backend of the layer.

    Returns:
        Keys and values of shape [batch, heads, halo + sequence, head_dim], the attention mask
        of shape [batch, 1, sequence, halo + sequence] (``None`` for FlashAttention), and the
        FlashAttention varlen arguments (empty for the other backends).
    """
    batch, heads, sequence, _ = key.shape
    halo_length = 0
    if documents.halo is not None:
        local = torch.cat((key, value), dim=1)
        halo = exchange_halo(local, documents.halo, None if sliding_window is None else sliding_window - 1)
        key, value = torch.cat((halo, local), dim=2).split(heads, dim=1)
        halo_length = halo.shape[2]
    if "flash_attention" in attn_implementation:
        varlen = {
            "cu_seq_lens_q": documents.cu_seq_lens,
            "cu_seq_lens_k": torch.cat((documents.cu_seq_lens[:1], documents.cu_seq_lens[1:] + halo_length)),
            "max_length_q": documents.max_length,
            "max_length_k": documents.max_length + halo_length,
        }
        return key, value, None, varlen
    document = documents.document_index(batch * sequence).view(batch, sequence)
    position = torch.arange(sequence, device=key.device)
    key_position = torch.arange(-halo_length, sequence, device=key.device)
    # The halo continues the first document.
    key_document = torch.nn.functional.pad(document, (halo_length, 0))
    allowed = (document[:, :, None] == key_document[:, None, :]) & (key_position <= position[:, None])
    if sliding_window is not None:
        allowed &= key_position > position[:, None] - sliding_window
    mask = allowed[:, None]
    if attn_implementation == "eager":
        mask = torch.zeros_like(mask, dtype=key.dtype).masked_fill(~mask, torch.finfo(key.dtype).min)
    return key, value, mask, {}


def _document_boundaries(batch: CPBatch) -> list[int]:
    """Return document boundaries over the flattened [batch * sequence] tokens and drop the entries they came from.

    Args:
        batch: Token Tensors of shape [batch, sequence]. The boundaries come from the first of:
            ``seq_lens_padded`` [batch, documents] of a THD batch (``qkv_format`` and ``seq_lens``
            are removed with it); ``cu_seq_lens_q`` [documents + 1] of a batch packed for
            FlashAttention; ``_packed_seq_ids`` [batch, sequence], 1-based document indices with
            0 at padding (removed); a 2D ``attention_mask`` [batch, sequence], nonzero at real
            tokens. Padding runs of the last two become segments of their own, so real tokens
            never attend to or convolve over padding. Any other batch holds one document per row.

    Returns:
        Boundaries as ints from 0 to batch * sequence; every row starts a new document.
    """
    rows, sequence = batch["input_ids"].shape
    if batch.pop("qkv_format", None) == "thd":
        batch.pop("seq_lens")
        boundaries = [0]
        for row, spans in enumerate(batch.pop("seq_lens_padded").tolist()):
            ends = list(accumulate(span for span in spans if span > 0))
            if not ends or ends[-1] != sequence:
                raise ValueError("LFM3 packed rows need `seq_lens_padded` that add up to the sequence length")
            boundaries += [row * sequence + end for end in ends]
        return boundaries
    if batch.get("cu_seq_lens_q") is not None:
        return batch["cu_seq_lens_q"].tolist()
    segments = batch.pop("_packed_seq_ids", None)
    if segments is None:
        attention_mask = batch.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            segments = attention_mask != 0
    if segments is None:
        return list(range(0, rows * sequence + 1, sequence))
    return packed_seq_params_from_doc_ids(segments.to(torch.long)).cu_seqlens.tolist()


def shard_documents(
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
    batch: CPBatch,
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
) -> tuple[Callable[[], contextlib.AbstractContextManager], CPBatch, ShardLayout]:
    """Keep this CP rank's contiguous shard and describe its documents to the model.

    Args:
        cp_mesh: CP mesh, or ``None``; every rank of it receives the same batch.
        tp_mesh: Unused TP mesh forwarded to the shared sharder.
        batch: Token Tensors of shape [batch, sequence]. A THD batch carries the document
            spans of each row in ``seq_lens_padded``, a batch packed for FlashAttention
            carries ``cu_seq_lens_q`` or the document indices ``_packed_seq_ids``, and an
            unpacked batch holds one document per row, split where a 2D ``attention_mask``
            marks padding (see :func:`_document_boundaries`).
        loss_mask: Optional Tensor of shape [batch, sequence] sharded with the batch.
        padding_token_id: Token used when the sequence is padded to a multiple of the CP size.

    Returns:
        A null context, the batch with Tensors of shape [batch, sequence / CP] plus
        ``cu_seq_lens_q``, ``max_length_q`` and, under CP, ``halo``, and the shard layout.
    """
    rows, sequence = batch["input_ids"].shape
    boundaries = _document_boundaries(batch)
    layout = ShardLayout(original_seq_len=sequence, padded_seq_len=sequence)
    if cp_mesh is not None and cp_mesh.size() > 1:
        if rows != 1:
            raise ValueError(f"LFM3 context parallelism takes one row per rank, got a local batch of {rows}")
        _, batch, layout = shard_batch_contiguous(
            cp_mesh, tp_mesh, batch, loss_mask=loss_mask, padding_token_id=padding_token_id
        )
        batch.pop("padding_mask", None)
        # CP padding extends the last document.
        boundaries[-1] = layout.padded_seq_len
        shard_length = layout.padded_seq_len // cp_mesh.size()
        starts = range(0, layout.padded_seq_len, shard_length)
        batch["halo"] = DocumentHalo(
            cp_mesh.get_group(),
            tuple(start - max(boundary for boundary in boundaries if boundary <= start) for start in starts),
        )
        start = starts[cp_mesh.get_local_rank()]
        inner = [boundary - start for boundary in boundaries if start < boundary < start + shard_length]
        boundaries = [0, *inner, shard_length]
    elif loss_mask is not None:
        batch["loss_mask"] = loss_mask
    batch["cu_seq_lens_q"] = torch.tensor(boundaries, dtype=torch.int32, device=batch["input_ids"].device)
    batch["max_length_q"] = max(end - start for start, end in pairwise(boundaries))
    return contextlib.nullcontext, batch, layout


def prepare_cp_inputs(batch: CPBatch, *, num_chunks: int = 1) -> dict[str, ContextParallelSharder]:
    """Return the sharder for packed THD batches at any CP size and for unpacked batches under CP.

    Args:
        batch: Unused here; the sharder reads the batch, with token Tensors of shape
            [batch, sequence], when it shards.
        num_chunks: Unused; LFM3 shards whole rows.

    Returns:
        Mapping with the ``cp_sharder`` whose ``shard_batch`` is :func:`shard_documents` and
        whose local token indices are those of the contiguous shard.
    """
    del batch, num_chunks
    return {
        "cp_sharder": ContextParallelSharder(
            shard_batch=shard_documents, local_token_global_indices=contiguous_local_indices
        )
    }
