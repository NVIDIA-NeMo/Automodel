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

"""Typed packed-sequence metadata for neat sequence packing.

NeMo AutoModel's neat packing encodes document boundaries in an indexed map
``[B, S]`` where each position holds the 1-based document index it belongs to
(``0`` = padding), e.g. ``[1, 1, 2, 2, 2, 0]``. This module turns that map into
typed metadata that drives HuggingFace's varlen FlashAttention path via
:class:`transformers.modeling_flash_attention_utils.FlashAttentionKwargs`
(``cu_seq_lens_q``/``cu_seq_lens_k``/``max_length_q``/``max_length_k``).

The varlen path reshapes ``[B, S, ...]`` to ``[B * S, ...]`` without unpadding
(see the ``is_fa_with_varlen_kwargs`` branch in
``transformers.modeling_flash_attention_utils._flash_attention_forward``).
``cu_seqlens`` therefore spans the whole flattened batch, including padding runs,
and a boundary is forced at every row start so a document that fills one row
cannot merge with the first document of the next row once the batch is flattened.
"""

from dataclasses import dataclass

import torch

__all__ = ["PackedSeqParams", "packed_seq_params_from_doc_ids", "to_flash_attention_kwargs"]


@dataclass
class PackedSeqParams:
    """Packed-sequence metadata derived from a neat-packing document map.

    Attributes:
        cu_seqlens: Int32 tensor of shape ``[num_segments + 1]``. Cumulative token
            counts over the flattened ``[batch * sequence]`` axis, including
            trailing padding runs, so ``cu_seqlens[-1] == batch * sequence``.
            Segment ``k`` spans flattened positions
            ``[cu_seqlens[k], cu_seqlens[k + 1])``.
        max_seqlen: Length of the longest segment, sized for the varlen
            flash-attention kernel.
        doc_ids: Int tensor of shape ``[batch, sequence]`` with the 1-based
            document index per token (``0`` = padding). Preserved unchanged for
            loss functions and context-parallel consumers.
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int
    doc_ids: torch.Tensor


def packed_seq_params_from_doc_ids(doc_ids: torch.Tensor) -> PackedSeqParams:
    """Build packed-sequence metadata from a neat-packing document map.

    Segments are maximal runs of an identical document index within a row, plus a
    forced boundary at every row start. Padding runs (index ``0``) become their
    own segments so ``cu_seqlens`` covers the full flattened ``[batch * sequence]``
    axis that the varlen flash-attention path consumes without unpadding. Padding
    tokens therefore attend only among themselves, which is harmless because their
    labels are ignored by the loss.

    Args:
        doc_ids: Int tensor of shape ``[batch, sequence]`` where each position
            holds the 1-based document index it belongs to (``0`` = padding).

    Returns:
        PackedSeqParams carrying ``cu_seqlens``, ``max_seqlen``, and the original
        ``doc_ids``.
    """
    if doc_ids.dim() != 2:
        raise ValueError(f"doc_ids must be 2D [batch, sequence], got shape {tuple(doc_ids.shape)}")

    batch, seq = doc_ids.shape
    total = batch * seq
    flat = doc_ids.reshape(-1)
    device = doc_ids.device

    # A segment starts at position 0, at every row boundary (so a row-filling
    # document cannot merge with the next row's first document once flattened),
    # and wherever the document index changes within a row.
    is_start = torch.zeros(total, dtype=torch.bool, device=device)
    is_start[0] = True
    is_start[1:] = flat[1:] != flat[:-1]
    is_start |= (torch.arange(total, device=device) % seq) == 0

    starts = torch.nonzero(is_start, as_tuple=False).flatten()
    # cu_seqlens is the segment start offsets with the flattened total appended;
    # starts[0] is already 0, so this begins at 0 and ends at batch * sequence.
    cu_seqlens = torch.cat([starts, starts.new_tensor([total])]).to(torch.int32)
    seglens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(seglens.max().item()) if seglens.numel() > 0 else 0

    return PackedSeqParams(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, doc_ids=doc_ids)


def to_flash_attention_kwargs(params: PackedSeqParams) -> dict[str, torch.Tensor | int]:
    """Convert packed-sequence params to HuggingFace ``FlashAttentionKwargs``.

    The returned mapping matches ``transformers`` ``FlashAttentionKwargs`` so a
    packed batch drives ``flash_attn_varlen_func`` directly. Query and key share
    one layout (self-attention), so the ``q`` and ``k`` entries are equal.

    Args:
        params: Packed-sequence metadata. ``params.cu_seqlens`` is an int32 tensor
            of shape ``[num_segments + 1]``.

    Returns:
        Mapping with ``cu_seq_lens_q``/``cu_seq_lens_k`` (the int32 ``cu_seqlens``
        tensor of shape ``[num_segments + 1]``) and ``max_length_q``/
        ``max_length_k`` (the ``max_seqlen`` int).
    """
    return {
        "cu_seq_lens_q": params.cu_seqlens,
        "cu_seq_lens_k": params.cu_seqlens,
        "max_length_q": params.max_seqlen,
        "max_length_k": params.max_seqlen,
    }
