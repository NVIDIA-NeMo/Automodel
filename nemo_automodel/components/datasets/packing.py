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

"""Dataset-owned packed-sequence construction contracts and helpers."""

from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypedDict

import torch
import torch.nn.functional as F

PackedMaskType = Literal["block_causal", "document_ids"]


class PackedSequenceContract(Protocol):
    """Structural model contract consumed while collating packed data."""

    @property
    def packed_mask_type(self) -> PackedMaskType:
        """Packed attention-mask representation required by the model."""
        ...

    @property
    def requires_packed_sequence_metadata(self) -> bool:
        """Whether the model consumes flat token indices and cumulative lengths."""
        ...


@dataclass(frozen=True)
class _DefaultPackedSequenceContract:
    """Block-causal packing defaults for callers without model requirements."""

    packed_mask_type: PackedMaskType = "block_causal"
    requires_packed_sequence_metadata: bool = False


DEFAULT_PACKED_SEQUENCE_CONTRACT: Final[PackedSequenceContract] = _DefaultPackedSequenceContract()


class PackedSequenceMetadata(TypedDict):
    """Batch-major metadata that remains valid after microbatch splitting."""

    packed_token_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


def get_seqlens_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract document lengths from an indexed packed-sequence mask.

    Args:
        attention_mask: Integer tensor of shape [batch, sequence]. Each nonzero
            value is a 1-based document index local to its batch row; zero marks
            padding.

    Returns:
        Tensor of shape [documents] containing nonzero document lengths in
        batch-major, document-index order.
    """
    batch_size = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_documents = int(torch.max(attention_mask).item())
    counts = torch.zeros((batch_size, max_documents), dtype=dtype, device=device)
    for document_idx in range(max_documents):
        counts[:, document_idx] = torch.sum(attention_mask == (document_idx + 1), dim=-1)

    counts = counts.flatten()
    return counts[counts.nonzero().squeeze(dim=-1)]


def get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build varlen metadata for an indexed or binary attention mask.

    Indexed masks treat every distinct positive document index in each batch
    row as a separate sequence. Binary masks treat every nonempty batch row as
    one sequence. Padding tokens are omitted from the flattened token stream.

    Args:
        attention_mask: Integer or boolean tensor of shape [batch, sequence].
            Positive values identify valid tokens and zero marks padding.

    Returns:
        A tuple containing ``indices`` of shape [tokens] into the flattened
        padded stream, ``cu_seqlens`` of shape [documents + 1] with dtype
        int32, and the largest document length as an integer.

    Raises:
        ValueError: If the mask is not rank two or contains no valid tokens.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {tuple(attention_mask.shape)}")
    if attention_mask.numel() == 0 or not bool(attention_mask.bool().any().item()):
        raise ValueError("attention_mask must contain at least one valid token")

    if attention_mask.dtype == torch.bool or int(attention_mask.max().item()) <= 1:
        seqlens_in_batch = attention_mask.bool().sum(dim=-1)
        seqlens_in_batch = seqlens_in_batch[seqlens_in_batch > 0]
    else:
        seqlens_in_batch = get_seqlens_in_batch(attention_mask)

    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def build_packed_sequence_metadata(attention_mask: torch.Tensor) -> PackedSequenceMetadata:
    """Build batch-major metadata for a padded indexed packing mask.

    Args:
        attention_mask: Integer tensor of shape [batch, sequence] containing
            1-based document IDs and zero-valued padding.

    Returns:
        Metadata containing row-local ``packed_token_indices`` of shape
        [batch, sequence] with ``-1`` at padding, per-row ``cu_seqlens`` of
        shape [batch, max_documents + 1] with ``-1`` at unused entries, and
        the largest document length. The leading batch axis lets pipeline
        schedules split these tensors without invalidating their offsets.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {tuple(attention_mask.shape)}")
    if attention_mask.numel() == 0 or not bool(attention_mask.bool().any().item()):
        raise ValueError("attention_mask must contain at least one valid token")

    batch_size, sequence_length = attention_mask.shape
    max_documents = max(1, int(attention_mask.max().item()))
    indices = torch.arange(sequence_length, device=attention_mask.device).expand(batch_size, -1).clone()
    indices.masked_fill_(~attention_mask.bool(), -1)
    cu_seqlens = torch.full(
        (batch_size, max_documents + 1),
        -1,
        dtype=torch.int32,
        device=attention_mask.device,
    )
    max_seqlen = 0
    for row_idx in range(batch_size):
        row_mask = attention_mask[row_idx : row_idx + 1]
        if not bool(row_mask.bool().any().item()):
            cu_seqlens[row_idx, 0] = 0
            continue
        _, row_cu_seqlens, row_max_seqlen = get_unpad_data(row_mask)
        cu_seqlens[row_idx, : row_cu_seqlens.numel()] = row_cu_seqlens
        max_seqlen = max(max_seqlen, row_max_seqlen)

    return {
        "packed_token_indices": indices,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
    }
