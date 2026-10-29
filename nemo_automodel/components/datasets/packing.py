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

import warnings
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypedDict

import torch
import torch.nn.functional as F

BatchOutputFormat = Literal["batched", "thd"]

PackedMaskType = Literal["block_causal", "document_ids", "flash_varlen"]


class PackedSequenceContract(Protocol):
    """Model-derived requests passed to every loader, independently of format.

    Batched packed inputs interpret ``packed_mask_type`` and
    ``return_packed_sequence_metadata``. Native THD inputs derive their kernel
    metadata from lengths instead. ``return_document_ids`` requests optional
    original membership in either format; effective segmentation always remains
    available separately. Unpacked native loaders ignore packing-only requests.
    ``max_sequence_length`` is the model's input-token limit per packed document,
    before batch flattening, and applies to both batched and THD output.
    """

    @property
    def packed_mask_type(self) -> PackedMaskType:
        """Attention representation for batched packed inputs; ignored by THD."""
        ...

    @property
    def return_packed_sequence_metadata(self) -> bool:
        """Return the gathered-token bundle for batched packed output only.

        This requests ``packed_token_indices`` [batch, sequence] (-1 at padding),
        ``cu_seqlens`` [batch, boundaries] for the non-padding token stream (-1
        at unused entries), and ``max_seqlen``. AutoModel's FA4 and recurrent
        consumers use these to gather tokens and delimit kernel/state sequences.
        Their model-entry code flattens the bundle after PP microbatch splitting.
        ``build_packed_sequence_metadata`` constructs this layout from IDs; a
        loader's physical padded offsets cannot simply be renamed to match it.
        THD ignores this request and derives its own metadata from seq_lens
        and seq_lens_padded. The flag does not select an output format.
        """
        ...

    @property
    def return_document_ids(self) -> bool:
        """Return original membership as ``document_ids`` in either output format.

        Effective segmentation always uses ``_packed_seq_ids``. Original IDs
        are opt-in metadata for model policies such as independent image groups;
        they do not select attention or MTP loss boundaries.
        """
        ...

    @property
    def max_sequence_length(self) -> int | None:
        """Decoder input-token limit, or None when the model does not declare one."""
        ...


@dataclass(frozen=True)
class _DefaultPackedSequenceContract:
    """Block-causal packing defaults for callers without model requirements."""

    packed_mask_type: PackedMaskType = "block_causal"
    return_packed_sequence_metadata: bool = False
    return_document_ids: bool = False
    max_sequence_length: int | None = None


DEFAULT_PACKED_SEQUENCE_CONTRACT: Final[PackedSequenceContract] = _DefaultPackedSequenceContract()
_LEGACY_FLASH_ATTENTION_IMPLEMENTATIONS = frozenset({"flash_attention_2", "flash_attention_3", "flash_attention_4"})


def resolve_packing_contract(
    packing: PackedSequenceContract,
    attn_implementation: str | None,
) -> PackedSequenceContract:
    """Translate the deprecated attention keyword to a packing contract.

    Args:
        packing: Explicit structural packing contract. It takes precedence when
            both migration surfaces are supplied.
        attn_implementation: Deprecated attention-backend name, or ``None``.

    Returns:
        The explicit contract, or a compatibility contract matching the legacy
        mask-layout behavior.
    """
    if attn_implementation is None:
        return packing

    if packing is DEFAULT_PACKED_SEQUENCE_CONTRACT:
        packed_mask_type: PackedMaskType = (
            "flash_varlen" if attn_implementation in _LEGACY_FLASH_ATTENTION_IMPLEMENTATIONS else "block_causal"
        )
        packing = _DefaultPackedSequenceContract(packed_mask_type=packed_mask_type)
        message = (
            "attn_implementation is deprecated for packed-sequence construction; "
            "pass the model-derived packing contract instead"
        )
    else:
        message = "attn_implementation is deprecated and ignored because an explicit packing contract was supplied"
    warnings.warn(message, FutureWarning, stacklevel=3)
    return packing


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
