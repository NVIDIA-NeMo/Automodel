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

"""Stateless mapping from Zephon tensors to AutoModel's packed input contract."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch

from nemo_automodel.components.datasets.packing import PackedSequenceContract


class TrainingBatch(TypedDict):
    """Collated CPU tensors; AutoModel aliases are applied only when yielding.

    Token fields have shape [batch, sequence]. ``document_ids`` preserves
    subdocument membership (zero for padding); ``attention_document_ids`` also
    excludes processor-ineligible tokens and joins subdocuments when xdoc is on.
    Model positions are [batch, sequence] or [axes, batch, sequence] for mRoPE.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    padding_mask: torch.Tensor
    document_ids: NotRequired[torch.Tensor]
    attention_document_ids: NotRequired[torch.Tensor]
    attention_mask: NotRequired[torch.Tensor]
    _packed_seq_ids: NotRequired[torch.Tensor]
    packed_token_indices: NotRequired[torch.Tensor]
    cu_seqlens: NotRequired[torch.Tensor]
    max_seqlen: NotRequired[int]
    seq_lens: NotRequired[torch.Tensor]
    seq_lens_padded: NotRequired[torch.Tensor]
    qkv_format: NotRequired[str]


def require_supported_packing_contract(contract: PackedSequenceContract) -> None:
    """Validate model input interfaces once, during loader construction.

    The text and image config builders call this before sources/workers start;
    the contract is fixed for that loader and is not rechecked for each batch.

    ``return_packed_sequence_metadata`` specifically requests AutoModel's
    gathered-token bundle: packed_token_indices [batch, sequence] (-1 at padding),
    cu_seqlens [batch, boundaries] (-1 at unused entries), and max_seqlen. These
    describe the stream after gathering non-padding tokens. AutoModel's FA4 and
    recurrent/hybrid models use it for kernel layouts and recurrent-state resets.
    Zephon's cu_seqlens instead describes physical offsets including padding.
    The final batch map computes the gathered-token offsets with AutoModel's
    ``build_packed_sequence_metadata`` on the selected attention document IDs.
    Xdoc therefore joins both attention and recurrent-state boundaries; MTP
    uses those same selected boundaries. Keeping this bundle batch-major
    lets the model flatten it after PP microbatch splitting.
    """
    if contract.packed_mask_type not in ("block_causal", "flash_varlen", "document_ids"):
        raise ValueError(f"Unsupported packed attention representation: {contract.packed_mask_type}")


def adapt_model_inputs(
    batch: TrainingBatch,
    *,
    attention_eligibility: torch.Tensor | None = None,
) -> TrainingBatch:
    """Prepare attention membership from Zephon's collated document IDs.

    This follows to_training so IDs already match input tokens after shifting and
    padding. Zephon owns structural membership; processor eligibility and the
    consumer's attention policy belong to this integration. AutoModel field names
    are applied later, when the loader yields the completed batch.

    Args:
        batch: Owned mapping with int64 ``document_ids`` [batch, sequence] from
            Zephon. Token/label/padding tensors have the same shape; model positions
            may additionally have a leading RoPE-axes dimension. Mutates mapping
            entries only; never modifies an existing tensor in place.
        attention_eligibility: Optional bool [batch, sequence], supplied by the
            processor to exclude synthetic image tokens independently of padding.

    Returns:
        The same mapping with ``attention_document_ids`` [batch, sequence]. It
        aliases ``document_ids`` unless processor exclusions require a new tensor.
        Only linear metadata is produced; masks/Flash offsets are prepared at
        model entry after PP microbatch splitting.
    """
    document_ids = batch["document_ids"]
    batch["attention_document_ids"] = (
        document_ids if attention_eligibility is None else document_ids.masked_fill(~attention_eligibility, 0)
    )
    return batch


def enable_xdoc_attention(batch: TrainingBatch, *, packing_contract: PackedSequenceContract) -> TrainingBatch:
    """Join attention and position ranges inside each packed batch document.

    Args:
        batch: Mapped training tensors as documented by ``adapt_model_inputs``;
            ``document_ids`` and ``attention_document_ids`` have shape
            [batch, sequence]; zero in the latter marks excluded positions. Model
            ``position_ids`` has shape [batch, sequence] or
            [position_axes, batch, sequence] for multi-axis RoPE. Subdocument
            coordinates are nonnegative and local to each subdocument. The owned
            mapping is updated in place, but no existing tensor is mutated.
        packing_contract: Model-owned packed attention representation.

    Returns:
        The same mapping with translated position IDs and joined attention IDs
        [batch, sequence]. Original subdocument IDs are always retained separately;
        the model entry chooses an attention representation. Position IDs retain
        their original shape. Each eligible subdocument's position
        range follows the previous one's maximum on all axes. One common offset
        per subdocument preserves local scalar/multi-axis geometry; excluded
        coordinates are unchanged. Labels, original boundaries, padding and
        media retain their existing tensors. Batch documents remain independent and
        attention remains causal.
    """
    allowed = batch["attention_document_ids"] > 0
    attention_ids = allowed.to(torch.long)
    if packing_contract.packed_mask_type == "flash_varlen":
        excluded_after_content = ~allowed & allowed.cumsum(dim=-1).gt(0)
        if bool((allowed & excluded_after_content.cumsum(dim=-1).gt(0)).any()):
            raise ValueError("Flash-varlen xdoc does not support internal attention-excluded slots")
    positions = batch["position_ids"]
    document_ids = batch["document_ids"].masked_fill(~allowed, 0)
    if positions.ndim not in (2, 3) or positions.shape[-2:] != document_ids.shape:
        raise ValueError("Xdoc positions must have shape [batch, sequence] or [position_axes, batch, sequence]")
    if document_ids.numel():
        # AutoModel has no model-side xdoc position policy; translate all RoPE axes together.
        maxima = positions.amax(dim=0) if positions.ndim == 3 else positions
        spans = positions.new_full((document_ids.shape[0], int(document_ids.max()) + 1), -1)
        spans.scatter_reduce_(1, document_ids, maxima, reduce="amax")
        spans[:, 0] = -1  # Excluded/fake-image/padding slots do not extend a document.
        spans = spans + 1
        offsets = (spans.cumsum(dim=-1) - spans).gather(1, document_ids)
        batch["position_ids"] = positions + offsets
    batch["attention_document_ids"] = attention_ids
    return batch
