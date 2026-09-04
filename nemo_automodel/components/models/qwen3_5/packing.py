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

"""Shared packed-sequence metadata for dense and MoE Qwen3.5 models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nemo_automodel.components.models.common.packing import (
    flatten_packed_sequence_metadata,
    is_indexed_packed_mask,
)


@dataclass(frozen=True)
class GatedDeltaPackedMetadata:
    """Packed-sequence metadata shared by every GatedDeltaNet layer.

    Args:
        document_ids: Indexed document mask of shape [batch, sequence] on the
            compute device, with zero denoting padding.
        indices: Flattened valid-token indices of shape [tokens] on the compute
            device.
        cu_seqlens: Cumulative document lengths of shape [documents + 1] on the
            compute device.
        cu_seqlens_cpu: CPU mirror of ``cu_seqlens`` with shape [documents + 1]
            for FLA host-side chunk planning.
    """

    document_ids: torch.Tensor
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor


def prepare_gated_delta_packed_metadata(
    attention_mask: torch.Tensor | None,
    packed_seq_ids: torch.Tensor | None,
    packed_token_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> GatedDeltaPackedMetadata | None:
    """Prepare dataset-provided GatedDeltaNet metadata once per model forward.

    Args:
        attention_mask: Optional indexed document mask of shape [batch,
            sequence] or a backend-specific attention mask.
        packed_seq_ids: Optional indexed document IDs of shape [batch,
            sequence] supplied beside a backend-specific attention mask.
        packed_token_indices: Optional batch-major valid-token indices supplied
            by dataset packing.
        cu_seqlens: Optional batch-major cumulative document lengths supplied
            by dataset packing.

    Returns:
        Device and CPU packed-sequence metadata whose tensor layouts are
        documented by :class:`GatedDeltaPackedMetadata`, or ``None`` for an
        unpacked mask.
    """
    if is_indexed_packed_mask(attention_mask):
        document_ids = attention_mask
    elif is_indexed_packed_mask(packed_seq_ids):
        document_ids = packed_seq_ids
    else:
        return None

    if packed_token_indices is None or cu_seqlens is None:
        raise ValueError("Packed Qwen3.5 inputs require dataset-provided packed_token_indices and cu_seqlens.")
    packed_token_indices, cu_seqlens = flatten_packed_sequence_metadata(
        packed_token_indices,
        cu_seqlens,
        batch_size=document_ids.shape[0],
        sequence_length=document_ids.shape[1],
    )
    cu_seqlens = cu_seqlens.to(torch.long)
    return GatedDeltaPackedMetadata(
        document_ids=document_ids,
        indices=packed_token_indices,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.detach().cpu(),
    )
