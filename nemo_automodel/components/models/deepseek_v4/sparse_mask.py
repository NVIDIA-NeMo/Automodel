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

"""Linear-size visibility metadata for the DeepSeek-V4 TileLang path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from nemo_automodel.components.models.deepseek_v4.cp import dsv4_cp_all_gather_metadata


@dataclass(frozen=True)
class DSV4SparseMask:
    """Read-only visibility metadata; no query-by-key mask is retained.

    Attributes:
        query_positions: Int64 [B, Q] positions used by the base causal mask.
        key_positions: Int64 [B, K] key positions in the same coordinate system.
        query_seq_ids: Optional int64 [B, Q] document IDs; zero means padding.
        key_seq_ids: Optional int64 [B, K] document IDs paired with query_seq_ids.
        key_padding: Optional bool [B, K], true for masked keys.
        sliding_window: Optional positive causal window.
    """

    query_positions: torch.Tensor
    key_positions: torch.Tensor
    query_seq_ids: torch.Tensor | None
    key_seq_ids: torch.Tensor | None
    key_padding: torch.Tensor | None
    sliding_window: int | None

    def filter_indices(
        self,
        indices: torch.Tensor,
        *,
        vanilla_key_len: int,
        compress_ratio: int,
        query_positions: torch.Tensor,
        query_seq_ids: torch.Tensor | None,
        pooled_seq_ids: torch.Tensor | None,
        pooled_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Filter selected IDs using compact metadata, preserving order and duplicates.

        Args:
            indices: Int64 [B, Q, R] raw-plus-pooled physical key IDs; -1 is padding.
            vanilla_key_len: Number of raw keys preceding the compressed pool.
            compress_ratio: Compression ratio for pooled-key visibility.
            query_positions: Int64 [B, Q] document-local compressor positions.
            query_seq_ids: Optional int64 [B, Q] compressor document IDs.
            pooled_seq_ids: Optional int64 [B, P] compressed-key document IDs.
            pooled_positions: Optional int64 [B, P] document-local pool positions.

        Returns:
            Fresh int64 [B, Q, R] IDs, with invisible entries replaced by -1.
            Tensor inputs may be strided and are not mutated.
        """
        batch, queries, width = indices.shape
        raw_ids = indices.clamp(0, vanilla_key_len - 1).reshape(batch, -1)
        key_pos = self.key_positions.expand(batch, -1).gather(1, raw_ids).view(batch, queries, width)
        distance = self.query_positions.unsqueeze(-1) - key_pos
        raw_allowed = distance >= 0
        if self.sliding_window is not None:
            raw_allowed &= distance < self.sliding_window
        if self.query_seq_ids is not None:
            key_doc = self.key_seq_ids.expand(batch, -1).gather(1, raw_ids).view(batch, queries, width)
            raw_allowed &= (self.query_seq_ids.unsqueeze(-1) > 0) & (self.query_seq_ids.unsqueeze(-1) == key_doc)
        if self.key_padding is not None:
            raw_allowed &= ~self.key_padding.expand(batch, -1).gather(1, raw_ids).view(batch, queries, width)

        allowed = (indices >= vanilla_key_len) | raw_allowed
        if query_seq_ids is not None and pooled_seq_ids is not None and pooled_positions is not None:
            pool_ids = (indices - vanilla_key_len).clamp(0, pooled_seq_ids.shape[-1] - 1).reshape(batch, -1)
            pool_doc = pooled_seq_ids.expand(batch, -1).gather(1, pool_ids).view(batch, queries, width)
            pool_pos = pooled_positions.expand(batch, -1).gather(1, pool_ids).view(batch, queries, width)
            pool_allowed = (
                (query_seq_ids.unsqueeze(-1) > 0)
                & (query_seq_ids.unsqueeze(-1) == pool_doc)
                & (pool_pos < ((query_positions + 1) // compress_ratio).unsqueeze(-1))
            )
            allowed &= (indices < vanilla_key_len) | pool_allowed
        return indices.masked_fill(~allowed, -1)


def build_dsv4_sparse_mask(
    query_positions: torch.Tensor,
    *,
    key_len: int,
    packed_seq_ids: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    sliding_window: int | None,
    cp_group: dist.ProcessGroup | None,
) -> DSV4SparseMask:
    """Gather linear-size metadata once per model forward.

    Args:
        query_positions: Int64 [B, Q] mask positions (physical slots for local
            text; document-local positions for packed CP). Arbitrary strides.
        key_len: Number of global raw keys.
        packed_seq_ids: Optional int64 [B, Q] document IDs, zero for padding.
        padding_mask: Optional bool [B, Q], true for padding.
        sliding_window: Optional positive causal window.
        cp_group: CP process group, or None for local execution.

    Returns:
        Read-only metadata. Local query tensors may alias the supplied tensors;
        key tensors are gathered when CP is active. No quadratic allocation.
    """
    key_positions = (
        dsv4_cp_all_gather_metadata(query_positions, dim=1, cp_group=cp_group)
        if packed_seq_ids is not None
        else torch.arange(key_len, device=query_positions.device).unsqueeze(0)
    )
    return DSV4SparseMask(
        query_positions=query_positions,
        key_positions=key_positions,
        query_seq_ids=packed_seq_ids,
        key_seq_ids=dsv4_cp_all_gather_metadata(packed_seq_ids, dim=1, cp_group=cp_group),
        key_padding=dsv4_cp_all_gather_metadata(padding_mask, dim=1, cp_group=cp_group),
        sliding_window=sliding_window,
    )
