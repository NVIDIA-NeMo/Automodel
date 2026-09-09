# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Packed multimodal attention mask helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.attention.flex_attention import and_masks


def _repeat_split_labels(
    split_lens: list[int], attn_modes: list[str], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token split labels used by the packed block-mask predicate."""
    if len(split_lens) != len(attn_modes):
        raise ValueError(
            f"split_lens and attn_modes must have the same length, got {len(split_lens)} and {len(attn_modes)}"
        )
    labels = torch.arange(len(split_lens), device=device, dtype=torch.long)
    lengths = torch.tensor(split_lens, device=device, dtype=torch.long)
    modes = list(attn_modes)

    full_or_noise = torch.tensor([mode in {"full", "noise"} for mode in modes], device=device, dtype=torch.bool)
    noise = torch.tensor([mode == "noise" for mode in modes], device=device, dtype=torch.bool)
    split_labels = torch.repeat_interleave(labels, lengths)
    full_or_noise_token_ids = torch.repeat_interleave(
        torch.where(full_or_noise, labels, -torch.ones_like(labels)), lengths
    )
    noise_token_ids = torch.repeat_interleave(torch.where(noise, labels, -torch.ones_like(labels)), lengths)
    if split_labels.numel() != sum(split_lens):
        raise RuntimeError("internal error while expanding packed attention split labels")
    return full_or_noise_token_ids, noise_token_ids


def _repeat_document_labels(document_lens: list[int], *, device: torch.device) -> torch.Tensor:
    """Return a per-token document id for packed samples."""
    if not document_lens:
        raise ValueError("document_lens must not be empty")
    labels = torch.arange(1, len(document_lens) + 1, device=device, dtype=torch.long)
    lengths = torch.tensor(document_lens, device=device, dtype=torch.long)
    return torch.repeat_interleave(labels, lengths)


def create_sparse_mask(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    device: torch.device,
) -> Any:
    """Create the block-mask predicate for packed multimodal attention."""
    full_or_noise_ids, noise_ids = _repeat_split_labels(split_lens, attn_modes, device=device)
    document_ids = _repeat_document_labels(document_lens, device=device)

    def causal_or_same_bidirectional_region(b, h, q_idx, kv_idx):
        same_region = full_or_noise_ids[q_idx] == full_or_noise_ids[kv_idx]
        region_is_bidirectional = full_or_noise_ids[q_idx] >= 0
        return (q_idx >= kv_idx) | (same_region & region_is_bidirectional)

    def does_not_cross_noise_region(b, h, q_idx, kv_idx):
        key_is_noise = noise_ids[kv_idx] >= 0
        query_and_key_share_noise_region = noise_ids[q_idx] == noise_ids[kv_idx]
        return (~key_is_noise) | query_and_key_share_noise_region

    def same_packed_sample(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    return and_masks(causal_or_same_bidirectional_region, does_not_cross_noise_region, same_packed_sample)
