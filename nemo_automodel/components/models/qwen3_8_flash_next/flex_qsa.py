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

"""FlexAttention execution of Qwen3.8-Flash-Next token-indexed sparse GQA.

One code path serves every training layout: dense right-padded batches,
packed (THD) rows, and context parallelism (local queries against gathered
global K/V) all reduce to "each query row attends exactly to its route IDs".
The routes are scattered into a boolean membership table, FlexAttention's
BlockMask skips fully-masked 128x128 tiles, and the flash-style kernel keeps
memory at O(S) instead of materializing scores. Rows whose routes are all
``-1`` (padding queries) produce exactly zero output and zero gradients.
"""

from __future__ import annotations

import functools

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@functools.cache
def _compiled_flex():
    """Compile lazily so CPU-only imports never trigger inductor."""
    return torch.compile(flex_attention, dynamic=False)


def _routes_to_membership(
    selected_token_ids: torch.Tensor,
    kv_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter route IDs into a boolean membership table.

    Args:
        selected_token_ids: Global route IDs ``[B, S_q, K]``; negative or
            out-of-range entries are padding.
        kv_length: Number of physical K/V rows.

    Returns:
        Boolean membership ``[B, S_q, kv_length]`` and a boolean
        ``[B, S_q]`` marking rows that selected at least one token.
    """
    batch_size, query_length, _ = selected_token_ids.shape
    valid = (selected_token_ids >= 0) & (selected_token_ids < kv_length)
    # scatter_ resolves duplicate indices nondeterministically, and every
    # padding slot clamps onto index zero, so a padding False could evict a
    # genuine True at token zero. scatter_add_ sums duplicates instead.
    hit_counts = torch.zeros(batch_size, query_length, kv_length, dtype=torch.int32, device=selected_token_ids.device)
    safe_ids = selected_token_ids.long().clamp(min=0, max=kv_length - 1)
    hit_counts.scatter_add_(-1, safe_ids, valid.to(torch.int32))
    return hit_counts > 0, valid.any(dim=-1)


def flex_sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run route-sparse GQA through FlexAttention.

    Args:
        query: BF16 CUDA queries ``[B, S_q, Hq, D]``.
        key: BF16 CUDA keys ``[B, S_kv, Hkv, D]``. ``S_kv`` may differ from
            ``S_q``; under context parallelism it is the gathered global
            length.
        value: BF16 CUDA values ``[B, S_kv, Hkv, D]``.
        selected_token_ids: int32/int64 route IDs ``[B, S_q, K]`` in global
            K/V coordinates; ``-1`` and out-of-range entries are padding.
        softmax_scale: Positive QK scale, defaulting to ``1 / sqrt(D)``.

    Returns:
        BF16 attention output ``[B, S_q, Hq, D]``. Padding-query rows are
        exactly zero, matching the PyTorch oracle contract.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("flex QSA expects [B, S, H, D] query/key/value")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")
    batch_size, query_length, num_query_heads, head_dim = query.shape
    kv_length, num_kv_heads = key.shape[1], key.shape[2]
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError(f"flex QSA requires Hq divisible by Hkv, got Hq={num_query_heads}, Hkv={num_kv_heads}")
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != (batch_size, query_length):
        raise ValueError(
            f"selected_token_ids must be [B, S_q, K] matching the queries; got {tuple(selected_token_ids.shape)}"
        )
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)

    membership, has_routes = _routes_to_membership(selected_token_ids, kv_length)

    def mask_mod(b, h, q_idx, kv_idx):
        # FlexAttention pads Q/KV to block multiples and probes the padded
        # coordinates; clamp the lookup and gate on in-range indices so the
        # padding region can never alias real membership entries.
        in_range = (q_idx < query_length) & (kv_idx < kv_length)
        safe_q = torch.clamp(q_idx, max=query_length - 1)
        safe_kv = torch.clamp(kv_idx, max=kv_length - 1)
        return in_range & membership[b, safe_q, safe_kv]

    block_mask = create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        device=str(query.device),
    )
    output = _compiled_flex()(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
    ).permute(0, 2, 1, 3)
    # Padding-query rows (no routes) must be exactly zero in both output and
    # gradient, matching the oracle and the padded-gap training contract.
    return output * has_routes[:, :, None, None].to(output.dtype)


__all__ = ["flex_sparse_gqa_attention"]
