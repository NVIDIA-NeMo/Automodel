# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""Interleaved default RoPE used by the public Tencent HY4-preview model.

The implementation follows the pinned vLLM HY4 forward, which constructs
``get_rope(..., is_neox_style=False)`` with the checkpoint's default RoPE
parameters. Unsupported scaling and Transformer Engine fused variants are
rejected by the HY4 config/model instead of being inferred here.
"""

from __future__ import annotations

import torch

__all__ = [
    "apply_rotary_emb",
    "freqs_cis_from_position_ids",
    "mla_softmax_scale",
    "precompute_freqs_cis",
]


def precompute_freqs_cis(qk_rope_head_dim: int, rope_theta: float) -> torch.Tensor:
    """Precompute the inverse frequencies for adjacent HY4 rotary pairs.

    Args:
        qk_rope_head_dim: Rotary width ``D_rope``; adjacent scalar pairs form
            the complex dimensions used by interleaved (GPT-J-style) RoPE.
        rope_theta: Base frequency from ``config.rope_parameters``.

    Returns:
        FP32 inverse frequencies with shape ``[D_rope / 2]``.

    Raises:
        ValueError: If ``qk_rope_head_dim`` is not positive and even, or if
            ``rope_theta`` is not positive.
    """
    if qk_rope_head_dim <= 0 or qk_rope_head_dim % 2:
        raise ValueError("HY4 qk_rope_head_dim must be a positive even integer")
    if rope_theta <= 0:
        raise ValueError("HY4 rope_theta must be positive")
    rotary_indices = torch.arange(0, qk_rope_head_dim, 2, dtype=torch.float32)
    return 1.0 / (float(rope_theta) ** (rotary_indices / qk_rope_head_dim))


@torch.no_grad()
def freqs_cis_from_position_ids(position_ids: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Build complex rotations for packed token positions.

    Args:
        position_ids: Integer token positions with shape ``[tokens]`` or
            ``[batch, tokens]``.
        freqs: FP32 inverse frequencies with shape ``[D_rope / 2]``.

    Returns:
        Complex64 rotations with shape ``[*position_ids.shape, D_rope / 2]``.
        No returned tensor aliases either input.
    """
    angles = torch.einsum(
        "...t,d->...td",
        position_ids.to(device=freqs.device, dtype=torch.float32),
        freqs,
    )
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, qkv_format: str = "thd") -> torch.Tensor:
    """Apply vLLM's interleaved HY4 RoPE to query or key states.

    Args:
        x: Query/key tensor. THD layout is ``[tokens, heads, D_rope]``;
            BSHD layout is ``[batch, tokens, heads, D_rope]``.
        freqs_cis: Complex rotations shaped ``[tokens, D_rope / 2]`` for THD
            or ``[batch, tokens, D_rope / 2]`` for BSHD.
        qkv_format: Either ``"thd"`` or ``"bshd"``.

    Returns:
        Rotated tensor with the same shape and dtype as ``x``. The output does
        not alias ``x``.

    Raises:
        ValueError: If the layout name, tensor rank, or rotary shapes differ
            from the public HY4 forward contract.
    """
    expected_rank = 3 if qkv_format == "thd" else 4 if qkv_format == "bshd" else None
    if expected_rank is None:
        raise ValueError(f"HY4 RoPE supports qkv_format='thd' or 'bshd', got {qkv_format!r}")
    if x.ndim != expected_rank:
        raise ValueError(f"HY4 {qkv_format} RoPE expected rank {expected_rank}, got shape {tuple(x.shape)}")
    if x.shape[-1] % 2:
        raise ValueError("HY4 interleaved RoPE requires an even rotary width")

    expected_freq_shape = (*x.shape[:-2], x.shape[-1] // 2)
    if tuple(freqs_cis.shape) != expected_freq_shape:
        raise ValueError(f"HY4 RoPE expected frequencies shaped {expected_freq_shape}, got {tuple(freqs_cis.shape)}")

    input_dtype = x.dtype
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    rotated = complex_x * freqs_cis.unsqueeze(-2)
    return torch.view_as_real(rotated).flatten(-2).to(input_dtype)


def mla_softmax_scale(qk_head_dim: int) -> float:
    """Return the exact scale used by the pinned vLLM HY4 MLA forward."""
    return float(qk_head_dim**-0.5)
