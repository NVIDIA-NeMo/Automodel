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

"""2D rotary position embedding of HunyuanImage-3.0.

Every token gets a (y, x) position. Text tokens use their sequence index for both coordinates, so a text-only
sequence reduces to ordinary 1D RoPE. An image of ``h x w`` tokens starting at sequence index ``L`` is placed on a
grid centered inside the span it occupies: ``y = L + (h*w - h) / 2 + row`` and ``x = L + (h*w - w) / 2 + col``.
Tokens after the image continue from ``L + h*w``.

The rotary frequencies alternate between the two axes: of the ``head_dim / 2`` frequencies, even ones rotate with
``y`` and odd ones with ``x``. The angles are laid out for the half-split ``rotate_half`` convention.
"""

from __future__ import annotations

import torch


def image_grid_positions(
    seq_len: int, image_start: int, token_h: int, token_w: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return ``[seq_len, 2]`` integer (y, x) positions for a sequence holding one image.

    Args:
        seq_len: Total sequence length.
        image_start: Index of the first image token.
        token_h: Image height in tokens.
        token_w: Image width in tokens.
        device: Device of the returned tensor.

    Returns:
        Long tensor of shape ``[seq_len, 2]`` with the (y, x) position of every token.
    """
    num_image = token_h * token_w
    if image_start < 0 or image_start + num_image > seq_len:
        raise ValueError(f"Image span [{image_start}, {image_start + num_image}) does not fit in {seq_len} tokens.")
    before = torch.arange(image_start, device=device, dtype=torch.float32)
    beta_y = image_start + (num_image - token_h) / 2
    beta_x = image_start + (num_image - token_w) / 2
    rows = torch.arange(token_h, device=device, dtype=torch.float32)
    cols = torch.arange(token_w, device=device, dtype=torch.float32)
    grid_y = (beta_y + rows)[:, None].expand(token_h, token_w).reshape(-1)
    grid_x = (beta_x + cols)[None, :].expand(token_h, token_w).reshape(-1)
    after = torch.arange(image_start + num_image, seq_len, device=device, dtype=torch.float32)
    y = torch.cat([before, grid_y, after])
    x = torch.cat([before, grid_x, after])
    # Truncate toward zero like the reference (half-integer grid offsets occur for odd spans).
    return torch.stack([y, x], dim=-1).long()


def text_positions(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return ``[seq_len, 2]`` positions of a text-only sequence (y = x = index)."""
    index = torch.arange(seq_len, device=device)
    return torch.stack([index, index], dim=-1)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fp32 rotary tables from (y, x) positions.

    Args:
        positions: Integer tensor ``[..., seq, 2]`` of (y, x) positions.
        head_dim: Attention head dimension; must be divisible by 4.
        base: RoPE base frequency.

    Returns:
        ``(cos, sin)``, each fp32 of shape ``[..., seq, head_dim]``.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    # [head_dim/4, 2]: column 0 pairs with y, column 1 with x.
    inv_freq = inv_freq.reshape(head_dim // 4, 2)
    angles = positions.float().unsqueeze(-2) * inv_freq  # [..., seq, head_dim/4, 2]
    angles = angles.flatten(-2)  # [..., seq, head_dim/2]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``[batch, heads, seq, head_dim]`` with ``[batch, seq, head_dim]`` tables.

    The result is fp32 (the tables are fp32), matching the reference, which rotates and then normalizes q / k in
    fp32 before casting back.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin
