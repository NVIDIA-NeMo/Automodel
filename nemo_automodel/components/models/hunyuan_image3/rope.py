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

"""2D rotary position embedding for HunyuanImage-3.0's joint text/image sequence.

Text tokens use position ``(n, n)``. An image block of ``h x w`` tokens that starts at sequence index ``L`` is laid
out on a grid centred on the span it occupies: rows start at ``L + (h*w - h) / 2`` and columns at
``L + (h*w - w) / 2``, and the (possibly fractional) grid coordinates are truncated to integers. The next text token
resumes at ``L + h*w``. Rotary frequencies alternate between the row and column axes and are applied in the
``rotate_half`` layout.
"""

from typing import Sequence

import torch


def image_block_positions(start: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row/column positions of an ``height x width`` image block starting at sequence index ``start``."""
    row0 = start + (width * height - height) / 2
    col0 = start + (width * height - width) / 2
    # Grid points row0 + i and col0 + j, computed in float32 and truncated like the released implementation.
    rows = row0 + torch.arange(height, dtype=torch.float32)
    cols = col0 + torch.arange(width, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    return grid_r.reshape(-1), grid_c.reshape(-1)


def build_2d_positions(seq_len: int, image_blocks: Sequence[tuple[int, int, int]]) -> torch.Tensor:
    """Return ``[seq_len, 2]`` integer (row, col) positions.

    Args:
        seq_len: Sequence length.
        image_blocks: ``(start, height, width)`` for every image block, in sequence order.
    """
    rows, cols = [], []
    cursor = 0
    for start, height, width in image_blocks:
        if cursor < start:
            text = torch.arange(cursor, start, dtype=torch.float32)
            rows.append(text)
            cols.append(text)
        r, c = image_block_positions(start, height, width)
        rows.append(r)
        cols.append(c)
        cursor = start + height * width
    tail = torch.arange(cursor, seq_len, dtype=torch.float32)
    rows.append(tail)
    cols.append(tail)
    pos = torch.stack([torch.cat(rows), torch.cat(cols)], dim=1)[:seq_len]
    return pos.long()


def build_2d_rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos``/``sin`` of shape ``[..., seq_len, head_dim]`` (float32) for ``positions`` ``[..., seq_len, 2]``."""
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    # Frequency pairs alternate between the row and column axes.
    inv_freq = inv_freq.reshape(head_dim // 4, 2)
    angles = positions.to(device=device, dtype=torch.float32).unsqueeze(-2) * inv_freq  # [..., S, D/4, 2]
    angles = angles.flatten(-2)  # [..., S, D/2]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` ``[B, H, S, D]`` with ``cos``/``sin`` ``[B, S, D]``; computes in ``cos.dtype``."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin
