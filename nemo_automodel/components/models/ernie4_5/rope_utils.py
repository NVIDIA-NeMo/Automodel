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

from __future__ import annotations

import torch
from torch import nn


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved RoPE pairs: [x0, x1] -> [-x1, x0]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ERNIE 4.5 interleaved rotary embeddings to q and k.

    Args:
        q: Query tensor in BSHD or THD format.
        k: Key tensor in BSHD or THD format.
        cos: Cosine tensor with shape [B, S, D] for BSHD or [T, D] for THD.
        sin: Sine tensor with shape [B, S, D] for BSHD or [T, D] for THD.
    """
    original_dtype = q.dtype
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q_embed = (q.float() * cos) + (rotate_every_two(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_every_two(k).float() * sin)
    return q_embed.to(original_dtype), k_embed.to(original_dtype)


class Ernie4_5RotaryEmbedding(nn.Module):
    """Rotary embedding module matching the Hugging Face ERNIE 4.5 implementation."""

    inv_freq: torch.Tensor

    def __init__(self, config, device: torch.device | None = None):
        super().__init__()
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 500000.0))
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        qkv_format: str = "bshd",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        angles = torch.einsum("bt,d->btd", position_ids.to(dtype=torch.float32), inv_freq)
        cos = angles.cos().repeat_interleave(2, dim=-1) * self.attention_scaling
        sin = angles.sin().repeat_interleave(2, dim=-1) * self.attention_scaling

        if qkv_format == "thd":
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
        return cos, sin
