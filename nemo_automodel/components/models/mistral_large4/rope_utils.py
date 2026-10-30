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

"""Mistral Large 4 differentiable rotary and SSMax floating-point numerics."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config


class MistralLarge4Rotary(nn.Module):
    """Shared learnable LeRoPE frequencies; build angles in FP32 after dtype casts."""

    def __init__(self, config: MistralLarge4Config) -> None:
        super().__init__()
        self.config = config
        self.alpha = nn.Parameter(torch.zeros(config.qk_rope_head_dim // 2, dtype=torch.float32))

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build differentiable interleaved YaRN tables.

        Args:
            positions: Integer tensor of shape [tokens] containing document-relative positions.

        Returns:
            FP32 cosine and sine tensors, each of shape [tokens, rotary_dim / 2].
            No low-precision frequency buffer is retained between calls.
        """
        c = self.config
        dim = c.qk_rope_head_dim
        pos_freq = c.rope_theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim)
        low = math.floor(
            dim
            * math.log(c.yarn_original_max_position_embeddings / (c.yarn_beta_fast * 2 * math.pi))
            / (2 * math.log(c.rope_theta))
        )
        high = math.ceil(
            dim
            * math.log(c.yarn_original_max_position_embeddings / (c.yarn_beta_slow * 2 * math.pi))
            / (2 * math.log(c.rope_theta))
        )
        low, high = max(low, 0), min(high, dim - 1)
        ramp = (
            (torch.arange(dim // 2, device=positions.device, dtype=torch.float32) - low)
            / (high - low if high != low else 0.001)
        ).clamp(0, 1)
        scale = self.alpha.float().exp()
        extrapolated = scale / pos_freq
        interpolated = scale / (c.yarn_factor * pos_freq)
        freq = interpolated * ramp + extrapolated * (1 - ramp)
        angles = positions.float()[:, None] * freq[None, :]
        return angles.cos(), angles.sin()


def apply_interleaved_rope(x: torch.Tensor, tables: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate adjacent channel pairs in FP32, then restore the activation dtype.

    Args:
        x: Tensor of shape [tokens, heads, rotary_dim] or [tokens, rotary_dim].
        tables: FP32 cosine and sine tensors of shape [tokens, rotary_dim / 2].

    Returns:
        Independent tensor with the same shape and dtype as x.
    """
    cos, sin = tables
    if x.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2).to(x.dtype)


class SSMax(nn.Module):
    """Per-head s * log(position + 1) + b in the query's compute dtype."""

    def __init__(self, heads: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.zeros(heads, dtype=dtype))
        self.b = nn.Parameter(torch.ones(heads, dtype=dtype))

    def forward(self, q: torch.Tensor, log_positions: torch.Tensor) -> torch.Tensor:
        """Scale a query without modifying storage shared with another consumer.

        Args:
            q: Tensor of shape [tokens, heads, head_dim].
            log_positions: FP32 tensor of shape [tokens].

        Returns:
            Independent tensor of shape [tokens, heads, head_dim] in q.dtype.
        """
        scale = log_positions.to(q.dtype)[:, None] * self.s.to(q.dtype)[None, :] + self.b.to(q.dtype)[None, :]
        return q * scale[..., None]


class FP32LayerNorm(nn.LayerNorm):
    """Normalize indexer keys with an FP32 parameter-owning module call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32 and restore the input activation dtype.

        Args:
            x: Tensor of shape [..., index_dim], with arbitrary leading dimensions.

        Returns:
            Independent tensor with the input shape/dtype. Parameters remain FP32.
        """
        return F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(x.dtype)
