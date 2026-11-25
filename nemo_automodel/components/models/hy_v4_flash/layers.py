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

"""Norms, hyper-connections and interleaved multimodal RoPE of Hy4-flash.

Semantics follow the transformers reference (PR 50903, ``modeling_hy_v4_flash.py``):

* ``HY4FlashRMSNorm`` casts the normalised activations back to the input dtype *before* the weight multiply.
* The hyper-connection (iHC) has no Sinkhorn mixing: a sigmoid ``pre`` weight collapses the ``hc_mult`` streams
  and a sigmoid ``post`` weight scatters the sub-layer output back, the old streams passing through unchanged.
* RoPE rotates interleaved pairs of the trailing ``qk_rope_head_dim`` channels, with the time/height/width
  planes of M-RoPE interleaved in the frequency axis (Qwen3-VL style).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashTextConfig


class HY4FlashRMSNorm(nn.Module):
    """RMSNorm that multiplies the weight after casting back to the activation dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalise the last axis.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of the same shape and dtype as ``hidden_states``.
        """
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def unweighted_rms_inverse(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Return ``rsqrt(mean(x^2) + eps)`` over the last axis in the dtype of ``x``.

    Args:
        x: Tensor of shape [..., features], with arbitrary leading dimensions.

    Returns:
        Tensor of shape [..., 1].
    """
    return torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


class HY4FlashHyperConnection(nn.Module):
    """iHC site: collapse ``hc_mult`` residual streams to one input and produce the scatter weights.

    All parameters are fp32 and the arithmetic runs in fp32 whatever the activation dtype.
    """

    def __init__(self, config: HY4FlashTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_magnitude = config.hc_magnitude
        self.norm_eps = config.rms_norm_eps
        self.fn = nn.Parameter(torch.empty(2 * self.hc_mult, self.hc_mult * config.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(2 * self.hc_mult, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(2, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, initializer_range: float) -> None:
        """Initialise as the reference does: small ``fn``, ``scale`` 0.01 and a ``-log(hc_mult - 1)`` pre bias."""
        nn.init.normal_(self.fn, mean=0.0, std=initializer_range)
        nn.init.constant_(self.scale, 0.01)
        base = torch.zeros_like(self.base)
        base[: self.hc_mult] = -math.log(max(self.hc_mult - 1, 1))
        self.base.copy_(base)

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the collapsed sub-layer input and the post weights.

        Args:
            streams: Tensor of shape [batch, sequence, hc_mult, hidden].

        Returns:
            ``post``: fp32 tensor of shape [batch, sequence, hc_mult]; ``collapsed``: tensor of shape
            [batch, sequence, hidden] in the dtype of ``streams``.
        """
        flat = streams.flatten(2).float()
        mixes = F.linear(flat, self.fn.float()) * unweighted_rms_inverse(flat, self.norm_eps).float()
        pre_b, post_b = self.base.split(self.hc_mult)
        pre_scale, post_scale = self.scale.unbind(0)
        pre_logits, post_logits = mixes.split(self.hc_mult, dim=-1)
        pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + self.hc_eps
        post = self.hc_magnitude * torch.sigmoid(post_logits * post_scale + post_b) + self.hc_eps
        collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=2)
        return post, collapsed.to(streams.dtype)

    @staticmethod
    def expand(post: torch.Tensor, output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Scatter a sub-layer output onto the streams: ``post * output + residual``.

        Args:
            post: fp32 tensor of shape [batch, sequence, hc_mult].
            output: Tensor of shape [batch, sequence, hidden].
            residual: Tensor of shape [batch, sequence, hc_mult, hidden]; its dtype is the result dtype.

        Returns:
            Tensor of shape [batch, sequence, hc_mult, hidden] in the dtype of ``residual``.
        """
        return (post.unsqueeze(-1) * output.float().unsqueeze(-2) + residual.float()).to(residual.dtype)


class HY4FlashHyperHead(nn.Module):
    """Final collapse of the streams before the output norm."""

    def __init__(self, config: HY4FlashTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, initializer_range: float) -> None:
        """Initialise as the reference does."""
        nn.init.normal_(self.hc_fn, mean=0.0, std=initializer_range)
        nn.init.constant_(self.hc_scale, 0.01)
        nn.init.constant_(self.hc_base, -math.log(max(self.hc_mult - 1, 1)))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """Collapse the streams.

        Args:
            streams: Tensor of shape [batch, sequence, hc_mult, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden] in the dtype of ``streams``.
        """
        flat = streams.flatten(2).float()
        mixes = F.linear(flat, self.hc_fn.float()) * unweighted_rms_inverse(flat, self.norm_eps).float()
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * streams.float()).sum(dim=2).to(streams.dtype)


def inverse_frequencies(rope_dim: int, theta: float) -> torch.Tensor:
    """fp32 inverse frequencies ``theta ** (-2i / rope_dim)`` of shape [rope_dim // 2], computed from the config."""
    return 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))


def mrope_cos_sin(
    position_ids: torch.Tensor, inv_freq: torch.Tensor, mrope_section: list[int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved M-RoPE tables.

    The frequency axis takes the time plane everywhere except every third slot (starting at 1 for height and
    at 2 for width) within ``3 * section`` entries, which take the height / width planes.

    Args:
        position_ids: Integer tensor of shape [3, batch, sequence] (time, height, width positions).
        inv_freq: fp32 tensor of shape [rope_dim // 2].
        mrope_section: Three entries summing to ``rope_dim // 2``.
        dtype: Output dtype.

    Returns:
        ``(cos, sin)``, each of shape [batch, sequence, rope_dim // 2] (one value per rotated pair).
    """
    freqs = position_ids[..., None].float() * inv_freq.to(position_ids.device)
    merged = freqs[0].clone()
    for plane, offset in ((1, 1), (2, 2)):
        merged[..., slice(offset, mrope_section[plane] * 3, 3)] = freqs[plane][
            ..., slice(offset, mrope_section[plane] * 3, 3)
        ]
    return merged.cos().to(dtype), merged.sin().to(dtype)


def apply_interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved channel pairs ``(x[2i], x[2i + 1])`` of the last axis in fp32.

    Args:
        x: Tensor of shape [batch, sequence, heads, rope_dim].
        cos: Tensor of shape [batch, sequence, rope_dim // 2].
        sin: Tensor of shape [batch, sequence, rope_dim // 2].

    Returns:
        Tensor of the shape and dtype of ``x``.
    """
    pairs = x.float().unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    c, s = cos.float().unsqueeze(2), sin.float().unsqueeze(2)
    out = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1)
    return out.flatten(-2).to(x.dtype)
