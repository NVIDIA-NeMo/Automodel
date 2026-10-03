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

"""Building blocks of HunyuanImage-3.0: attention, norms, timestep embedders and the UNet image projections.

Module and parameter names follow the released checkpoint so most weights load without renaming.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.hunyuan_image3.config import HunyuanImage3Config
from nemo_automodel.components.models.hunyuan_image3.rope import apply_rope


class HunyuanRMSNorm(nn.Module):
    """RMSNorm that normalizes in fp32 and returns ``weight * x`` in the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last axis.

        Args:
            x: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., hidden] in the promoted dtype of ``weight`` and ``x``.
        """
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class HunyuanImage3Attention(nn.Module):
    """GQA self-attention with a fused, KV-head-grouped QKV projection.

    ``qkv_proj`` outputs, for each KV head, its group of query heads followed by one key and one value head:
    ``[kv_heads, (q_per_kv + 2), head_dim]``. RoPE is applied before the per-head QK RMSNorm.
    """

    def __init__(self, config: HunyuanImage3Config, backend: BackendConfig, dtype: torch.dtype):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.q_per_kv = self.num_heads // self.num_kv_heads
        self.qkv_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, dtype=dtype
        )
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
            self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Attend over the joint sequence.

        Args:
            x: ``[batch, seq, hidden]`` input.
            cos: ``[batch, seq, head_dim]`` fp32 rotary table.
            sin: ``[batch, seq, head_dim]`` fp32 rotary table.
            attention_mask: Boolean ``[batch, 1, seq, seq]`` mask, true where attention is allowed; ``None`` means
                plain causal attention.

        Returns:
            ``[batch, seq, hidden]`` attention output.
        """
        batch, seq, _ = x.shape
        qkv = self.qkv_proj(x).view(batch, seq, self.num_kv_heads, self.q_per_kv + 2, self.head_dim)
        q, k, v = torch.split(qkv, [self.q_per_kv, 1, 1], dim=3)
        q = q.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # Query head h uses KV head h // q_per_kv (the fused layout groups query heads by KV head).
        k = k[:, :, None].expand(batch, self.num_kv_heads, self.q_per_kv, seq, self.head_dim).flatten(1, 2)
        v = v[:, :, None].expand(batch, self.num_kv_heads, self.q_per_kv, seq, self.head_dim).flatten(1, 2)
        if attention_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        return self.o_proj(out)

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        for linear in (self.qkv_proj, self.o_proj):
            nn.init.normal_(linear.weight, std=init_std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        if self.use_qk_norm:
            self.query_layernorm.reset_parameters()
            self.key_layernorm.reset_parameters()


class HunyuanSharedMLP(nn.Module):
    """SwiGLU MLP with the release's fused projection: ``gate_and_up_proj`` holds ``[up; gate]`` (up first).

    Keeping the released layout lets the shared expert load and save by renaming alone.
    """

    def __init__(self, dim: int, inter_dim: int, backend: BackendConfig, bias: bool, dtype: torch.dtype):
        super().__init__()
        self.gate_and_up_proj = initialize_linear_module(backend.linear, dim, 2 * inter_dim, bias=bias, dtype=dtype)
        self.down_proj = initialize_linear_module(backend.linear, inter_dim, dim, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared expert.

        Args:
            x: Tensor of shape [tokens, hidden].

        Returns:
            Tensor of shape [tokens, hidden].
        """
        up, gate = self.gate_and_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(up * F.silu(gate))

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        for linear in (self.gate_and_up_proj, self.down_proj):
            nn.init.normal_(linear.weight, std=init_std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal timestep features, cosine half first.

    Args:
        t: Tensor of shape [batch] holding (possibly fractional) timesteps.
        dim: Feature size.
        max_period: Longest sinusoid period.

    Returns:
        fp32 tensor of shape [batch, dim].
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep features followed by a two-layer GELU MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, dtype: torch.dtype | None = None):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Tensor of shape [batch] holding timesteps in [0, 1000].

        Returns:
            Tensor of shape [batch, hidden] in the MLP weight dtype.
        """
        freq = timestep_embedding(t, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        return self.mlp(freq)

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        for index in (0, 2):
            nn.init.normal_(self.mlp[index].weight, std=init_std)
            nn.init.zeros_(self.mlp[index].bias)


class GroupNorm32(nn.GroupNorm):
    """32-group GroupNorm computed in fp32 (the reference runs it under autocast, which upcasts it)."""

    def __init__(self, channels: int, dtype: torch.dtype | None = None):
        super().__init__(32, channels, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize channel groups.

        Args:
            x: Tensor of shape [batch, channels, height, width].

        Returns:
            Tensor of shape [batch, channels, height, width] in the dtype of ``x``.
        """
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.group_norm(x.float(), self.num_groups, weight, bias, self.eps).to(x.dtype)


class ResBlock(nn.Module):
    """Residual conv block with timestep-conditioned adaptive GroupNorm (no up/down sampling)."""

    def __init__(self, in_channels: int, emb_channels: int, out_channels: int, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_layers = nn.Sequential(
            GroupNorm32(in_channels, dtype=dtype),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dtype=dtype),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * out_channels, dtype=dtype))
        self.out_layers = nn.Sequential(
            GroupNorm32(out_channels, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dtype=dtype),
        )
        self.skip_connection = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, dtype=dtype)
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Args:
            x: Tensor of shape [batch, in_channels, height, width].
            emb: Tensor of shape [batch, emb_channels] timestep embedding.

        Returns:
            Tensor of shape [batch, out_channels, height, width].
        """
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_layers[0](h) * (1.0 + scale) + shift
        h = self.out_layers[1:](h)
        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    """Latent ``[B, C, H, W]`` -> token sequence ``[B, H*W, hidden]`` (patch size 1)."""

    def __init__(
        self, in_channels: int, emb_channels: int, hidden_channels: int, out_channels: int, dtype: torch.dtype | None
    ):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1, dtype=dtype),
                ResBlock(hidden_channels, emb_channels, out_channels, dtype=dtype),
            ]
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Embed latents as tokens.

        Args:
            x: Tensor of shape [batch, channels, height, width] VAE latents.
            emb: Tensor of shape [batch, hidden] timestep embedding.

        Returns:
            Tensor of shape [batch, height * width, hidden], tokens in row-major (height, width) order.
        """
        x = self.model[0](x)
        x = self.model[1](x, emb)
        return x.flatten(2).transpose(1, 2)


class UNetUp(nn.Module):
    """Token sequence ``[B, H*W, hidden]`` -> latent velocity ``[B, C, H, W]`` (patch size 1)."""

    def __init__(
        self, in_channels: int, emb_channels: int, hidden_channels: int, out_channels: int, dtype: torch.dtype | None
    ):
        super().__init__()
        self.model = nn.ModuleList(
            [
                ResBlock(in_channels, emb_channels, hidden_channels, dtype=dtype),
                nn.Sequential(
                    GroupNorm32(hidden_channels, dtype=dtype),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, out_channels, 3, padding=1, dtype=dtype),
                ),
            ]
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        """Project image tokens back to latent space.

        Args:
            x: Tensor of shape [batch, token_h * token_w, hidden], tokens in row-major (height, width) order.
            emb: Tensor of shape [batch, hidden] timestep embedding.
            token_h: Image height in tokens.
            token_w: Image width in tokens.

        Returns:
            Tensor of shape [batch, channels, token_h, token_w].
        """
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, token_h, token_w)
        x = self.model[0](x, emb)
        return self.model[1](x)


@torch.no_grad()
def init_unet_weights(module: nn.Module, init_std: float = 0.02) -> None:
    """Default init for the UNet projections: GroupNorm affine = identity, convs / linears normal."""
    for sub in module.modules():
        if isinstance(sub, nn.GroupNorm):
            nn.init.ones_(sub.weight)
            nn.init.zeros_(sub.bias)
        elif isinstance(sub, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(sub.weight, std=init_std)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
