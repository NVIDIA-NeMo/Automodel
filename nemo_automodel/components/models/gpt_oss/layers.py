# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# See the License for the specific governing permissions and
# limitations under the License.

import functools
import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.moe.utils import (
    BackendConfig,
    initialize_attn_module_and_func,
    initialize_linear_module,
)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    @functools.cache
    @torch.no_grad()
    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) / self.head_dim)
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[1]
        cos, sin = self._compute_cos_sin(num_tokens)

        query = _apply_rotary_emb(query, cos, sin)

        key = _apply_rotary_emb(key, cos, sin)
        return query, key


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, backend: BackendConfig, use_sliding_attention: bool = False):
        super().__init__()

        self.sliding_window = config.sliding_window if use_sliding_attention else None
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_attention_heads * self.head_dim, self.hidden_size, bias=True
        )
        self.sinks = nn.Parameter(torch.empty(self.num_attention_heads))

        self.softmax_scale = self.head_dim**-0.5

        # RoPE is computed and cached at the model level; attention consumes provided cos|sin

        assert backend.attn == "flex", "Only Flex Attention is supported for GPT-OSS"
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=config.num_attention_heads,
            num_qk_channels=config.head_dim,
            num_v_channels=config.head_dim,
            softmax_scale=self.softmax_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.q_proj(x).view(hidden_shape)
        k = self.k_proj(x).view(hidden_shape)
        v = self.v_proj(x).view(hidden_shape)

        # freqs_cis is concatenated [cos, sin] along last dim with shape (B, T, head_dim)
        cos, sin = freqs_cis.split(self.head_dim // 2, dim=-1)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        output = self.attn_module(
            q,
            k,
            v,
            scale=self.softmax_scale,
            sink_weights=(self.sinks.to_local() if isinstance(self.sinks, DTensor) else self.sinks),
            sliding_window=(self.sliding_window if self.sliding_window is not None else 0),
            enable_gqa=True,
        )
        output = output.transpose(1, 2).contiguous()  # (B, H, T, D) -> (B, T, H, D)

        # Reshape and project output
        output = output.view(bsz, seqlen, -1)
        # (bsz, seqlen, n_heads * v_head_dim)
        output = self.o_proj(output)  # (bsz, seqlen, dim)
        return output

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        with buffer_device:
            linear_list = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.o_proj,
            ]

            nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)
            for linear in linear_list:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
