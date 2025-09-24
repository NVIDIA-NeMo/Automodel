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
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.models.gpt_oss.layers import _apply_rotary_emb
from nemo_automodel.components.moe.utils import (
    BackendConfig,
    initialize_attn_module_and_func,
    initialize_linear_module,
    initialize_rms_norm_module,
)


def _preprocess_for_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor | None, backend: BackendConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Preprocess attention inputs based on backend requirements.

    Mirrors deepseek_v3.layers.preprocess_args_and_kwargs_for_attn but inlined to avoid import cycles.
    """
    if backend.attn == "te":
        if attention_mask is None:
            attn_kwargs = {}
        else:
            padding_mask = attention_mask.logical_not()
            attn_kwargs = {
                "attn_mask_type": "padding_causal",
                "window_size": (-1, 0),
                "attention_mask": padding_mask.unsqueeze(1).unsqueeze(2),
            }
    else:  # sdpa / flex
        if attention_mask is None:
            attn_kwargs = {}
        else:
            attn_kwargs = {
                "attention_mask": attention_mask.bool(),
            }
        # SDPA expects (B, H, S, D)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        attn_kwargs["is_causal"] = True

    return q, k, v, attn_kwargs


def _postprocess_from_attn(x: torch.Tensor, backend: BackendConfig) -> torch.Tensor:
    if backend.attn == "sdpa":
        x = x.transpose(1, 2).contiguous()
    return x


class Qwen3MoeAttention(nn.Module):
    """Qwen3 MoE attention (query/key per-head RMSNorm + RoPE) compatible with TE/SDPA backends.

    Shapes:
      - Input: x -> [B, S, H]
      - Projections:
          q: [B, S, n_heads, head_dim]
          k/v: [B, S, n_kv_heads, head_dim] -> repeated to n_heads via groups
      - Output: [B, S, H]
    """

    def __init__(self, config: Qwen3MoeConfig, backend: BackendConfig):
        super().__init__()
        self.backend = backend

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)

        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, False
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, False
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, False
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, False
        )

        # Per-head RMSNorm
        self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)

        # Attention implementation
        softmax_scale = self.head_dim**-0.5
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=softmax_scale,
            num_gqa_groups=self.num_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        # Projections
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (complex rotation)
        cos, sin = freqs_cis.split(self.head_dim // 2, dim=-1)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # Backend-specific attention
        q, k, v, _attn_kwargs = _preprocess_for_attn(q, k, v, attention_mask, self.backend)
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = _postprocess_from_attn(out, self.backend)

        out = self.o_proj(out.flatten(2))
        return out

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [self.q_proj, self.k_proj, self.v_proj, self.o_proj]
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()
