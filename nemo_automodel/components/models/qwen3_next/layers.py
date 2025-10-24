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
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb
from nemo_automodel.components.moe.utils import (
    BackendConfig,
    initialize_linear_module,
)


class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class Qwen3NextAttention(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.backend = backend
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Query projection outputs 2x size for gating
        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim * 2, False
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

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Attention implementation
        self.scaling = self.head_dim**-0.5
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=self.scaling,
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
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
        else:
            qkv_format = "bshd"
            bsz, seqlen, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim * 2)
            k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim * 2)
            k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q, gate = torch.chunk(q, 2, dim=-1)
        gate = gate.reshape(*x.shape[:-1], -1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE (split freqs_cis by half, accounting for partial rotary factor)
        cos, sin = freqs_cis.split(freqs_cis.shape[-1] // 2, dim=-1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Backend-specific attention
        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        attn_output = self.attn_func(q, k, v, **_attn_kwargs)
        attn_output = postprocess_output_for_attn(attn_output, self.backend.attn)

        # Reshape and apply gating
        attn_output = attn_output.reshape(*x.shape[:-1], -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        # Output projection
        attn_output = self.o_proj(attn_output)
        return attn_output

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [self.q_proj, self.k_proj, self.v_proj, self.o_proj]
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()
