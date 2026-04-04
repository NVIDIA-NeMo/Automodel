# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Afmoe attention layer with gated output, QK normalization, and conditional RoPE."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk


class AfmoeAttention(nn.Module):
    """Afmoe attention with gated output, per-head QK RMSNorm, and conditional RoPE.

    Key differences from standard attention:
      - RoPE is applied only to sliding-window (local) attention layers.
      - Attention output is gated: ``output = output * sigmoid(gate_proj(x))``.
      - Per-head RMSNorm on Q and K before attention.
    """

    def __init__(self, config, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.backend = backend
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None

        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.gate_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        # Per-head RMSNorm on Q and K
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
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
        else:
            qkv_format = "bshd"
            bsz, seqlen, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        gate = self.gate_proj(x)

        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE only on local (sliding-window) attention layers
        if self.is_local_attention:
            q, k = apply_rotary_emb_qk(
                q,
                k,
                freqs_cis,
                format=qkv_format,
                rope_fusion=self.backend.rope_fusion,
                cu_seqlens=attn_kwargs.get("cu_seqlens", None),
                cp_size=attn_kwargs.get("cp_size", 1),
                cp_rank=attn_kwargs.get("cp_rank", 0),
            )

        # Backend-specific attention
        window_size = (self.sliding_window, 0) if self.is_local_attention else (-1, 0)
        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, window_size=window_size, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        out = out.flatten(flatten_dim)

        # Gated attention output
        out = out * F.sigmoid(gate)
        out = self.o_proj(out)
        return out

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.gate_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()
