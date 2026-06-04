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

"""Attention layer for BailingMoeV2 (Ling 2.0).

GQA + per-head QK-RMSNorm + partial RoPE.  Equivalent to Qwen3-MoE attention
with an additional ``partial_rotary_factor`` knob that rotates only the first
``head_dim * partial_rotary_factor`` channels and passes the rest through
(GPT-J / GPT-NeoX half-RoPE).
"""

from typing import Any

import torch
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


class BailingMoeV2Attention(nn.Module):
    """Bailing MoE V2 attention block.

    Shapes:
      - Input ``x``: ``[B, S, H]`` (or ``[T, H]`` in THD format).
      - Projections:
          ``q``: ``[B, S, n_heads, head_dim]``
          ``k, v``: ``[B, S, n_kv_heads, head_dim]``
      - Output: ``[B, S, H]``.
    """

    def __init__(self, config, backend: BackendConfig):
        super().__init__()
        self.backend = backend

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (config.hidden_size // self.num_heads)
        self.use_qk_norm = bool(getattr(config, "use_qk_norm", False))

        attention_bias = bool(getattr(config, "use_qkv_bias", False))
        out_bias = bool(getattr(config, "use_bias", False))

        # Separate q / k / v projections.  HF Bailing fuses them into a single
        # ``query_key_value`` linear; the state_dict adapter splits that fused
        # tensor at load time so this module can stay aligned with the rest of
        # the framework (and with TP plans that expect separate q/k/v).
        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, attention_bias
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, out_bias
        )

        if self.use_qk_norm:
            self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

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

        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ``freqs_cis`` is shaped for the rotary half (``rotary_dim``) when
        # ``partial_rotary_factor < 1``; ``apply_rotary_emb_qk`` -> ``apply_rotary_emb``
        # detects that and only rotates the first ``rotary_dim`` channels of each head.
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

        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        out = self.o_proj(out.flatten(flatten_dim))
        return out

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if getattr(linear, "bias", None) is not None:
                nn.init.zeros_(linear.bias)
        if self.use_qk_norm:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()
