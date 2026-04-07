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
from transformers import Qwen3Config

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
from nemo_automodel.components.models.common.combined_projection.combined_qkv import (
    CombinedQKVAttentionMixin,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk


class Qwen3Attention(CombinedQKVAttentionMixin, nn.Module):
    """Qwen3 dense attention with combined QKV, per-head RMSNorm, RoPE, and TE/SDPA backends.

    Key differences from Qwen2:
    - Per-head q_norm and k_norm (RMSNorm) applied after QKV split, before RoPE
    - No attention bias (Qwen3 config has attention_bias=False)
    - Supports both bshd (standard) and thd (sequence-packed) formats for CP

    Combined QKV layout for TP: [Q_group_0 | K_0 | V_0 | Q_group_1 | K_1 | V_1 | ...]
    This layout ensures ColwiseParallel TP sharding gives each rank complete KV-head groups.
    """

    def __init__(self, config: Qwen3Config, backend: BackendConfig):
        super().__init__()
        self.backend = backend

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)

        attention_bias = getattr(config, "attention_bias", False)

        # Manually set up CombinedQKVAttentionMixin attributes so we can use
        # initialize_linear_module for TE Linear (FP8) support.
        self.use_combined_qkv = True
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self._num_kv_groups = self.num_heads // self.num_kv_heads
        self._head_dim = self.head_dim
        self._tp_checked = False

        total_qkv_out = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = initialize_linear_module(backend.linear, config.hidden_size, total_qkv_out, attention_bias)
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, False
        )

        # Per-head RMSNorm -- distinguishes Qwen3 from Qwen2
        self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)

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

        # Combined QKV projection with KV-head-grouped de-interleaving (mixin)
        q, k, v = self.compute_qkv(x)

        # Reshape to expose head dimension for per-head norms
        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm (Qwen3-specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE with CP support
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

        # Backend-specific attention (TE with CP or SDPA)
        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        out = self.o_proj(out.flatten(flatten_dim))
        return out

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        for linear in [self.qkv_proj, self.o_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(linear, "bias") and linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()
