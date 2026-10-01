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

"""Decoder layers of the HunyuanImage-3.0 MoE backbone."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.hunyuan_image3.config import per_layer
from nemo_automodel.components.models.hunyuan_image3.rope import apply_rope
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE


class HunyuanImage3RMSNorm(nn.Module):
    """RMSNorm that normalises in float32 and returns ``weight * x`` in the input dtype's promotion."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class HunyuanImage3MLP(nn.Module):
    """SwiGLU MLP with a fused input projection whose *second* half is the gate: ``down(x1 * silu(x2))``."""

    def __init__(self, dim: int, inter_dim: int, backend: BackendConfig, bias: bool = False):
        super().__init__()
        self.gate_and_up_proj = initialize_linear_module(backend.linear, dim, 2 * inter_dim, bias=bias)
        self.down_proj = initialize_linear_module(backend.linear, inter_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.gate_and_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(up * F.silu(gate))

    def init_weights(self, std: float = 0.02) -> None:
        for linear in (self.gate_and_up_proj, self.down_proj):
            nn.init.normal_(linear.weight, std=std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)


class HunyuanImage3Attention(nn.Module):
    """GQA attention with a fused QKV projection grouped per KV head, RoPE applied before per-head QK RMSNorm."""

    def __init__(self, config: Any, backend: BackendConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.groups = self.num_heads // self.num_kv_heads
        hidden = config.hidden_size
        self.qkv_proj = initialize_linear_module(
            backend.linear, hidden, self.head_dim * (self.num_heads + 2 * self.num_kv_heads), bias=config.attention_bias
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.head_dim * self.num_heads, hidden, bias=config.attention_bias
        )
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = HunyuanImage3RMSNorm(self.head_dim, config.rms_norm_eps)
            self.key_layernorm = HunyuanImage3RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[B, S, hidden]``.
            cos, sin: ``[B, S, head_dim]`` rotary tables (float32).
            attention_mask: ``[B, 1, S, S]`` boolean (True = attend) or None for plain causal attention.
        """
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(bsz, seq_len, self.num_kv_heads, self.groups + 2, self.head_dim)
        q, k, v = qkv.split([self.groups, 1, 1], dim=3)
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        if attention_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attention_mask
            )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)

    def init_weights(self, std: float = 0.02) -> None:
        for linear in (self.qkv_proj, self.o_proj):
            nn.init.normal_(linear.weight, std=std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        if self.use_qk_norm:
            self.query_layernorm.reset_parameters()
            self.key_layernorm.reset_parameters()


class HunyuanImage3Block(nn.Module):
    """Pre-norm decoder layer: attention, then routed MoE plus an always-on shared expert MLP."""

    def __init__(self, layer_idx: int, config: Any, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        hidden = config.hidden_size
        self.layer_idx = layer_idx
        self.input_layernorm = HunyuanImage3RMSNorm(hidden, config.rms_norm_eps)
        self.self_attn = HunyuanImage3Attention(config, backend)
        self.post_attention_layernorm = HunyuanImage3RMSNorm(hidden, config.rms_norm_eps)

        self.is_moe = config.num_experts > 1 and layer_idx >= config.moe_layer_num_skipped
        if self.is_moe:
            self.mlp = MoE(moe_config, backend)
            if config.use_mixed_mlp_moe:
                shared_inter = per_layer(config.moe_intermediate_size, layer_idx) * per_layer(
                    config.num_shared_expert, layer_idx
                )
                self.shared_mlp = HunyuanImage3MLP(hidden, shared_inter, backend, bias=config.mlp_bias)
            else:
                self.shared_mlp = None
        else:
            self.mlp = HunyuanImage3MLP(hidden, config.intermediate_size, backend, bias=config.mlp_bias)
            self.shared_mlp = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)
        h = self.post_attention_layernorm(x)
        if self.is_moe:
            out = self.mlp(h, padding_mask)
            if self.shared_mlp is not None:
                out = self.shared_mlp(h) + out
        else:
            out = self.mlp(h)
        return x + out

    def init_weights(self, buffer_device: torch.device) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights()
        if self.is_moe:
            self.mlp.init_weights(buffer_device)
            if self.shared_mlp is not None:
                self.shared_mlp.init_weights()
        else:
            self.mlp.init_weights()
