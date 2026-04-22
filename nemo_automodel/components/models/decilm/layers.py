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

"""DeciLM layer implementations for CP / packed-sequence / TP support.

DeciLM is structurally a stripped-down Llama with per-layer variable GQA
(via ``block_configs``).  Each layer can be:
  - Normal GQA attention (``n_heads_in_group`` set, ``no_op=False``)
  - No-op (attention skipped entirely)
  - Linear replacement (attention replaced with a single linear projection)

The same applies to the FFN block (with ``ffn_mult`` controlling width).
"""

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
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk


def _ffn_mult_to_intermediate_size(ffn_mult: float, hidden_size: int) -> int:
    """Compute per-layer intermediate size from ffn_mult (matches HF DeciLM)."""
    intermediate_size = int(2 * ffn_mult * hidden_size / 3)
    # Round up to nearest multiple of 256
    if intermediate_size % 256 != 0:
        intermediate_size = intermediate_size + 256 - (intermediate_size % 256)
    return intermediate_size


class DeciLMAttention(nn.Module):
    """DeciLM GQA attention with separate QKV projections, RoPE, and TE/SDPA.

    Uses separate q_proj / k_proj / v_proj so native key names match the HF
    checkpoint directly -- no state-dict conversion needed.

    Uses nn.Linear (not TE linear) for projections so PyTorch
    ColwiseParallel / RowwiseParallel can shard them without errors.

    Key difference from Llama: per-layer variable GQA via n_heads_in_group.
    Key difference from Qwen3: no per-head RMSNorm (q_norm / k_norm).

    Supports both bshd (standard) and thd (sequence-packed + CP) formats.
    """

    def __init__(self, config, attention_config, backend: BackendConfig):
        super().__init__()
        self.backend = backend

        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        # Per-layer variable GQA
        self.num_key_value_groups = attention_config.n_heads_in_group
        self.num_key_value_heads = self.num_heads // self.num_key_value_groups

        attention_bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=attention_bias)

        softmax_scale = self.head_dim**-0.5
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=softmax_scale,
            num_gqa_groups=self.num_key_value_heads,
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
            k = k.view(num_tokens, self.num_key_value_heads, self.head_dim)
            v = v.view(num_tokens, self.num_key_value_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        # RoPE with CP support (no per-head norms unlike Qwen3)
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
        for linear in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(linear, "bias") and linear.bias is not None:
                nn.init.zeros_(linear.bias)


class DeciLMLinearAttention(nn.Module):
    """Linear replacement for attention (matches HF checkpoint key ``self_attn.linear_attn``)."""

    def __init__(self, config):
        super().__init__()
        self.linear_attn = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.linear_attn(x)


class DeciLMMLP(nn.Module):
    """SiLU-gated MLP with per-layer variable intermediate size from ffn_mult.

    Uses nn.Linear so ColwiseParallel / RowwiseParallel can shard the projections.
    Key names (gate_proj, up_proj, down_proj) match HF checkpoint directly.
    """

    def __init__(self, config, ffn_config, backend: str = "torch"):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = _ffn_mult_to_intermediate_size(ffn_config.ffn_mult, hidden_size)
        mlp_bias = getattr(config, "mlp_bias", False)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        for linear in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(linear, "bias") and linear.bias is not None:
                nn.init.zeros_(linear.bias)


class DeciLMLinearMLP(nn.Module):
    """Linear replacement for MLP (matches HF checkpoint key ``mlp.linear_mlp``)."""

    def __init__(self, config):
        super().__init__()
        self.linear_mlp = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_mlp(x)


class Block(nn.Module):
    """Single DeciLM decoder layer with per-layer block_config dispatch.

    Conditionally creates attention/MLP modules based on block_config:
      - no_op: skip entirely (no module created)
      - replace_with_linear: use linear projection instead
      - normal: full GQA attention / SiLU-gated MLP
    """

    def __init__(self, layer_idx: int, config, backend: BackendConfig):
        super().__init__()
        self.layer_idx = layer_idx
        block_config = config.block_configs[layer_idx]
        self.attention_config = block_config.attention
        self.ffn_config = block_config.ffn

        if not self.attention_config.no_op:
            self.input_layernorm = initialize_rms_norm_module(
                backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
            )
            if not self.attention_config.replace_with_linear:
                self.self_attn = DeciLMAttention(config, self.attention_config, backend)
            else:
                self.self_attn = DeciLMLinearAttention(config)

        if not self.ffn_config.no_op:
            self.post_attention_layernorm = initialize_rms_norm_module(
                backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
            )
            if not self.ffn_config.replace_with_linear:
                self.mlp = DeciLMMLP(config, self.ffn_config)
            else:
                self.mlp = DeciLMLinearMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        # Attention block
        if self.attention_config.no_op:
            pass
        elif self.attention_config.replace_with_linear:
            residual = x
            x = self.input_layernorm(x)
            x = self.self_attn(x)
            x = residual + x
        else:
            residual = x
            x = self.self_attn(
                x=self.input_layernorm(x),
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                **attn_kwargs,
            )
            x = residual + x

        # FFN block
        if not self.ffn_config.no_op:
            residual = x
            x = self.mlp(self.post_attention_layernorm(x))
            x = residual + x

        return x

    def init_weights(self, buffer_device: torch.device):
        if not self.attention_config.no_op:
            self.input_layernorm.reset_parameters()
            if not self.attention_config.replace_with_linear and hasattr(self.self_attn, "init_weights"):
                self.self_attn.init_weights(buffer_device)
        if not self.ffn_config.no_op:
            self.post_attention_layernorm.reset_parameters()
            if not self.ffn_config.replace_with_linear and hasattr(self.mlp, "init_weights"):
                self.mlp.init_weights(buffer_device)
