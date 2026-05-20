# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Minimal Llama-based draft model for EAGLE-1 / EAGLE-2 training."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaConfig, PreTrainedModel

from nemo_automodel.components.models.common import initialize_rms_norm_module
from nemo_automodel.components.models.llama.rope_utils import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


def _build_causal_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a standard causal + padding mask for eager attention."""
    batch_size, seq_len = attention_mask.shape
    causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=attention_mask.device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    expanded = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    return causal + expanded


class EagleLlamaAttention(nn.Module):
    """Standard Llama-style self attention for the EAGLE-1/2 draft."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def _repeat_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return tensor
        return tensor.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class EagleLlamaMLP(nn.Module):
    """Standard SwiGLU MLP used by the EAGLE-1/2 draft."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class EagleLlamaDecoderLayer(nn.Module):
    """Single decoder layer for the minimal EAGLE-1/2 draft model."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = EagleLlamaAttention(config)
        self.mlp = EagleLlamaMLP(config)
        self.input_layernorm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LlamaEagleDraftModel(PreTrainedModel):
    """Minimal Llama draft that predicts next-step hidden states."""

    config_class = LlamaConfig
    main_input_name = "input_ids"

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        num_layers = max(1, int(getattr(config, "draft_num_hidden_layers", config.num_hidden_layers)))
        self.layers = nn.ModuleList([EagleLlamaDecoderLayer(config) for _ in range(num_layers)])
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def copy_embeddings_from_target(self, target_embeddings: nn.Embedding) -> None:
        """Copy the target model token embeddings into the draft embeddings."""
        with torch.no_grad():
            self.embed_tokens.weight.copy_(target_embeddings.weight.to(self.embed_tokens.weight.device))

    def freeze_embeddings(self) -> None:
        """Freeze draft token embeddings."""
        self.embed_tokens.weight.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids).to(target_hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, target_hidden_states), dim=-1))

        batch_size, seq_len, _ = hidden_states.shape
        position_ids = (
            torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        causal_mask = _build_causal_mask(attention_mask, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)
        return self.norm(hidden_states)
