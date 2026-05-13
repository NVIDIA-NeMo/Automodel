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

"""Minimal Llama-based draft model for EAGLE-3 training."""

from __future__ import annotations

from typing import Optional

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
    """Build a standard causal + padding mask for SDPA/eager attention."""
    batch_size, seq_len = attention_mask.shape
    causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=attention_mask.device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    expanded = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    return causal + expanded


class Eagle3LlamaAttention(nn.Module):
    """EAGLE-3 draft attention over ``[input_emb, hidden]`` 2H features.

    Supports two paths:

    - ``cache_hidden is None`` -- single forward, regular causal attention
      (used for evaluation and for tests).
    - ``cache_hidden = [[K_0, K_1, ...], [V_0, V_1, ...]]`` -- TTT
      recurrence (test-time-training). At each call:

      * the rotary position is shifted by ``lck = len(cache_hidden[0])``
        so the RoPE encodes "this is ``lck`` tokens into the future";
      * the freshly-computed K and V (already GQA-expanded) are appended
        to ``cache_hidden``;
      * the output is the SpecForge ``llama3_eagle.py`` recurrence:
        full ``Q @ K_0`` (with the standard causal mask added) plus, for
        every cached later step ``i >= 1``, a *diagonal* contribution
        ``(Q_t * K_i_t).sum(-1)`` -- each Q position only attends to the
        *same* position in each previous draft step, never to another
        position. This is the core EAGLE-3 multi-step training trick.

    ``cache_hidden`` is mutated in place; the caller is responsible for
    re-initializing it (typically ``[[], []]``) at the start of each
    training batch.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        in_features = config.hidden_size * 2
        self.q_proj = nn.Linear(in_features, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(in_features, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(in_features, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def _project_qkv(self, combined_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = combined_states.shape
        q = self.q_proj(combined_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(combined_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(combined_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        return q, k, v

    def _repeat_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        return k, v

    def forward(
        self,
        combined_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: Optional[list[list[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = combined_states.shape
        q, k, v = self._project_qkv(combined_states)

        if cache_hidden is None:
            # Single-step path: regular causal attention over [B, H, T, T].
            cos, sin = self.rotary_emb(combined_states, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k, v = self._repeat_kv(k, v)

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            attn_weights = attn_weights + attention_mask
            attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)
            attn_output = torch.matmul(attn_probs, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.o_proj(attn_output)

        # TTT recurrence path. ``cache_hidden`` is ``[K_list, V_list]``.
        lck = len(cache_hidden[0])
        cos, sin = self.rotary_emb(combined_states, position_ids + lck)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = self._repeat_kv(k, v)

        cache_hidden[0].append(k)
        cache_hidden[1].append(v)
        cache_k = cache_hidden[0]
        cache_v = cache_hidden[1]
        new_lck = len(cache_k)

        # Standard T x T attention against the step-0 keys.
        k0 = cache_k[0]
        v0 = cache_v[0]
        attn_weights = torch.matmul(q, k0.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask

        # Diagonal extensions: one column per cached later step ``i``,
        # holding ``(Q_t * K_i_t).sum(-1) / sqrt(d)`` for every Q position
        # ``t``. Each new column is unmasked because the diagonal is
        # intrinsically causal across draft steps.
        for i in range(1, new_lck):
            ki = cache_k[i]
            attn_weightsi = (q * ki).sum(-1) * self.scaling
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)

        # Output: ``attn_probs[..., :T] @ V_0`` covers the regular block;
        # for each cached step ``i >= 1``, the single column at
        # ``attn_probs[..., T + i - 1]`` scales the same-position ``V_i``.
        attn_output = torch.matmul(attn_probs[..., :seq_len], v0)
        for i in range(1, new_lck):
            vi = cache_v[i]
            probs_i = attn_probs[..., seq_len + i - 1]  # [B, H, T]
            attn_output = attn_output + probs_i[..., None] * vi

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class Eagle3LlamaMLP(nn.Module):
    """Standard Llama SwiGLU MLP on hidden-size activations."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        from transformers.activations import ACT2FN

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Eagle3LlamaDecoderLayer(nn.Module):
    """Single decoder layer used by the minimal EAGLE-3 draft model."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_emb_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.hidden_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.self_attn = Eagle3LlamaAttention(config)
        self.mlp = Eagle3LlamaMLP(config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: Optional[list[list[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_input_embeds = self.input_emb_layernorm(input_embeds)
        norm_hidden_states = self.hidden_layernorm(hidden_states)
        combined_states = torch.cat((norm_input_embeds, norm_hidden_states), dim=-1)
        hidden_states = residual + self.self_attn(
            combined_states,
            attention_mask,
            position_ids,
            cache_hidden=cache_hidden,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class LlamaEagle3DraftModel(PreTrainedModel):
    """Minimal Llama-only EAGLE-3 draft model.

    This intentionally starts narrow:
    - Llama config only
    - single draft decoder layer
    - no KV-cache optimization
    - no speculative runtime integration
    """

    config_class = LlamaConfig
    base_model_prefix = "draft_model"

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.hidden_proj = nn.Linear(self.target_hidden_size * 3, config.hidden_size, bias=False)
        self.decoder = Eagle3LlamaDecoderLayer(config)
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, device=None)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        self.post_init()

    def copy_embeddings_from_target(self, target_embedding: nn.Embedding) -> None:
        """Initialize draft embeddings from the target model embeddings."""
        with torch.no_grad():
            self.embed_tokens.weight.copy_(target_embedding.weight)

    def freeze_embeddings(self) -> None:
        """Freeze draft input embeddings."""
        self.embed_tokens.weight.requires_grad_(False)

    def project_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated target aux states from ``3 * H_target`` to draft hidden size."""
        return self.hidden_proj(aux_hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids with the draft embedding table."""
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute draft logits on the configured draft vocabulary."""
        return self.lm_head(self.norm(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        projected_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache_hidden: Optional[list[list[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Run one full-sequence draft update step.

        ``cache_hidden`` activates the EAGLE-3 TTT recurrence in the
        attention layer. Pass ``[[], []]`` on the first step of a TTT
        unroll and the same list object on each subsequent step; the
        attention layer appends the per-step K and V to it.
        """
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(input_ids.shape[0], -1)

        draft_input_embeds = self.embed_input_ids(input_ids)
        causal_mask = _build_causal_mask(attention_mask=attention_mask, dtype=projected_hidden_states.dtype)
        return self.decoder(
            input_embeds=draft_input_embeds,
            hidden_states=projected_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_hidden=cache_hidden,
        )
