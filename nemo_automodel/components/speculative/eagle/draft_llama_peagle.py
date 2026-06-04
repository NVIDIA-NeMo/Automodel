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

"""Llama-style draft model for P-EAGLE parallel-drafting training.

P-EAGLE (arXiv:2602.01469) keeps the EAGLE-3 drafter shape -- a single
Llama-style decoder layer fed ``[token_embedding, projected_target_hidden]`` --
but trains it to draft ``num_depths`` tokens per position in one forward pass.
The depth>=1 prediction positions have no real preceding hidden state, so they
share a single learnable ``mask_hidden`` vector; positional information for those
positions comes purely from attention. Because COD sampling packs all depths of a
sequence onto one flat axis, attention runs through a FlexAttention block mask
(see :func:`peagle_attention.create_peagle_mask_mod`) rather than the EAGLE-3
test-time-training KV cache.

State-dict layout matches the EAGLE-3 Llama draft (``model.embed_tokens``,
``model.fc``, ``model.layers.0.*``, ``model.norm``, ``lm_head``) plus the extra
``model.mask_hidden`` parameter, so trained checkpoints reuse the EAGLE-3 export
path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from transformers import PretrainedConfig, PreTrainedModel

from nemo_automodel.components.models.common import initialize_rms_norm_module
from nemo_automodel.components.models.llama.rope_utils import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from nemo_automodel.components.speculative.eagle.draft_llama import Eagle3LlamaMLP


class PEagleLlamaAttention(nn.Module):
    """GQA attention over ``[input_emb, hidden]`` 2H features via FlexAttention.

    Unlike the EAGLE-3 attention (which threads a per-TTT-step KV cache), this
    runs a single parallel pass: the COD block mask supplied per batch encodes
    both the depth-0 causal context and the in-rollout depth ordering.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        in_features = config.hidden_size * 2
        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(in_features, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(in_features, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(in_features, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        combined_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
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

        cos, sin = self.rotary_emb(combined_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = flex_attention(q, k, v, block_mask=block_mask, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class PEagleLlamaDecoderLayer(nn.Module):
    """Single P-EAGLE draft layer: norm(embeds) || norm(hidden) -> attn -> MLP."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.input_layernorm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PEagleLlamaAttention(config)
        self.mlp = Eagle3LlamaMLP(config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        residual = hidden_states
        combined_states = torch.cat((self.input_layernorm(input_embeds), self.hidden_norm(hidden_states)), dim=-1)
        hidden_states = residual + self.self_attn(combined_states, position_ids, block_mask)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class PEagleLlamaModel(nn.Module):
    """Inner backbone: embeddings, target-hidden projection, draft layer, final norm."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        num_aux_hidden_states = getattr(config, "num_aux_hidden_states", 3)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.fc = nn.Linear(target_hidden_size * num_aux_hidden_states, config.hidden_size, bias=False)
        # Optional EAGLE-3.1 per-aux RMSNorm applied to each target hidden-state
        # chunk before ``fc`` (counters the attention drift that worsens with
        # speculation depth -- a regime parallel drafting exercises heavily).
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    initialize_rms_norm_module("torch", target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(num_aux_hidden_states)
                ]
            )
        # Shared learnable hidden state used for every depth>=1 position (which
        # has no real preceding target hidden state). Sized to the pre-projection
        # ``num_aux * target_hidden_size`` so it is substituted before ``fc``.
        self.mask_hidden = nn.Parameter(torch.randn(1, 1, target_hidden_size * num_aux_hidden_states) * 0.02)
        self.layers = nn.ModuleList([PEagleLlamaDecoderLayer(config)])
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)


class LlamaPEagleDraftModel(PreTrainedModel):
    """P-EAGLE draft model (Llama / Phi-3 / Qwen3-style dense targets)."""

    config_class = PretrainedConfig
    base_model_prefix = "model"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)

        self.model = PEagleLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)
        self.post_init()

    @property
    def mask_hidden(self) -> nn.Parameter:
        return self.model.mask_hidden

    def copy_embeddings_from_target(self, target_embedding: nn.Embedding) -> None:
        """Seed draft embeddings from the target (gathering DTensor shards if needed)."""
        target_weight = target_embedding.weight
        if hasattr(target_weight, "full_tensor"):
            target_weight = target_weight.full_tensor()
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(target_weight)

    def freeze_embeddings(self) -> None:
        """Freeze draft input embeddings. P-EAGLE trains them by default, so this is opt-in."""
        self.model.embed_tokens.weight.requires_grad_(False)

    def project_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated target aux states (``num_aux * H_target``) to draft hidden size.

        With ``config.fc_norm`` set, each aux chunk is RMS-normed independently
        (EAGLE-3.1) before the chunks are re-concatenated and projected.
        """
        if getattr(self.config, "fc_norm", False):
            chunks = aux_hidden_states.chunk(len(self.model.fc_norm), dim=-1)
            aux_hidden_states = torch.cat([norm(chunk) for norm, chunk in zip(self.model.fc_norm, chunks)], dim=-1)
        return self.model.fc(aux_hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model.norm(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        projected_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
        cache_hidden: Optional[list] = None,  # unused; kept for API symmetry with EAGLE-3
    ) -> torch.Tensor:
        """Run the parallel draft pass over a COD-packed sequence.

        Args:
            input_ids: ``[1, total_sampled]`` packed token ids (mask token at depth>=1).
            projected_hidden_states: ``[1, total_sampled, H]`` post-``fc`` hidden states.
            position_ids: ``[1, total_sampled]`` original sequence positions.
            block_mask: a FlexAttention ``BlockMask`` from ``create_peagle_mask_mod``.
        """
        input_embeds = self.embed_input_ids(input_ids)
        return self.model.layers[0](
            input_embeds=input_embeds,
            hidden_states=projected_hidden_states,
            position_ids=position_ids,
            block_mask=block_mask,
        )
