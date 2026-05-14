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

"""Minimal Llama-based draft model for EAGLE-3 training.

Module naming is aligned to ``sglang/srt/models/llama_eagle3.py`` so that a
checkpoint produced by this trainer can be loaded directly by SGLang's
``LlamaForCausalLMEagle3.load_weights`` without any key remapping. The state
dict layout is:

    model.embed_tokens.weight
    model.fc.weight
    model.layers.0.input_layernorm.weight
    model.layers.0.hidden_norm.weight
    model.layers.0.post_attention_layernorm.weight
    model.layers.0.self_attn.{q,k,v,o}_proj.weight
    model.layers.0.mlp.{gate,up,down}_proj.weight
    model.norm.weight
    lm_head.weight

SGLang merges ``q_proj/k_proj/v_proj`` into a single ``qkv_proj`` and
``gate_proj/up_proj`` into ``gate_up_proj`` via its ``stacked_params_mapping``
at load time, so the un-fused storage above is the canonical on-disk format.
"""

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

    Driven through a shared ``cache_hidden = [K_list, V_list]`` pair. At
    step ``k`` (0-indexed), with ``K_list`` and ``V_list`` already holding
    entries from steps ``0..k-1``:

    1. ``step_idx = len(K_list)`` (equal to ``k``) gives the rotary phase
       shift, so the draft's ``K_k`` encodes "this is ``k`` tokens into
       the future". The shifted ``cos`` / ``sin`` are computed from
       ``position_ids + step_idx``.
    2. The freshly projected K, V (after GQA expansion) are appended to
       the cache lists in place.
    3. The attention output is the EAGLE-3 mixed pattern:

       ``attn_weights = [ Q @ K_0^T / sqrt(d) + mask ]  ||  diag_1  ||  ...  ||  diag_k``

       where ``diag_i[t] = (Q_t * K_i_t).sum(-1) / sqrt(d)``. The softmax
       is taken over the full extended column axis of length ``T + k``.
       Output is

       ``out = attn_probs[..., :T] @ V_0  +  sum_{i=1..k} attn_probs[..., T+i-1, None] * V_i``.

       In English: Q at position ``t`` attends to all K_0 positions (the
       regular ``T x T`` causal block), and additionally to the *same*
       position ``t`` in each previous draft step ``i >= 1``.
       Implementation-wise we replace SpecForge ``llama3_eagle.py``'s
       two ``O(k^2)`` ``cat`` / ``add`` Python loops with single
       vectorized ``einsum`` calls.

    ``cache_hidden`` is mutated in place; callers are responsible for
    re-initializing it to ``[[], []]`` at the start of each training
    batch.
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
        cache_hidden: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = combined_states.shape
        q, k, v = self._project_qkv(combined_states)

        # ``step_idx`` is the cache length BEFORE this step's append; it
        # equals the 0-indexed TTT step number and doubles as the rotary
        # phase shift. After the append below the cache holds
        # ``step_idx + 1`` entries (indices ``0..step_idx``). On the first
        # call ``cache_hidden = [[], []]`` so ``step_idx = 0`` and the
        # diagonal-extension blocks below collapse to a plain causal
        # attention, equivalent to the non-cached path.
        cache_k, cache_v = cache_hidden
        step_idx = len(cache_k)

        cos, sin = self.rotary_emb(combined_states, position_ids + step_idx)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = self._repeat_kv(k, v)
        cache_k.append(k)
        cache_v.append(v)

        # Block 1: full T x T causal attention against the step-0 keys.
        k0, v0 = cache_k[0], cache_v[0]
        attn_weights = torch.matmul(q, k0.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask  # [B, 1, T, T] additive mask

        # Block 2: diagonal extensions for cached later steps ``i = 1..step_idx``.
        # Each contributes one column ``(Q_t * K_i_t).sum(-1) / sqrt(d)``,
        # i.e. Q at position ``t`` attends only to position ``t`` of
        # ``K_i``. Replaces SpecForge's ``O(k^2)`` cat-in-loop with a
        # single ``einsum`` + single ``cat``.
        if step_idx >= 1:
            later_k = torch.stack(cache_k[1:], dim=0)  # [step_idx, B, H, T, D]
            diag = torch.einsum("bhtd,sbhtd->bhts", q, later_k) * self.scaling
            attn_weights = torch.cat((attn_weights, diag), dim=-1)

        # Block 3: softmax over the extended ``T + step_idx`` key axis.
        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)

        # Block 4: output =
        #   ``attn_probs[..., :T] @ V_0``  (regular T x T block)
        # + ``sum_{i=1..step_idx} attn_probs[..., T+i-1, None] * V_i``
        # Same fusion as Block 2 -- one ``einsum`` instead of an O(k^2)
        # accumulator loop.
        attn_output = torch.matmul(attn_probs[..., :seq_len], v0)
        if step_idx >= 1:
            later_v = torch.stack(cache_v[1:], dim=0)  # [step_idx, B, H, T, D]
            diag_probs = attn_probs[..., seq_len:]  # [B, H, T, step_idx]
            attn_output = attn_output + torch.einsum("bhts,sbhtd->bhtd", diag_probs, later_v)

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
    """Single decoder layer used by the minimal EAGLE-3 draft model.

    Attribute names mirror SGLang's ``LlamaDecoderLayer`` in
    ``sglang/srt/models/llama_eagle3.py``: ``input_layernorm`` is applied
    to the per-step token embeddings (``embeds`` in SGLang),
    ``hidden_norm`` is applied to the carried hidden state.
    ``is_input_layer`` is the layer-0 flag that gates the ``[embeds,
    hidden]`` concatenation (always true for our single-layer draft).
    """

    def __init__(self, config: LlamaConfig, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.is_input_layer = layer_id == 0
        self.input_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.hidden_norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, device=None)
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
        cache_hidden: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        residual = hidden_states
        norm_input_embeds = self.input_layernorm(input_embeds)
        norm_hidden_states = self.hidden_norm(hidden_states)
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


class Eagle3LlamaModel(nn.Module):
    """Inner backbone matching SGLang's ``LlamaModel`` in ``llama_eagle3.py``.

    Owns ``embed_tokens``, the ``fc`` projection from concatenated target
    aux hidden states to draft hidden size, the (single-element) draft
    ``layers`` ModuleList, and the final ``norm``. The ``LlamaEagle3DraftModel``
    wrapper around this module adds the top-level ``lm_head`` and the
    training-facing public API.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        # SGLang uses ``num_aux_hidden_states`` (default 3) to size ``fc``'s
        # input dim. We mirror that convention so the weight shape is
        # identical and the key ``model.fc.weight`` round-trips cleanly.
        num_aux_hidden_states = getattr(config, "num_aux_hidden_states", 3)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.fc = nn.Linear(target_hidden_size * num_aux_hidden_states, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([Eagle3LlamaDecoderLayer(config, layer_id=0)])
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, device=None)


class LlamaEagle3DraftModel(PreTrainedModel):
    """Minimal Llama-only EAGLE-3 draft model.

    State dict keys match SGLang's ``LlamaForCausalLMEagle3`` so the saved
    checkpoint can be loaded by SGLang's inference engine without any
    remapping (SGLang's ``load_weights`` fuses ``q/k/v_proj`` into
    ``qkv_proj`` and ``gate/up_proj`` into ``gate_up_proj`` via its
    standard ``stacked_params_mapping``).

    This intentionally starts narrow:
    - Llama config only
    - single draft decoder layer
    - no KV-cache optimization
    - no speculative runtime integration
    """

    config_class = LlamaConfig
    base_model_prefix = "model"

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)

        self.model = Eagle3LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        self.post_init()

    def copy_embeddings_from_target(self, target_embedding: nn.Embedding) -> None:
        """Initialize draft embeddings from the target model embeddings."""
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(target_embedding.weight)

    def freeze_embeddings(self) -> None:
        """Freeze draft input embeddings."""
        self.model.embed_tokens.weight.requires_grad_(False)

    def project_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated target aux states from ``num_aux * H_target`` to draft hidden size."""
        return self.model.fc(aux_hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids with the draft embedding table."""
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute draft logits on the configured draft vocabulary."""
        return self.lm_head(self.model.norm(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        projected_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache_hidden: Optional[list[list[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Run one full-sequence draft update step.

        ``cache_hidden`` is the EAGLE-3 TTT cache. Pass ``[[], []]`` on
        the first step of a TTT unroll and the same list object on each
        subsequent step; the attention layer appends the per-step K and V
        to it. If ``None`` is passed (e.g. from a one-shot evaluation
        call) a fresh ``[[], []]`` is allocated locally -- step 0 of TTT
        is mathematically equivalent to a plain causal forward.
        """
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(input_ids.shape[0], -1)
        if cache_hidden is None:
            cache_hidden = [[], []]

        draft_input_embeds = self.embed_input_ids(input_ids)
        causal_mask = _build_causal_mask(attention_mask=attention_mask, dtype=projected_hidden_states.dtype)
        return self.model.layers[0](
            input_embeds=draft_input_embeds,
            hidden_states=projected_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_hidden=cache_hidden,
        )
