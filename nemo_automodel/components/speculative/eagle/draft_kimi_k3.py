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

"""Kimi K3 (NoPE MLA backbone) EAGLE-3 draft model.

Kimi K3 interleaves Kimi Delta Attention (KDA) layers with Multi-head Latent
Attention (MLA) layers. The draft mirrors only the MLA layer, which differs from
the DeepSeek MLA the shared MLA draft (``draft_deepseek``) implements in three
ways:

* **NoPE.** K3 fixes ``mla_use_nope=True``: the ``qk_rope_head_dim`` slice is
  concatenated to the query and key heads unrotated, and no rotary embedding is
  applied anywhere in the MLA layer (positional information reaches the full
  attention layers through the interleaved KDA recurrence). The draft keeps that
  contract, so it registers no rotary cache and applies no per-TTT-step rotary
  phase offset.
* **Output gate.** ``mla_use_output_gate=True`` multiplies the attention output
  by a sigmoid gate projected from the layer input.
* **SiTU MLP.** The feed-forward block is K3's ``SituAndMul`` gated MLP
  (``KimiK3MLP``), not SwiGLU, and every norm is K3's fp32-variance
  ``KimiRMSNorm``.

Everything EAGLE-3 specific is inherited from the DeepSeek MLA draft: the
``[embed, hidden]`` fused first layer, the ``cache_hidden = [K_list, V_list]``
TTT recurrence with diagonal-extension attention, ``project_hidden_states`` /
``compute_logits`` / ``set_vocab_mapping`` and the ``d2t`` / ``t2d`` buffers. The
SiTU MLP and the norm are imported from the onboarded K3 target
(``components/models/kimi_k3.model``) and the MLA projection layout and the gate
follow its ``KimiMLAAttention`` (widened to the fused ``[embed, hidden]`` input),
so the draft's math matches the target's MLA.

Scope: EAGLE-3 single fused draft layer, eager attention, dense draft (KDA,
routed experts and the attention-residual mixer stay in the target only). The
draft itself consumes the shared ``[B, 1, T, T]`` block-causal packed mask like
the DeepSeek MLA draft, but the recipe rejects sequence packing for a K3 target:
K3 owns its own packed-attention path, which the EAGLE-3 target wrapper does not
drive (see ``examples/speculative/eagle3/README_kimi_k3.md``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from nemo_automodel.components.models.kimi_k3.model import KimiK3MLP, KimiRMSNorm
from nemo_automodel.components.speculative.eagle.draft_deepseek import (
    DeepseekV3Eagle3DraftModel,
    Eagle3DeepseekMLAAttention,
)

# K3's norm and MLP modules default to bf16 while torch's do not; build the whole draft
# in fp32 (torch's default) and let the recipe cast it to the target's compute dtype.
_INIT_DTYPE = torch.float32


class KimiK3Eagle3MLAAttention(Eagle3DeepseekMLAAttention):
    """NoPE MLA self-attention with an output gate for the Kimi K3 EAGLE-3 draft.

    Reuses the DeepSeek draft's TTT recurrence verbatim (``_eager_attention_forward``:
    full ``T x T`` causal attention against the step-0 keys plus one diagonal column
    per cached later step). Only the projections change: no rotary cache is built,
    the rope slice rides along unrotated, and the attention output is gated. The
    parent ``__init__`` is bypassed deliberately -- it registers a rotary buffer
    that a NoPE layer must not carry.
    """

    def __init__(self, config: PretrainedConfig, fuse_input: bool = True):
        nn.Module.__init__(self)
        self.config = config
        # EAGLE-3's first layer attends over the concatenated ``[embed, hidden]``
        # (2H); a plain hidden input (H) is kept for symmetry with the sibling drafts.
        in_features = config.hidden_size * 2 if fuse_input else config.hidden_size

        self.num_heads = config.num_attention_heads
        if config.num_key_value_heads != self.num_heads:
            raise ValueError(
                "Kimi K3 MLA expects one K/V head per query head, got "
                f"num_key_value_heads={config.num_key_value_heads} vs num_attention_heads={self.num_heads}."
            )
        if not config.mla_use_nope:
            raise ValueError("Kimi K3 EAGLE-3 draft requires mla_use_nope=True.")
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(in_features, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(in_features, self.q_lora_rank, bias=False)
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps, dtype=_INIT_DTYPE)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(in_features, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=_INIT_DTYPE)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(in_features, self.num_heads * self.v_head_dim, bias=False)

    def forward(
        self,
        combined_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        # ``position_ids`` is unused: K3's MLA is NoPE, so the TTT step index only
        # extends the key axis (via cache_hidden) and never shifts a rotary phase.
        del position_ids
        batch_size, seq_len, _ = combined_states.shape

        if self.q_lora_rank is None:
            query_states = self.q_proj(combined_states)
        else:
            query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(combined_states)))
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(combined_states)
        kv_latent, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        key_states, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # One shared unrotated rope slice per token, broadcast over the query heads.
        k_rot = k_rot.view(batch_size, 1, seq_len, self.qk_rope_head_dim).expand(*key_states.shape[:-1], -1)
        key_states = torch.cat([key_states, k_rot], dim=-1)

        cache_k, cache_v = cache_hidden
        step_idx = len(cache_k)
        cache_k.append(key_states)
        cache_v.append(value_states)
        attn_output = self._eager_attention_forward(
            query_states, cache_k, cache_v, attention_mask, step_idx, batch_size, seq_len
        )
        if self.use_output_gate:
            attn_output = attn_output * self.g_proj(combined_states).sigmoid()
        return self.o_proj(attn_output)


class KimiK3Eagle3DecoderLayer(nn.Module):
    """Fused EAGLE-3 first layer: ``[embed, hidden]`` -> NoPE MLA -> SiTU MLP."""

    def __init__(self, config: PretrainedConfig, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=_INIT_DTYPE)
        self.hidden_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=_INIT_DTYPE)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=_INIT_DTYPE)
        self.self_attn = KimiK3Eagle3MLAAttention(config)
        self.mlp = KimiK3MLP(config, dtype=_INIT_DTYPE)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        residual = hidden_states
        combined_states = torch.cat((self.input_layernorm(input_embeds), self.hidden_norm(hidden_states)), dim=-1)
        hidden_states = residual + self.self_attn(combined_states, attention_mask, position_ids, cache_hidden)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Eagle3KimiK3Model(nn.Module):
    """Inner backbone: ``embed_tokens``, the ``fc`` aux-projection, the fused draft layer, and ``norm``."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        num_aux_hidden_states = getattr(config, "num_aux_hidden_states", 3)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.fc = nn.Linear(target_hidden_size * num_aux_hidden_states, config.hidden_size, bias=False)
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    KimiRMSNorm(target_hidden_size, eps=config.rms_norm_eps, dtype=_INIT_DTYPE)
                    for _ in range(num_aux_hidden_states)
                ]
            )
        self.layers = nn.ModuleList([KimiK3Eagle3DecoderLayer(config, layer_id=0)])
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=_INIT_DTYPE)


class KimiK3Eagle3DraftModel(DeepseekV3Eagle3DraftModel):
    """Kimi K3 (NoPE MLA) EAGLE-3 draft model.

    Same public training API as every other EAGLE-3 draft
    (``project_hidden_states`` / ``embed_input_ids`` / ``compute_logits`` /
    ``set_vocab_mapping`` / ``freeze_embeddings`` / ``forward``) and the same
    ``d2t`` / ``t2d`` vocab-remap buffers, so the EAGLE-3 trainer and
    checkpointing are reused unchanged; only the backbone is K3's.
    """

    def _build_inner_model(self, config: PretrainedConfig) -> nn.Module:
        return Eagle3KimiK3Model(config)
