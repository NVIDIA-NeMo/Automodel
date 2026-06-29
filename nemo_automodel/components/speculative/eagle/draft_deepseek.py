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

"""DeepSeek (MLA-backbone) EAGLE-3 draft model.

The Llama-style draft (``draft_llama.LlamaEagle3DraftModel``) covers dense
multi-head-attention targets (Llama / Phi-3 / Qwen3). DeepSeek-V3 targets use
**Multi-head Latent Attention** (MLA): queries and keys are produced through
low-rank ``q_lora`` / ``kv_lora`` projections, the rotary part of the head is a
separate ``qk_rope_head_dim`` slice rotated with DeepSeek's *interleaved* RoPE,
and the value head dimension (``v_head_dim``) differs from the query/key one
(``qk_nope_head_dim + qk_rope_head_dim``). A Llama-shaped draft cannot represent
that, so DeepSeek targets get this dedicated draft.

This mirrors ``draft_llama`` one-to-one for everything EAGLE-3 specific (the
``[embed, hidden]`` fused first layer, the ``cache_hidden = [K_list, V_list]``
TTT recurrence with per-step rotary phase offset and diagonal-extension
attention, ``project_hidden_states`` / ``compute_logits`` / ``set_vocab_mapping``
and the ``d2t`` / ``t2d`` buffers). Only the attention block is replaced with MLA.

To guarantee the draft's rotary math matches the target's exactly, the MLA
projection layout and the interleaved RoPE are taken from the onboarded DeepSeek
target (``components/models/deepseek_v3``: the ``MLA`` projection structure and
``rope_utils``), not reimplemented.

Scope (v1): EAGLE-3 single fused draft layer, eager attention. P-EAGLE
parallel-drafting, flash-attention, and sequence packing are intentionally left
to follow-ups (they are orthogonal to the MLA attention this file adds).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from nemo_automodel.components.models.common import initialize_rms_norm_module
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb,
    freqs_cis_from_position_ids,
    precompute_freqs_cis,
)


def _build_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build a standard ``[B, 1, T, T]`` additive causal + padding mask for eager attention."""
    batch_size, seq_len = attention_mask.shape
    causal = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=attention_mask.device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    padding = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    return causal + padding


class Eagle3DeepseekMLAAttention(nn.Module):
    """MLA self-attention for the DeepSeek EAGLE-3 draft.

    Driven through the shared ``cache_hidden = [K_list, V_list]`` recurrence,
    exactly like ``Eagle3LlamaAttention``: ``step_idx = len(K_list)`` is both the
    TTT step index and the rotary phase offset; each step appends its rotated K/V
    and attends over the step-0 keys (full ``T x T`` causal) plus one diagonal
    column per later step. The only difference from the Llama attention is how
    ``q / k / v`` are produced (MLA low-rank projections + interleaved RoPE on the
    ``qk_rope_head_dim`` slice) and that the value head dim differs from the q/k
    head dim.
    """

    def __init__(self, config: PretrainedConfig, fuse_input: bool = True):
        super().__init__()
        self.config = config
        # EAGLE-3 first layer attends over the concatenated ``[embed, hidden]``
        # (2H); a plain hidden input (H) is supported for symmetry with the Llama
        # draft's deeper layers.
        in_features = config.hidden_size * 2 if fuse_input else config.hidden_size

        self.num_heads = config.num_attention_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # DeepSeek's softmax scale is the q/k head dim (yarn mscale folds into the
        # learned weights for a from-scratch draft, so it is not applied here).
        self.scaling = self.qk_head_dim**-0.5

        # Q: optional low-rank compression (q_lora_rank) then per-head projection.
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(in_features, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(in_features, self.q_lora_rank, bias=False)
            self.q_a_layernorm = initialize_rms_norm_module(
                "torch", self.q_lora_rank, eps=config.rms_norm_eps, device=None
            )
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # KV: down-project to the latent (+ a shared rope slice), then up-project.
        self.kv_a_proj_with_mqa = nn.Linear(in_features, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = initialize_rms_norm_module(
            "torch", self.kv_lora_rank, eps=config.rms_norm_eps, device=None
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

        # Interleaved-RoPE frequencies, built exactly as the DeepSeek target does
        # so the draft's rotary phase matches the target's.
        freqs = precompute_freqs_cis(
            self.qk_rope_head_dim,
            getattr(config, "max_position_embeddings", 4096),
            getattr(config, "rope_theta", 10000.0),
            getattr(config, "rope_scaling", None),
        )
        self.register_buffer("rope_freqs", freqs, persistent=False)

    def _project_qkv(
        self, combined_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q_nope, q_pe, kv_latent, k_pe)`` from the MLA down/up projections."""
        batch_size, seq_len, _ = combined_states.shape
        if self.q_lora_rank is None:
            q = self.q_proj(combined_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(combined_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(combined_states)
        kv_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = k_pe.unsqueeze(2)  # [B, S, 1, rope] -- one shared rope head, expanded after rotation
        return q_nope, q_pe, kv_latent, k_pe

    def _assemble_qkv(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, kv_latent: torch.Tensor, k_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Up-project the KV latent and assemble per-head ``q[..,qk], k[..,qk], v[..,v]`` (heads first)."""
        batch_size, seq_len = q_nope.shape[:2]
        kv = self.kv_b_proj(kv_latent).view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.expand(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)  # [B, H, S, qk_head_dim]
        k = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)  # [B, H, S, qk_head_dim]
        v = v.transpose(1, 2)  # [B, H, S, v_head_dim]
        return q, k, v

    def forward(
        self,
        combined_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_hidden: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = combined_states.shape
        q_nope, q_pe, kv_latent, k_pe = self._project_qkv(combined_states)

        cache_k, cache_v = cache_hidden
        step_idx = len(cache_k)  # TTT step == rotary phase offset
        freqs_cis = freqs_cis_from_position_ids(position_ids + step_idx, self.rope_freqs, qkv_format="bshd")
        q_pe = apply_rotary_emb(q_pe, freqs_cis, qkv_format="bshd")
        k_pe = apply_rotary_emb(k_pe, freqs_cis, qkv_format="bshd")

        q, k, v = self._assemble_qkv(q_nope, q_pe, kv_latent, k_pe)
        cache_k.append(k)
        cache_v.append(v)
        attn_output = self._eager_attention_forward(q, cache_k, cache_v, attention_mask, step_idx, batch_size, seq_len)
        return self.o_proj(attn_output)

    def _eager_attention_forward(
        self,
        q: torch.Tensor,
        cache_k: list[torch.Tensor],
        cache_v: list[torch.Tensor],
        attention_mask: torch.Tensor,
        step_idx: int,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        # Block 1: full T x T causal attention against the step-0 keys.
        k0, v0 = cache_k[0], cache_v[0]
        attn_weights = torch.matmul(q, k0.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask  # [B, 1, T, T] additive

        # Block 2: one diagonal column per cached later step (Q_t attends to K_i_t).
        if step_idx >= 1:
            later_k = torch.stack(cache_k[1:], dim=0)  # [step_idx, B, H, T, qk]
            diag = torch.einsum("bhtd,sbhtd->bhts", q, later_k) * self.scaling
            attn_weights = torch.cat((attn_weights, diag), dim=-1)

        # Block 3: softmax over the extended T + step_idx key axis.
        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)

        # Block 4: weighted sum over V_0 plus the diagonal later-step values.
        attn_output = torch.matmul(attn_probs[..., :seq_len], v0)
        if step_idx >= 1:
            later_v = torch.stack(cache_v[1:], dim=0)  # [step_idx, B, H, T, v]
            attn_output = attn_output + torch.einsum("bhts,sbhtd->bhtd", attn_probs[..., seq_len:], later_v)
        return attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)


class Eagle3DeepseekMLP(nn.Module):
    """Plain SwiGLU MLP for the draft (the draft is small; it does not replicate the target MoE)."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Eagle3DeepseekDecoderLayer(nn.Module):
    """Fused EAGLE-3 first layer: ``[embed, hidden]`` -> MLA -> MLP (mirrors Eagle3LlamaDecoderLayer)."""

    def __init__(self, config: PretrainedConfig, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.hidden_norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, device=None)
        self.post_attention_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.self_attn = Eagle3DeepseekMLAAttention(config)
        self.mlp = Eagle3DeepseekMLP(config)

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
        hidden_states = residual + self.self_attn(combined_states, attention_mask, position_ids, cache_hidden)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Eagle3DeepseekModel(nn.Module):
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
                    initialize_rms_norm_module("torch", target_hidden_size, eps=config.rms_norm_eps, device=None)
                    for _ in range(num_aux_hidden_states)
                ]
            )
        self.layers = nn.ModuleList([Eagle3DeepseekDecoderLayer(config, layer_id=0)])
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps, device=None)


class DeepseekV3Eagle3DraftModel(PreTrainedModel):
    """DeepSeek-V3 (MLA) EAGLE-3 draft model.

    The MLA counterpart to ``LlamaEagle3DraftModel``: same public training API
    (``project_hidden_states`` / ``embed_input_ids`` / ``compute_logits`` /
    ``set_vocab_mapping`` / ``forward``) and the same ``d2t`` / ``t2d`` vocab-remap
    buffers, so the EAGLE-3 trainer and checkpointing are reused unchanged.
    """

    config_class = PretrainedConfig
    base_model_prefix = "model"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)

        self.model = Eagle3DeepseekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        # d2t/t2d only when the draft vocab is compressed (matches draft_llama).
        self.has_vocab_compression = self.draft_vocab_size < config.vocab_size
        if self.has_vocab_compression:
            self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long), persistent=True)
            self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool), persistent=True)

        self.post_init()

    def copy_embeddings_from_target(self, target_embedding: nn.Embedding) -> None:
        """Seed the draft embedding table from the (possibly FSDP-sharded) target embeddings."""
        target_weight = target_embedding.weight
        if hasattr(target_weight, "full_tensor"):
            target_weight = target_weight.full_tensor()
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(target_weight.to(self.model.embed_tokens.weight.dtype))

    def set_vocab_mapping(self, selected_token_ids: torch.Tensor) -> None:
        """Populate ``d2t`` / ``t2d`` from the draft->target id map (offset form vLLM/SGLang expect)."""
        if not self.has_vocab_compression:
            return
        selected = selected_token_ids.reshape(-1).to(dtype=torch.long, device=self.d2t.device)
        if selected.numel() != self.draft_vocab_size:
            raise ValueError(
                "set_vocab_mapping expected selected_token_ids of length "
                f"draft_vocab_size={self.draft_vocab_size}, got {selected.numel()}."
            )
        draft_ids = torch.arange(self.draft_vocab_size, device=self.d2t.device)
        with torch.no_grad():
            self.d2t.copy_(selected - draft_ids)
            self.t2d.zero_()
            self.t2d[selected] = True

    def project_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated target aux hidden states to draft hidden size (with optional fc_norm)."""
        if getattr(self.config, "fc_norm", False):
            num_aux = len(self.model.fc_norm)
            chunks = torch.chunk(aux_hidden_states, num_aux, dim=-1)
            aux_hidden_states = torch.cat([norm(chunk) for norm, chunk in zip(self.model.fc_norm, chunks)], dim=-1)
        return self.model.fc(aux_hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids with the draft embedding table."""
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Draft logits over the draft vocabulary (applies ``model.norm`` unless already normed)."""
        if getattr(self.config, "norm_output", False):
            return self.lm_head(hidden_states)
        return self.lm_head(self.model.norm(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        projected_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache_hidden: Optional[list[list[torch.Tensor]]] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one EAGLE-3 TTT draft step (eager attention; ``seq_lens`` packing not supported in v1)."""
        if seq_lens is not None:
            raise NotImplementedError(
                "DeepseekV3Eagle3DraftModel does not support sequence packing (seq_lens) yet; use the unpacked path."
            )
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(input_ids.shape[0], -1)
        if cache_hidden is None:
            cache_hidden = [[], []]

        causal_mask = _build_causal_mask(attention_mask=attention_mask, dtype=projected_hidden_states.dtype)
        draft_input_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.model.layers[0](
            input_embeds=draft_input_embeds,
            hidden_states=projected_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_hidden=cache_hidden,
        )
        if getattr(self.config, "norm_output", False):
            hidden_states = self.model.norm(hidden_states)
        return hidden_states
