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
"""DSpark speculative-decoding draft model with a GLM-5.2 attention backbone.

This mirrors the DeepSeek V4 DSpark draft (``draft_deepseek_v4.py``) structurally;
only the attention internals and rotary plumbing are swapped for GLM-5.2's
Multi-head Latent Attention (MLA). The draft runs dense, non-causal attention over
a ``[context | noise-block]`` layout, with visibility supplied entirely by the
DFlash additive attention mask.

The GLM MLA is the classic DeepSeek-V3 layout (``q_a_proj``/``q_a_layernorm``/
``q_b_proj``, a compressed ``kv_a_proj_with_mqa``/``kv_a_layernorm``/``kv_b_proj``
latent, a separate ``o_proj``, and interleaved complex ``freqs_cis`` RoPE over the
``qk_rope_head_dim`` slice), NOT V4's single shared K=V latent + grouped O-LoRA +
per-head sink. There is no DSA indexer / sparse-attention path here: the draft is
always dense, and GLM's sparse top-k machinery belongs to the target model, not
this draft.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from nemo_automodel.components.attention.dflash_mask import create_dflash_sdpa_mask
from nemo_automodel.components.models.common import initialize_rms_norm_module
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb,
    freqs_cis_from_position_ids,
    precompute_freqs_cis,
)
from nemo_automodel.components.speculative.dspark._sampling import sample_tokens
from nemo_automodel.components.speculative.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)
from nemo_automodel.components.speculative.dspark.markov_head import build_markov_head


def _glm_5_2_eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
) -> torch.Tensor:
    """Dense eager attention over an additive mask (no GQA repeat, no sink).

    GLM MLA keeps one K/V per query head (``num_key_value_heads == num_attention_heads``),
    so there is no group repeat. The additive ``attention_mask`` uses ``-inf`` for masked
    positions (the DFlash SDPA mask); a plain matmul + fp32 softmax handles it correctly
    (the DFlash mask guarantees no fully-masked query row, so no row softmaxes to NaN).

    Args:
        query: ``[B, H, q_len, qk_head_dim]``.
        key:   ``[B, H, kv_len, qk_head_dim]``.
        value: ``[B, H, kv_len, v_head_dim]``.
        attention_mask: ``[B, 1, q_len, kv_len]`` additive mask, or ``None``.
        scaling: softmax scale (``qk_head_dim ** -0.5``).

    Returns:
        ``[B, q_len, H, v_head_dim]`` (heads folded back next to the feature dim).
    """
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
    return torch.matmul(attn_weights, value).transpose(1, 2).contiguous()


class Glm5_2DSparkMLP(nn.Module):
    """Dense SwiGLU MLP for the DSpark draft.

    This intentionally avoids GLM's MoE and shared-expert paths; the draft keeps a
    single dense feed-forward block sized to ``moe_intermediate_size`` (the per-expert
    width), so the drafter stays lightweight relative to the 256-expert target.
    """

    def __init__(self, config):
        super().__init__()
        intermediate_size = int(config.moe_intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Glm5_2DSparkAttention(nn.Module):
    """Dense, non-causal GLM MLA over a ``[context | noise-block]`` layout.

    Reuses GLM-5.2's DeepSeek-V3-style MLA projections (Q-LoRA, a compressed KV latent,
    a separate value up-projection, interleaved complex RoPE on the ``qk_rope_head_dim``
    slice) but drops the DSA indexer (the draft is always dense). Queries come from the
    noise block only, while the compressed KV latent spans context plus noise. Visibility
    is set entirely by the DFlash additive ``attention_mask`` (no causal flag, no indexer).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_heads = int(config.num_attention_heads)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(config.v_head_dim)
        self.scaling = self.qk_head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = initialize_rms_norm_module("torch_fp32", self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = initialize_rms_norm_module("torch_fp32", self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        del kwargs
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        kv_len = ctx_len + q_len

        # Queries come from the noise block only.
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # One compressed KV latent over the concatenated context and noise tokens.
        # kv_a_proj_with_mqa and kv_a_layernorm are per-token, so projecting the
        # concatenation once is equivalent to projecting each part separately.
        kv_input = torch.cat([target_hidden_states, hidden_states], dim=1)
        compressed_kv = self.kv_a_proj_with_mqa(kv_input)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)

        # Interleaved complex RoPE. freqs_cis spans the full [context | draft] positions,
        # so the draft queries take the suffix slice while the shared latent's rope part
        # (k_pe, one head broadcast to all) takes the full span. k_pe keeps its singleton
        # head dim from the rotary so it can be broadcast to every head with one expand.
        q_pe = apply_rotary_emb(q_pe, freqs_cis[:, -q_len:], qkv_format="bshd")
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, qkv_format="bshd")

        kv = self.kv_b_proj(compressed_kv).view(bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.expand(bsz, kv_len, self.num_heads, self.qk_rope_head_dim)

        query = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        key = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = _glm_5_2_eager_attention(query, key, value, attention_mask, scaling=self.scaling)
        return self.o_proj(attn_output.reshape(bsz, q_len, -1)), None


class Glm5_2DSparkDecoderLayer(nn.Module):
    """Pre-norm residual block: GLM DSpark MLA followed by a dense SwiGLU MLP."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Glm5_2DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = Glm5_2DSparkMLP(config)
        self.input_layernorm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[object] = None,
        use_cache: Optional[bool] = False,
        freqs_cis: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, past_key_value, use_cache, kwargs
        assert hidden_states is not None, "hidden_states must be provided."
        assert target_hidden_states is not None, "target_hidden_states must be provided."
        assert freqs_cis is not None, "freqs_cis must be provided."
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Glm5_2DSparkModel(nn.Module):
    """DSpark draft model with a GLM-5.2 attention backbone (dense, non-causal)."""

    _no_split_modules = ["Glm5_2DSparkDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), "config.markov_head_type must be provided when markov_rank > 0."
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when enable_confidence_head is true."
            )
        self.target_layer_ids = config.target_layer_ids

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [Glm5_2DSparkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)

        # GLM reuses DeepSeek-V3's rotary: precompute the fp32 frequency table over the
        # qk_rope_head_dim slice. rope_theta lives under config.rope_parameters (the HF
        # GlmMoeDsaConfig has no top-level rope_theta); rope_scaling=None matches the
        # released GLM-5.2 "default" rope_type.
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = float(rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
        freqs = precompute_freqs_cis(
            qk_rope_head_dim=int(config.qk_rope_head_dim),
            max_seq_len=int(config.max_position_embeddings),
            rope_theta=rope_theta,
            rope_scaling=None,
        )
        self.register_buffer("freqs", freqs.to(torch.float32), persistent=False)

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = initialize_rms_norm_module("torch", config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        # Markov head.
        self.markov_head = build_markov_head(config)

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

        # The RMSNorm factory may build norms in bf16 while the Linear / Embedding layers
        # are fp32. Unify to fp32 so the freshly built draft is single-dtype (matching the
        # other DSpark drafts); FSDP2 fully_shard requires a uniform original parameter
        # dtype before any later cast to the training compute dtype.
        self.to(torch.float32)

    def _apply(self, fn, recurse=True):
        """Keep the rotary ``freqs`` buffer in fp32 across dtype casts.

        ``model.to(bfloat16)`` (the training build path) would otherwise round the RoPE
        ``freqs`` table to bf16 and dephase RoPE with absolute position, eroding draft
        acceptance (the mismatch grows with position, and a bf16 round-trip cannot be
        undone by upcasting). Snapshot the fp32 frequencies before the cast and restore
        them after (the Fp32Safe rotary idiom used elsewhere in the repo), so the buffer
        never makes a bf16 round-trip.
        """
        freqs = getattr(self, "freqs", None)
        freqs_fp32 = (
            freqs.detach().clone().to(torch.float32)
            if freqs is not None and freqs.is_floating_point() and not freqs.is_meta
            else None
        )
        module = super()._apply(fn, recurse=recurse)
        if freqs_fp32 is not None:
            self.register_buffer("freqs", freqs_fp32.to(device=self.freqs.device), persistent=False)
        return module

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(dtype=hidden_states.dtype)
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2, (
            f"sample_draft_token_step expects base_logits shaped [batch, vocab], got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        # Complex RoPE table over the full [context | draft] positions; each MLA layer
        # slices the draft suffix for Q and uses the full span for the KV latent.
        freqs_cis = freqs_cis_from_position_ids(position_ids, self.freqs, qkv_format="bshd")
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                freqs_cis=freqs_cis,
                **kwargs,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        # The draft attention is always dense eager attention, so it always uses the
        # DFlash dense additive (SDPA) mask; there is no flex/eager kernel axis here.
        dspark_attn_mask = create_dflash_sdpa_mask(
            anchor_positions, block_keep_mask, seq_len, self.block_size, device, noise_embedding.dtype
        )
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(dtype=output_hidden_4d.dtype)
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = [
    "Glm5_2DSparkModel",
    "Glm5_2DSparkAttention",
    "Glm5_2DSparkDecoderLayer",
]
