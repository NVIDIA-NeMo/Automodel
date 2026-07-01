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
"""DSpark speculative-decoding draft model with a MiniMax M3 attention backbone.

Mirrors the Qwen3 DSpark draft's control flow (standard GQA: separate q/k/v/o
projections, per-head norm on Q/K, K/V built by projecting the context and the
noise block separately then concatenating). The two differences from Qwen3 are
MiniMax M3's own building blocks: per-head Gemma RMSNorm (``MiniMaxM3RMSNorm``),
SwiGLU-OAI activation (``swiglu_oai``), and partial RoPE via the gpt_oss rotary
utilities (matching the target's own ``MiniMaxM3Attention``/``MiniMaxM3TextModel``).

The draft runs dense, non-causal attention over a ``[context | noise-block]``
layout, with visibility supplied entirely by the DFlash additive/block
attention mask. There is no block-sparse indexer and no MoE here: M3's sparse
attention and routed experts belong to the target model, not this draft (the
draft is always dense, matching the DeepSeek V4 DSpark draft's convention).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from nemo_automodel.components.attention.dflash_mask import create_dflash_block_mask, create_dflash_sdpa_mask
from nemo_automodel.components.models.common import get_rope_config
from nemo_automodel.components.models.gpt_oss.rope_utils import (
    RotaryEmbedding,
    apply_rotary_emb,
    position_ids_to_freqs_cis,
)
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3RMSNorm, swiglu_oai
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


class MiniMaxM3DSparkMLP(nn.Module):
    """Dense SwiGLU-OAI MLP for the DSpark draft (plain ``nn.Linear``, no MoE)."""

    def __init__(self, config):
        super().__init__()
        self.alpha = float(getattr(config, "swiglu_alpha", 1.702))
        self.limit = float(getattr(config, "swiglu_limit", 7.0))
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_oai(self.gate_proj(x), self.up_proj(x), self.alpha, self.limit))


class MiniMaxM3DSparkAttention(nn.Module):
    """Dense, non-causal MiniMax M3 GQA attention over a ``[context | noise-block]`` layout.

    Standard GQA (separate q/k/v/o projections, per-head Gemma RMSNorm on Q and
    K before RoPE), mirroring ``Qwen3DSparkAttention``'s control flow. M3 has no
    shared K=V latent (unlike DeepSeek V4's MLA draft), so K and V are genuinely
    separate projections; there is no block-sparse indexer or MoE here (target-only
    machinery, not the draft's concern).

    RoPE follows the gpt_oss rotary utilities' broadcasting convention (``x`` in
    ``[B, S, H, D]`` layout while cos/sin apply), matching the target's own
    ``MiniMaxM3Attention``, so RoPE is applied to Q/K *before* the transpose to
    ``[B, H, S, D]`` for attention -- not after, unlike the Gemma4/Qwen3 drafts'
    own RoPE helpers, which use the opposite (post-transpose) convention.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5

        gemma = bool(getattr(config, "use_gemma_norm", True))
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps, gemma=gemma)
        self.k_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps, gemma=gemma)

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return hidden_states
        return hidden_states.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]

        # Queries come from the noise block only; per-head norm before RoPE,
        # still in [B, S, H, D] layout (gpt_oss apply_rotary_emb's convention).
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        # K/V span [context | noise]. wkv is per-token, so projecting context and
        # noise separately then concatenating is equivalent to (and cheaper than)
        # projecting the concatenation once per matrix, matching Qwen3DSparkAttention.
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k)

        # Partial RoPE. cos/sin span the full [context | draft] positions, so the
        # draft queries take the suffix slice while K takes the full span.
        q = apply_rotary_emb(q, cos[:, -q_len:], sin[:, -q_len:])
        k = apply_rotary_emb(k, cos, sin)

        # Only now transpose to [B, H, S, D] for attention (V never gets RoPE).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if self.config._attn_implementation == "flex_attention" and attention_mask is not None:
            attn_output = flex_attention(q, k, v, block_mask=attention_mask, scale=self.scaling)
        else:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class MiniMaxM3DSparkDecoderLayer(nn.Module):
    """Pre-norm residual block: MiniMax M3 DSpark attention followed by a dense SwiGLU-OAI MLP."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniMaxM3DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = MiniMaxM3DSparkMLP(config)
        gemma = bool(getattr(config, "use_gemma_norm", True))
        self.input_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)
        self.post_attention_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        assert hidden_states is not None, "hidden_states must be provided."
        assert target_hidden_states is not None, "target_hidden_states must be provided."
        assert cos is not None and sin is not None, "cos/sin position embeddings must be provided."
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniMaxM3DSparkModel(nn.Module):
    """DSpark draft model with a MiniMax M3 attention backbone (dense, non-causal)."""

    _no_split_modules = ["MiniMaxM3DSparkDecoderLayer"]

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
            [MiniMaxM3DSparkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        gemma = bool(getattr(config, "use_gemma_norm", True))
        self.norm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

        self.head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
        # Mirrors MiniMaxM3TextModel.__init__: the released config carries
        # rope_theta/rotary_dim as flat fields, not the rope_parameters dict
        # get_rope_config expects, so derive it lazily the same way the target does.
        if not hasattr(config, "rope_parameters") or config.rope_parameters is None:
            rotary_dim = getattr(config, "rotary_dim", self.head_dim)
            config.rope_parameters = {
                "rope_theta": getattr(config, "rope_theta", 5000000.0),
                "rope_type": "default",
                "partial_rotary_factor": rotary_dim / self.head_dim,
            }
        base, rope_scaling, partial_rotary_factor = get_rope_config(config)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            partial_rotary_factor=partial_rotary_factor,
            device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
        )

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        # Markov head.
        self.markov_head = build_markov_head(config)

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = self.enable_confidence_head and bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    def _apply(self, fn, recurse: bool = True):
        """Keep the rotary module's device in sync across ``.to()`` calls.

        ``RotaryEmbedding`` caches its concentration/``inv_freq`` via
        ``functools.cache`` keyed on the module instance, not a persistent
        buffer, so ``nn.Module._apply`` (which only visits parameters/buffers)
        never touches it. A later ``.to(device=...)`` (the training build path)
        would otherwise silently leave the cached frequencies on whatever
        device was current at construction, causing a device mismatch the
        first time they are used against q/k on the real device. Sync
        ``.device`` and drop the cache after any move so the next call
        recomputes on the right device.
        """
        module = super()._apply(fn, recurse=recurse)
        rotary_emb = getattr(self, "rotary_emb", None)
        if rotary_emb is not None:
            try:
                target_device = next(self.parameters()).device
            except StopIteration:
                target_device = None
            if target_device is not None and rotary_emb.device != target_device:
                rotary_emb.device = target_device
                rotary_emb._compute_concentration_and_inv_freq.cache_clear()
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
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        freqs_cis = position_ids_to_freqs_cis(self.rotary_emb, position_ids, qkv_format="bshd", for_fused_rope=False)
        cos, sin = freqs_cis.split(freqs_cis.shape[-1] // 2, dim=-1)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
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
        if self.config._attn_implementation == "flex_attention":
            dspark_attn_mask = create_dflash_block_mask(
                anchor_positions, block_keep_mask, seq_len, self.block_size, device
            )
        else:
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
    "MiniMaxM3DSparkModel",
    "MiniMaxM3DSparkAttention",
    "MiniMaxM3DSparkDecoderLayer",
]
