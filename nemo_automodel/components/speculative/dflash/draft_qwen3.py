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

"""DFlash draft model (Qwen3-style).

Ported from SpecForge's ``specforge/modeling/draft/dflash.py``. DFlash drafts a
whole block of ``block_size`` tokens in parallel: the block's first position
holds the real anchor token and the rest are ``MASK`` tokens, and the draft
predicts the whole block in a single non-causal forward conditioned on the
target model's context hidden states.

The draft attention is therefore **not causal** -- a draft block's queries
attend to (a) the projected target-hidden context strictly before its anchor and
(b) bidirectionally to the other (noise) tokens of the same block. The attention
mask that enforces this is built by the trainer wrapper in
``nemo_automodel.components.speculative.dflash.core``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Greedy (temperature ~ 0) or temperature sampling over the last dim."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE where queries (draft block) are a suffix of the key positions.

    The keys span ``[context | noise-block]`` while the queries are only the
    noise block, so ``q`` is rotated with the trailing ``q_len`` slice of the
    rotary tables and ``k`` with the full table.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Non-causal attention whose keys/values are ``[context | noise-block]``.

    Queries come from the draft (noise) tokens only; keys and values are the
    concatenation of the projected target-hidden context and the noise tokens.
    The bidirectional/block structure is supplied entirely by ``attention_mask``.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    """A DFlash decoder block: non-causal attention over ``[context | noise]`` + MLP."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Pick ``num_draft_layers`` target layers spread across the target's depth."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def extract_context_feature(hidden_states: list[torch.Tensor], layer_ids: list[int]) -> torch.Tensor:
    """Concatenate the selected target layers' hidden states along the feature dim.

    ``hidden_states`` follows HF's ``output_hidden_states`` convention where
    index 0 is the embedding output, so layer ``i``'s output is at index
    ``i + 1``.
    """
    offset = 1
    return torch.cat([hidden_states[layer_id + offset] for layer_id in layer_ids], dim=-1)


class Qwen3DFlashDraftModel(Qwen3PreTrainedModel):
    """DFlash draft model: a small non-causal Qwen3 stack over ``[context | noise]``."""

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        # Optional Domino correction head (ported from SpecForge#571). DFlash drafts
        # a block in parallel and is non-causal; the Domino head adds a *causal*
        # low-rank logit correction conditioned on a GRU state built from the
        # block's previous tokens. ``projector_type=None`` leaves DFlash untouched.
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        if self.projector_type == "domino":
            self.emb_dim = dflash_config["emb_dim"]
            self.gru_hidden_dim = dflash_config["gru_hidden_dim"]
            self.prefix_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            in_dim = config.hidden_size + self.gru_hidden_dim
            self.embed_proj = nn.Sequential(
                nn.Linear(in_dim, self.emb_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.emb_dim, config.vocab_size, bias=False),
            )
        elif self.projector_type is not None:
            raise ValueError(f"Unknown draft projector_type: {self.projector_type}")
        self.post_init()

    def _apply(self, fn, recurse=True):
        """Keep the RoPE ``inv_freq`` buffer in fp32 across dtype casts.

        ``Qwen3RotaryEmbedding`` computes the rotary angles in fp32 but reads the
        frequencies from a stored ``inv_freq`` buffer. ``model.to(bfloat16)`` -- the
        training build path -- rounds that buffer to bf16, whereas the serving
        runtime (SGLang keeps an fp32 RoPE cache) and HF's ``from_pretrained`` reload
        keep it in fp32. The resulting train/inference RoPE mismatch grows with
        absolute position (the bf16 frequencies dephase) and erodes draft
        acceptance, so ``inv_freq`` must stay fp32 on both the training and reload
        paths. A bf16 round-trip cannot be undone by upcasting, so when a cast
        rounds the buffer we recompute fresh fp32 frequencies from the rotary
        config (the same values HF derives on the fp32 paths) instead of upcasting
        the corrupted ones.
        """
        module = super()._apply(fn, recurse=recurse)
        rotary_emb = getattr(self, "rotary_emb", None)
        inv_freq = getattr(rotary_emb, "inv_freq", None) if rotary_emb is not None else None
        if (
            inv_freq is not None
            and inv_freq.is_floating_point()
            and not inv_freq.is_meta
            and inv_freq.dtype != torch.float32
        ):
            fresh = type(rotary_emb)(rotary_emb.config).inv_freq.to(device=inv_freq.device)
            rotary_emb.inv_freq = fresh
            if hasattr(rotary_emb, "original_inv_freq"):
                rotary_emb.original_inv_freq = fresh.clone()
        return module

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: Optional[list[int]],
        temperature: float,
    ) -> torch.LongTensor:
        """Block-parallel speculative decoding: draft a block, verify with the target, accept the matching prefix."""
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size

        output_ids = torch.full(
            (1, max_length + block_size), self.mask_token_id, dtype=torch.long, device=target.device
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill the target on the prompt.
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        start = num_input_tokens
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]
            if stop_token_ids is not None and any(
                stop_id in output_ids[:, num_input_tokens:] for stop_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_ids).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]
        return output_ids
