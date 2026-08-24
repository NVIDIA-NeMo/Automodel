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

from typing import Callable, Tuple

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

from nemo_automodel.components.speculative.dflash.target import resolve_text_config


def resolve_output_head(target: nn.Module) -> nn.Module:
    """Return the target's output projection.

    Prefers ``lm_head`` and falls back to ``get_output_embeddings()``, which is
    what a ``*ForConditionalGeneration`` target exposes instead.

    Args:
        target: The frozen verifier.

    Returns:
        The module projecting hidden states to vocabulary logits.
    """
    head = getattr(target, "lm_head", None)
    return head if head is not None else target.get_output_embeddings()


def validate_sampling(temperature: float, top_p: float = 1.0, top_k: int = 0) -> None:
    """Reject sampling parameters the decode path cannot honor.

    Args:
        temperature: Sampling temperature; ``0`` means greedy.
        top_p: Nucleus mass to keep, in ``(0, 1]``.
        top_k: Candidates to keep, ``0`` for the whole vocabulary.

    Raises:
        ValueError: If any parameter is outside its supported range.
    """
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError(
            f"Invalid sampling parameters: temperature={temperature} must be >= 0, "
            f"top_p={top_p} must be in (0, 1], top_k={top_k} must be >= 0."
        )


def sampling_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Full-vocabulary sampling distribution after temperature, top-k, and top-p.

    Speculative decoding needs the *distribution*, not just a draw: rejection
    sampling compares the target's probability of a drafted token against the
    draft's, so both sides must be built the same way. Filtered-out tokens keep a
    probability of exactly 0 so the residual ``clamp(p - q, 0)`` stays correct.

    Args:
        logits: Tensor of shape [..., vocab]; raw scores, with arbitrary leading
            dimensions.
        temperature: Sampling temperature; must be > 0.
        top_p: Nucleus mass to keep, in ``(0, 1]``.
        top_k: Candidates to keep, ``0`` for the whole vocabulary.

    Returns:
        Tensor of shape [..., vocab] in float32; a probability distribution over
        the last dimension that is zero outside the kept set.
    """
    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    indices = None
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)

    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        # Exclusive cumulative mass: always keeps the top token, even when it
        # already exceeds top_p on its own.
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Greedy (``temperature == 0``) or truncated-distribution sampling over the last dim.

    Args:
        logits: Tensor of shape [..., vocab]; raw scores, with arbitrary leading
            dimensions.
        temperature: Sampling temperature; ``0`` selects greedily.
        top_p: Nucleus mass to keep, in ``(0, 1]``.
        top_k: Candidates to keep, ``0`` for the whole vocabulary.

    Returns:
        Long tensor of shape [...]; the sampled token id per position.
    """
    validate_sampling(temperature, top_p, top_k)
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    probs = sampling_probs(logits, temperature, top_p, top_k)
    shape = probs.shape[:-1]
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).view(shape)


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


# Layer kinds whose cache a rejected block can be rewound out of by dropping KV
# rows. Anything else -- notably the ``linear_attention`` layers the Qwen3.5
# family interleaves -- carries a recurrent state that cropping cannot rewind.
_REWINDABLE_LAYER_TYPES = frozenset({"full_attention", "sliding_attention", "chunked_attention"})


def assert_target_supports_rollback(target: nn.Module) -> None:
    """Reject targets whose cache cannot be rewound after a rejected block.

    Speculative decoding verifies a whole block and then rewinds the target to
    the accepted prefix. For attention layers that is just dropping KV rows, but
    a linear-attention layer keeps a recurrent state that has already absorbed
    the rejected tokens; ``Cache.crop`` truncates the KV entries and leaves that
    state where it is. The target then predicts from a corrupted state, so the
    "lossless" guarantee quietly stops holding -- greedy decoding drifts away
    from the target's own output instead of reproducing it.

    Training is unaffected: it is a single forward with no cache and no rewind.

    Args:
        target: The frozen verifier.

    Raises:
        ValueError: If the target has any layer whose state cropping cannot
            rewind.
    """
    config = getattr(target, "config", None)
    if config is None:
        return
    text_config = resolve_text_config(config)
    layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
        return
    unsupported = sorted(set(layer_types) - _REWINDABLE_LAYER_TYPES)
    if unsupported:
        raise ValueError(
            f"Speculative decoding cannot verify against this target: its {', '.join(unsupported)} layers keep a "
            "recurrent state that cropping the KV cache does not rewind, so a rejected block permanently corrupts "
            "the target and the decoded output stops matching the target's own. Training against such a target is "
            "fine (a single forward, no cache); serve the trained draft with an engine that rewinds hybrid state."
        )


def _sliding_window_mask(query: torch.Tensor, key: torch.Tensor, sliding_window: int) -> torch.Tensor:
    """Band mask keeping keys within ``sliding_window`` of each query's position.

    The draft's queries are the trailing ``q_len`` positions of the concatenated
    ``[context | noise-block]`` key axis, so a query at row ``i`` sits at key
    position ``k_len - q_len + i``. Both bounds are strict, matching transformers'
    ``sliding_window_overlay`` and the reference DFlash decode mask. The draft is
    non-causal, so the band is symmetric; the forward half is inert in practice
    because the block is far shorter than any real window.

    Args:
        query: Tensor of shape [batch, heads, query, head_dim].
        key: Tensor of shape [batch, kv_heads, key, head_dim], where ``key``
            spans the context followed by the draft block.
        sliding_window: Maximum absolute position distance, exclusive.

    Returns:
        Floating-point tensor of shape [1, 1, query, key], in ``query.dtype``;
        ``0`` where attention is kept and ``-inf`` elsewhere. This additive
        representation has the same semantics for eager, SDPA, and the dense
        score-mask path of FlexAttention.
    """
    q_len, k_len = query.shape[-2], key.shape[-2]
    query_position = torch.arange(q_len, device=query.device).unsqueeze(-1) + (k_len - q_len)
    key_position = torch.arange(k_len, device=query.device).unsqueeze(0)
    distance = query_position - key_position
    keep = ((distance < sliding_window) & (-distance < sliding_window))[None, None]
    zero = torch.zeros((), dtype=query.dtype, device=query.device)
    neg_inf = torch.full((), float("-inf"), dtype=query.dtype, device=query.device)
    return torch.where(keep, zero, neg_inf)


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
        # Training enforces the same window through the block mask
        # (``components/attention/dflash_mask.py``); this covers the decode path,
        # where no explicit mask is supplied. ``is_causal`` stays False regardless --
        # the draft is non-causal by construction.
        layer_types = getattr(config, "layer_types", None)
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        window = getattr(config, "sliding_window", None)
        self.sliding_window = int(window) if layer_type == "sliding_attention" and window else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
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
        if attention_mask is None and self.sliding_window is not None:
            attention_mask = _sliding_window_mask(q, k, self.sliding_window)
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
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] | None = None,
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


def build_qwen3_dflash_draft_config(
    target_config,
    *,
    num_draft_layers: int,
    num_target_layers: int,
    block_size: int,
    dflash_config: dict,
    attention_backend: str,
) -> Qwen3Config:
    """Build the DFlash draft config for a Qwen3-shaped target.

    The draft is a small non-causal Qwen3 stack that reuses the target's
    architecture defaults (head_dim, rope_theta, rms_norm_eps, ...) and only
    shrinks the depth.

    Args:
        target_config: The target's config.
        num_draft_layers: Number of draft decoder layers.
        num_target_layers: Depth of the target; recorded so a reloaded draft
            config still describes which target it was trained on.
        block_size: DFlash block size.
        dflash_config: The recipe's DFlash block (``mask_token_id``,
            ``target_layer_ids``, and any subclass extras such as Domino's
            projector fields).
        attention_backend: The draft's attention implementation; must agree with
            the mask format the trainer builds.

    Returns:
        The draft ``Qwen3Config``.
    """
    draft_config = target_config.to_dict()
    draft_config["architectures"] = ["Qwen3DFlashDraftModel"]
    draft_config["num_hidden_layers"] = num_draft_layers
    # ``layer_types``/``max_window_layers`` are sized to the target's depth;
    # rebuild them for the (shallower) draft. The DFlash attention never uses
    # sliding windows, so every draft layer is full attention.
    draft_config["layer_types"] = ["full_attention"] * num_draft_layers
    draft_config["max_window_layers"] = num_draft_layers
    draft_config["num_target_layers"] = num_target_layers
    draft_config["block_size"] = block_size
    draft_config["dflash_config"] = dflash_config
    draft_config_obj = Qwen3Config.from_dict(draft_config)
    draft_config_obj._attn_implementation = attention_backend
    return draft_config_obj


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
    # Block class the stack is built from. DFlash 2 swaps in a block that wraps
    # each sublayer in a two-tap convolution; everything else is unchanged.
    decoder_layer_cls: type[Qwen3DFlashDecoderLayer] = Qwen3DFlashDecoderLayer

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [self.decoder_layer_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
        # The published drafters (DFlash and DFlash 2 alike) keep ``block_size``
        # inside ``dflash_config`` and carry no top-level key, so read it there
        # first; the top-level attribute is what this repo's own recipes have
        # always written, and stays the fallback for checkpoints saved that way.
        self.block_size = dflash_config.get("block_size", getattr(config, "block_size", None))
        if self.block_size is None:
            raise ValueError(
                "DFlash draft config carries no block_size, neither in dflash_config nor at the top level."
            )
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        # Output/input transforms some targets apply around the shared embedding
        # and head; identity for every Qwen3-family target (see compute_logits).
        multiplier = float(dflash_config.get("output_multiplier", 1.0))
        if multiplier <= 0:
            # A non-positive multiplier would flip or flatten the argmax, changing
            # every greedy pick without raising anywhere.
            raise ValueError(f"output_multiplier must be > 0 when set, got {multiplier}.")
        self.output_multiplier = multiplier
        softcap = dflash_config.get("final_logit_softcapping", None)
        if softcap is not None and float(softcap) <= 0:
            raise ValueError(f"final_logit_softcapping must be > 0 when set, got {softcap}.")
        self.final_logit_softcapping = None if softcap is None else float(softcap)
        self.input_embedding_scale = float(dflash_config.get("input_embedding_scale", 1.0))
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
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
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

    def embed_noise_block(self, target: nn.Module, block_ids: torch.LongTensor) -> torch.Tensor:
        """Embed a draft block with the target's (frozen, shared) input embedding.

        Applies ``input_embedding_scale`` from ``dflash_config`` -- 1.0, hence a
        no-op, for every Qwen3-family target. Unlike the reference, this calls the
        embedding *module* rather than indexing its weight table directly: under
        tensor parallelism the target's ``embed_tokens`` is vocab-parallel and only
        the module call carries the DTensor plan.

        Args:
            target: The frozen verifier.
            block_ids: Long tensor of shape [batch, draft]; the block's token ids,
                position 0 holding the verified anchor and the rest ``MASK``.

        Returns:
            Tensor of shape [batch, draft, hidden].
        """
        embedded = target.get_input_embeddings()(block_ids)
        return embedded if self.input_embedding_scale == 1.0 else embedded * self.input_embedding_scale

    def compute_logits(self, hidden: torch.Tensor, output_head: nn.Module) -> torch.Tensor:
        """Project draft hidden states to logits, applying the target's output transform.

        Some targets do not read their ``lm_head`` output raw: Muse Glimmer scales
        it by ``output_multiplier`` and squashes it through
        ``final_logit_softcapping``. The draft is trained against, and verified
        by, those transformed logits, so both training and decoding have to apply
        them -- otherwise the draft learns one distribution and is served another.
        Both fields live in ``dflash_config``; absent (every Qwen3-family target),
        this is just the head.

        Args:
            hidden: Tensor of shape [..., hidden]; draft hidden states with
                arbitrary leading dimensions.
            output_head: The frozen target's output projection.

        Returns:
            Tensor of shape [..., vocab].
        """
        logits = output_head(hidden)
        if self.output_multiplier != 1.0:
            logits = logits * self.output_multiplier
        if self.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int] | None,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.LongTensor:
        """Block-parallel speculative decoding: draft a block, verify with the target, accept the matching prefix.

        ``top_p`` / ``top_k`` truncate the *target's* distribution, which is what
        defines the emitted tokens; the draft proposes from plain temperature.

        Args:
            target: The frozen verifier.
            input_ids: Long tensor of shape [1, prompt].
            max_new_tokens: Maximum number of tokens to generate.
            stop_token_ids: Token ids that end generation, or ``None``.
            temperature: Sampling temperature; ``0`` decodes greedily.
            top_p: Nucleus mass to keep, in ``(0, 1]``.
            top_k: Candidates to keep, ``0`` for the whole vocabulary.

        Returns:
            Long tensor of shape [1, prompt + generated].
        """
        self.eval()
        validate_sampling(temperature, top_p, top_k)
        assert_target_supports_rollback(target)
        output_head = resolve_output_head(target)
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size

        output_ids = torch.full(
            (1, max_length + block_size), self.mask_token_id, dtype=torch.long, device=target.device
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
        # See the DFlash 2 draft's spec_generate: a bare ``DynamicCache()`` gives
        # every layer a plain attention cache, which a hybrid target (the Qwen3.5
        # family interleaves ``linear_attention`` with ``full_attention``) rejects.
        past_key_values_target = DynamicCache(config=getattr(target, "config", None))
        past_key_values_draft = DynamicCache(config=self.config)

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
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature, top_p, top_k)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        stop_tokens = (
            torch.tensor(stop_token_ids, dtype=output_ids.dtype, device=output_ids.device) if stop_token_ids else None
        )
        start = num_input_tokens
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = self.embed_noise_block(target, block_output_ids)
            draft_logits = self.compute_logits(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                )[:, -block_size + 1 :, :],
                output_head,
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
            posterior = sample(output.logits, temperature, top_p, top_k)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]
            if stop_tokens is not None and bool(
                torch.isin(output_ids[0, start - acceptance_length - 1 : start + 1], stop_tokens).any()
            ):
                break

        # ``start`` indexes the last committed token (the bonus token the target sampled
        # at the end of the previous block), so the generated sequence is exactly
        # ``[0, start]``; everything past it is still MASK padding.
        output_ids = output_ids[:, : min(start + 1, max_length)]
        if stop_tokens is not None:
            stop_indices = torch.isin(output_ids[0, num_input_tokens:], stop_tokens).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]
        return output_ids
