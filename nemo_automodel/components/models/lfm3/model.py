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
#
# The layer composition and forward passes are adapted from Hugging Face Transformers
# (Apache-2.0), https://github.com/huggingface/transformers at v5.23.0
# (52b080b6416a7897e2af90febeb4dfad22b84d19): src/transformers/models/lfm3/modeling_lfm3.py.
# Copyright 2026 The HuggingFace Team. All rights reserved.

"""LFM3 causal LM built on the LFM2 and Qwen3 blocks of the pinned Transformers release."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.lfm2.modeling_lfm2 import Lfm2RMSNorm, Lfm2ShortConv, apply_mask_to_padding_states
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeMLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from nemo_automodel.components.models.common import compute_lm_head_logits
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.lfm3.config import Lfm3Config
from nemo_automodel.components.models.llama.rope_utils import apply_rotary_pos_emb
from nemo_automodel.shared.import_utils import get_check_model_inputs_decorator

__all__ = ["Lfm3ForCausalLM"]

check_model_inputs = get_check_model_inputs_decorator()


def rotary_factors(
    position_ids: torch.Tensor, rope: dict[str, str | float], head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one attention type's rotary factors in float32, straight from the config.

    The first ``partial_rotary_factor`` share of the frequencies follows the
    progression of the full head dimension; the rest are zero, so those
    channels pass through unrotated. Nothing is cached, so no model-wide dtype
    cast can round the frequencies.

    Args:
        position_ids: Long Tensor of shape [batch, sequence].
        rope: The attention type's entry of ``Lfm3Config.rope_parameters``.
        head_dim: Channels per attention head.

    Returns:
        Pair of float32 cosine and sine Tensors, each of shape [batch, sequence, head_dim].
    """
    rotated = int(rope.get("partial_rotary_factor", 1.0) * head_dim) // 2
    inv_freq = torch.zeros(head_dim // 2)
    inv_freq[:rotated] = 1.0 / (rope["rope_theta"] ** (torch.arange(0, 2 * rotated, 2, dtype=torch.float) / head_dim))
    angles = position_ids[..., None].float() * inv_freq.to(position_ids.device)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


class Lfm3Attention(Qwen3Attention):
    """Qwen3 grouped-query attention with a sigmoid gate on each head's output."""

    def __init__(self, config: Lfm3Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.gate_proj = (
            nn.Linear(config.hidden_size, config.num_attention_heads, bias=False) if config.use_attention_gate else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend over the sequence, then scale each head by its gate.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_embeddings: Pair of float32 cosine and sine Tensors, each of
                shape [batch, sequence, head_dim].
            attention_mask: Optional Tensor of shape [batch, 1, sequence, key_sequence],
                additive or boolean as the attention backend expects.
            past_key_values: Optional generation cache, updated in place.
            **kwargs: Arguments for the Hugging Face attention dispatcher.

        Returns:
            Pair of the output Tensor of shape [batch, sequence, hidden] and, for the
            eager backend only, attention weights of shape [batch, heads, sequence, key_sequence].
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if self.gate_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), attn_weights


class Lfm3ShortConv(Lfm2ShortConv):
    """LFM2 gated short convolution, always run in PyTorch.

    The parent dispatches to Hub kernels that only cover widths 2 to 4; LFM3 uses width 7.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gate, convolve causally along the sequence, and gate again.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            past_key_values: Optional generation cache. Its window of earlier gated
                inputs is prepended as left context and updated in place.
            attention_mask: Optional Tensor of shape [batch, sequence], nonzero for
                real tokens and zero for padding.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        sequence = hidden_states.shape[1]
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        input_gate, output_gate, values = self.in_proj(hidden_states).transpose(-1, -2).chunk(3, dim=-2)
        gated = input_gate * values
        if past_key_values is not None:
            gated = past_key_values.update_conv_state(gated, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
        # `self.conv` pads both ends: the leading outputs are causal, the trailing ones look ahead.
        mixed = self.conv(gated.to(self.conv.weight.dtype))[..., : gated.shape[-1]][..., -sequence:].to(gated.dtype)
        return self.out_proj((output_gate * mixed).transpose(-1, -2).contiguous())


class Lfm3DecoderLayer(GradientCheckpointingLayer):
    """Pre-norm operator (attention or convolution) and MLP, each normalized again before its residual add."""

    def __init__(self, config: Lfm3Config, layer_idx: int) -> None:
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] != "conv"
        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(config, layer_idx)
        else:
            self.conv = Lfm3ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MoeMLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        post_norm = Lfm2RMSNorm if config.use_sandwich_norm else nn.Identity
        self.post_operator_norm = post_norm(config.hidden_size, eps=config.norm_eps)
        self.post_ffn_norm = post_norm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Run one decoder layer.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_embeddings: For attention layers, a pair of float32 cosine and
                sine Tensors, each of shape [batch, sequence, head_dim].
            attention_mask: Attention layers take a Tensor of shape
                [batch, 1, sequence, key_sequence]; convolution layers take a padding
                Tensor of shape [batch, sequence]. Either may be ``None``.
            past_key_values: Optional generation cache, updated in place.
            **kwargs: Arguments for the Hugging Face attention dispatcher.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        operator_input = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            operator_output, _ = self.self_attn(
                operator_input, position_embeddings, attention_mask, past_key_values, **kwargs
            )
        else:
            operator_output = self.conv(operator_input, past_key_values, attention_mask)
        hidden_states = hidden_states + self.post_operator_norm(operator_output)
        return hidden_states + self.post_ffn_norm(self.feed_forward(self.ffn_norm(hidden_states)))


class Lfm3PreTrainedModel(PreTrainedModel):
    """Weight initialization and loading shared by the LFM3 models."""

    config_class = Lfm3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Lfm3DecoderLayer, "attentions": Lfm3Attention}


class Lfm3Model(Lfm3PreTrainedModel):
    """LFM3 decoder: token embeddings, the mixed convolution and attention stack, and a final norm."""

    def __init__(self, config: Lfm3Config) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(Lfm3DecoderLayer(config, idx) for idx in range(config.num_hidden_layers))
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def _layer_masks(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Build one mask per layer type present in the model.

        Each attention mask is sized from the first cache layer of its own type:
        with a leading convolution layer the mask helpers would otherwise pick
        whichever attention layer comes first.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            attention_mask: Optional padding Tensor of shape [batch, key_sequence],
                or an already prepared Tensor of shape [batch, 1, sequence, key_sequence].
            past_key_values: Optional generation cache.
            position_ids: Long Tensor of shape [batch, sequence].

        Returns:
            Mapping from layer type to its mask. Attention types map to a Tensor of
            shape [batch, 1, sequence, key_sequence], ``"conv"`` to a padding Tensor
            of shape [batch, sequence]; either is ``None`` when nothing needs masking.
        """
        layer_types = self.config.layer_types
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        builders = {"full_attention": create_causal_mask, "sliding_attention": create_sliding_window_causal_mask}
        masks = {
            layer_type: build(**mask_kwargs, layer_idx=layer_types.index(layer_type))
            for layer_type, build in builders.items()
            if layer_type in layer_types
        }
        if "conv" in layer_types:
            masks["conv"] = create_recurrent_attention_mask(**mask_kwargs)
        return masks

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Run the decoder.

        Args:
            input_ids: Long Tensor of shape [batch, sequence]. Exclusive with ``inputs_embeds``.
            attention_mask: Optional padding Tensor of shape [batch, key_sequence], where
                ``key_sequence`` also counts cached tokens, or the per-layer-type mapping
                that ``_layer_masks`` returns.
            position_ids: Optional long Tensor of shape [batch, sequence]. Defaults to
                consecutive positions continuing from the cache.
            past_key_values: Optional generation cache, updated in place.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            use_cache: Whether to create and return a generation cache.
            **kwargs: Hugging Face output-capture and attention arguments.

        Returns:
            Output whose ``last_hidden_state`` is a Tensor of shape [batch, sequence, hidden].
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen).unsqueeze(0)

        masks = attention_mask
        if not isinstance(masks, dict):
            masks = self._layer_masks(inputs_embeds, attention_mask, past_key_values, position_ids)
        position_embeddings = {
            layer_type: rotary_factors(position_ids, self.config.rope_parameters[layer_type], self.config.head_dim)
            for layer_type in masks
            if layer_type != "conv"
        }

        hidden_states = inputs_embeds
        for layer_type, layer in zip(self.config.layer_types, self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings.get(layer_type),
                attention_mask=masks[layer_type],
                past_key_values=past_key_values,
                **kwargs,
            )
        return BaseModelOutputWithPast(
            last_hidden_state=self.embedding_norm(hidden_states), past_key_values=past_key_values
        )


class Lfm3ForCausalLM(HFCheckpointingMixin, Lfm3PreTrainedModel, GenerationMixin):
    """LFM3 with a vocabulary head, loadable from LiquidAI's released checkpoints."""

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallelism is undeclared until corresponding recipes are verified."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: Lfm3Config) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.model = Lfm3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def tie_weights(self, **kwargs: bool | set[str] | None) -> None:
        """Share the vocabulary head with the token embeddings when the config asks for it."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding table."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the token embedding table and retie the head."""
        self.model.embed_tokens = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        """Return the vocabulary head."""
        return self.lm_head

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Predict next-token logits and, given labels, the causal LM loss.

        Args:
            input_ids: Long Tensor of shape [batch, sequence]. Exclusive with ``inputs_embeds``.
            attention_mask: Optional padding Tensor of shape [batch, key_sequence], where
                ``key_sequence`` also counts cached tokens.
            position_ids: Optional long Tensor of shape [batch, sequence].
            past_key_values: Optional generation cache, updated in place.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            labels: Optional long Tensor of shape [batch, sequence]; ``-100`` is ignored.
                Shifted internally.
            use_cache: Whether to create and return a generation cache.
            logits_to_keep: Project only the last ``logits_to_keep`` positions (0 keeps
                all), or the positions in a long Tensor of shape [kept].
            **kwargs: Hugging Face output-capture, attention and loss arguments.

        Returns:
            Output whose ``logits`` is a Tensor of shape [batch, kept, vocab] and whose
            ``loss`` is a scalar Tensor when labels are given.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        logits = compute_lm_head_logits(self.lm_head, outputs.last_hidden_state, logits_to_keep).logits
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


ModelClass = Lfm3ForCausalLM
