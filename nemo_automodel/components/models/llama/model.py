# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Custom Llama model implementation for NeMo Automodel.

This module provides a self-contained Llama implementation following HuggingFace's
implementation. Uses separate q_proj/k_proj/v_proj and gate_proj/up_proj (HF-style).

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

# Import HuggingFace's Llama components for attention
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    compute_lm_head_logits,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.deprecation import warn_deprecated_model_class
from nemo_automodel.components.models.llama.rope_utils import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_fused,
)
from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter
from nemo_automodel.shared.import_utils import get_check_model_inputs_decorator

check_model_inputs = get_check_model_inputs_decorator()


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper.

    Uses separate q_proj / k_proj / v_proj -- identical to the default
    HuggingFace Llama implementation.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        backend: Optional["BackendConfig"] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.backend = backend or BackendConfig()
        self.rope_fusion = self.backend.rope_fusion

        # Separate projections -- same layout as HuggingFace default Llama
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
        if self.backend.attn == "te":
            # BSHD continues through the HuggingFace attention interface; this
            # module is selected only for packed THD batches.
            self._te_thd_only = True
            self.attn_module, self.attn_func = initialize_attn_module_and_func(
                attn_impl="te",
                num_attention_heads=config.num_attention_heads,
                num_qk_channels=self.head_dim,
                num_v_channels=self.head_dim,
                softmax_scale=self.scaling,
                num_gqa_groups=config.num_key_value_heads,
                attention_dropout=config.attention_dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run dense attention over padded BSHD or packed THD hidden states.

        Args:
            hidden_states: Hidden states ``[B, S, H]`` or packed local states
                ``[T, H]``. ``B`` is batch, ``S`` is sequence, ``T`` is local
                total tokens, and ``H`` is hidden size.
            position_embeddings: RoPE tensors ``(cos, sin)`` for SDPA/eager or
                ``(cos, sin, freqs_cis)`` for fused TE RoPE. Local cosine/sine
                tensors follow ``[B, S, D]`` or ``[T, D]``; ``freqs_cis`` is
                global ``[S, 1, 1, D]``.
            attention_mask: Padded attention mask for BSHD. THD requires
                ``None`` and uses cumulative document lengths from ``kwargs``.
            past_key_values: Optional KV cache for BSHD generation. THD training
                does not support a cache.
            cache_position: Optional BSHD cache positions ``[S]``.
            **kwargs: THD requires ``qkv_format='thd'`` and ``cu_seqlens``
                ``[N + 1]``; CP additionally supplies ``cp_size`` and ``cp_rank``.

        Returns:
            Attention output shaped like ``hidden_states`` and optional BSHD
            attention weights. THD returns ``None`` for the weights.
        """
        is_thd = kwargs.get("qkv_format") == "thd"
        if is_thd and hidden_states.ndim != 2:
            raise ValueError(f"THD attention requires hidden_states [T, H], got {tuple(hidden_states.shape)}.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if is_thd:
            query_states = q.view(hidden_shape)
            key_states = k.view(hidden_states.shape[0], -1, self.head_dim)
            value_states = v.view(hidden_states.shape[0], -1, self.head_dim)
        else:
            query_states = q.view(hidden_shape).transpose(1, 2)
            key_states = k.view(*input_shape, -1, self.head_dim).transpose(1, 2)
            value_states = v.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        if self.rope_fusion and len(position_embeddings) == 3:
            cos, sin, freqs_cis = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_fused(
                query_states,
                key_states,
                freqs_cis,
                cu_seqlens=kwargs.get("cu_seqlens"),
                cp_size=kwargs.get("cp_size", 1),
                cp_rank=kwargs.get("cp_rank", 0),
            )
        else:
            cos, sin = position_embeddings[:2]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if is_thd:
            if past_key_values is not None:
                raise ValueError("Packed THD attention does not support past_key_values.")
            attn_module = getattr(self, "attn_module", None)
            if attn_module is None:
                raise ValueError("Packed THD attention requires backend.attn='te'.")
            query_states, key_states, value_states, te_kwargs = preprocess_args_and_kwargs_for_attn(
                query_states,
                key_states,
                value_states,
                attention_mask,
                "te",
                **kwargs,
            )
            attn_output = attn_module(query_states, key_states, value_states, **te_kwargs)
            attn_output = postprocess_output_for_attn(attn_output, "te")
            return self.o_proj(attn_output.flatten(1)), None

        # Handle past_key_values if provided (for generation)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Select attention interface based on config (matches HuggingFace)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaMLP(nn.Module):
    """SwiGLU MLP with separate gate_proj and up_proj -- identical to HuggingFace default."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        from transformers.activations import ACT2FN

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(GradientCheckpointingLayer):
    """Single Llama decoder layer with RMSNorm, attention, and MLP.

    Inherits from GradientCheckpointingLayer for efficient activation checkpointing.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        backend: BackendConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(
            config=config,
            layer_idx=layer_idx,
            backend=backend,
        )

        self.mlp = LlamaMLP(config=config)

        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


class LlamaModel(LlamaPreTrainedModel):
    """Llama transformer model (embeddings + decoder layers + norm)."""

    def __init__(
        self,
        config: LlamaConfig,
        backend: BackendConfig,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    backend=backend,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config, rope_fusion=backend.rope_fusion)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Run the Llama decoder in padded BSHD or packed THD layout.

        Args:
            input_ids: Token IDs ``[B, S]`` or packed local IDs ``[T]``.
            attention_mask: Optional padded mask ``[B, S]`` or broadcastable
                causal mask. THD ignores this mask and uses ``cu_seqlens``.
            position_ids: Position IDs ``[B, S]`` or packed local IDs ``[T]``.
            past_key_values: Optional BSHD generation cache; unsupported for THD.
            inputs_embeds: Alternative hidden inputs ``[B, S, H]`` or ``[T, H]``.
            use_cache: Whether to update the BSHD KV cache.
            output_attentions: Whether to request attention outputs.
            output_hidden_states: Whether to retain per-layer hidden states.
            return_dict: Whether to return ``BaseModelOutputWithPast``.
            cache_position: Optional BSHD cache positions ``[S]``.
            **kwargs: THD metadata including ``qkv_format``, ``cu_seqlens``,
                ``max_seqlen``, ``cp_size``, and ``cp_rank``.

        Returns:
            Decoder output with final states ``[B, S, H]`` or packed local
            states ``[T, H]`` and any requested hidden-state tensors in the
            same layout.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Validate inputs
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        is_thd = kwargs.get("qkv_format") == "thd"
        if is_thd and inputs_embeds.ndim != 2:
            raise ValueError(f"THD model input must be [T, H], got {tuple(inputs_embeds.shape)}.")
        if is_thd:
            use_cache = False
            if past_key_values is not None:
                raise ValueError("Packed THD training does not support past_key_values.")

        # Initialize cache if needed
        if not is_thd and use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Cache position (for tracking sequence position with KV cache)
        if not is_thd and cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Position IDs
        if position_ids is None:
            if is_thd:
                if kwargs.get("cp_size", 1) > 1:
                    raise ValueError("THD context parallelism requires explicit position_ids.")
                position_ids = torch.arange(inputs_embeds.shape[0], device=inputs_embeds.device)
            else:
                position_ids = cache_position.unsqueeze(0)

        # Create proper causal mask (matches HuggingFace implementation)
        causal_mask = None
        if not is_thd:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
            qkv_format="thd" if is_thd else "bshd",
            cp_size=kwargs.get("cp_size", 1),
        )

        all_hidden_states = () if output_hidden_states else None

        # Decoder layers (slice to support partial layer execution like in HF)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values if use_cache else None,
                    all_hidden_states,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class LlamaForCausalLM(HFCheckpointingMixin, LlamaPreTrainedModel):
    """Llama model with causal language modeling head."""

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = True
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = False

    @classmethod
    def from_config(
        cls,
        config: LlamaConfig,
        backend: Optional[BackendConfig] = None,
        **kwargs,
    ):
        return cls(config, backend, **kwargs)

    def __init__(
        self,
        config: LlamaConfig,
        backend: Optional[BackendConfig] = None,
    ):
        reject_unsupported_tie_word_embeddings(type(self), config)
        warn_deprecated_model_class("LlamaForCausalLM")
        super().__init__(config)
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = LlamaModel(config=config, backend=self.backend)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Create state_dict_adapter
        self.state_dict_adapter = LlamaStateDictAdapter(config=self.config)
        # Initialize weights and apply final processing
        self.post_init()

        # Transformers v5 does not reliably tie this custom model from the
        # dict-shaped _tied_weights_keys alone. Explicitly honor the config
        # flag after initialization.
        if getattr(config, "tie_word_embeddings", False):
            self.tie_weights()

        # Convert to configured dtype if specified
        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
            self.to(dtype=config.torch_dtype)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[LlamaForCausalLM] Attention implementation: {self.config._attn_implementation}")
            print(f"[LlamaForCausalLM] torch_dtype: {self.config.torch_dtype}")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        Forward pass returning CausalLMOutputWithPast.

        Args:
            input_ids: Token IDs ``[B, S]`` or packed local IDs ``[T]``.
            attention_mask: Optional padded attention mask. THD uses document
                boundaries from ``kwargs`` instead.
            position_ids: Position IDs ``[B, S]`` or packed local IDs ``[T]``.
            past_key_values: Optional BSHD KV cache; unsupported for THD.
            inputs_embeds: Optional hidden inputs ``[B, S, H]`` or ``[T, H]``.
            labels: Optional labels ``[B, S]`` or packed ``[T]``.
            use_cache: Whether to update the BSHD KV cache.
            output_attentions: Whether to request attention outputs.
            output_hidden_states: Whether to return per-layer hidden states.
            return_dict: Whether to return ``CausalLMOutputWithPast``.
            cache_position: Optional BSHD cache positions ``[S]``.
            logits_to_keep: Positions to project from hidden size ``H`` to
                vocabulary size ``V``.
            **kwargs: THD metadata. ``cu_seqlens`` is ``[N + 1]`` and identifies
                packed-document boundaries; CP adds ``cp_size`` and ``cp_rank``.

        Returns:
            Causal LM output with logits ``[B, S, V]``. Packed THD logits are
            restored to ``[1, T, V]`` for the shared loss contract.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_thd = kwargs.get("qkv_format") == "thd"
        if is_thd:
            if position_ids is None:
                raise ValueError("Packed THD input requires position_ids.")
            kwargs.pop("padding_mask", None)
            if input_ids is not None and input_ids.ndim > 1:
                input_ids = input_ids.squeeze(0)
            if position_ids.ndim > 1:
                position_ids = position_ids.squeeze(0)
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "max_seqlen":
                    kwargs[key] = value.item()
                    continue
                if value.ndim > 1:
                    value = value.squeeze(0)
                if key in ("cu_seqlens", "cu_seqlens_padded"):
                    value = value[value != -1000].contiguous()
                kwargs[key] = value
            if inputs_embeds is not None and inputs_embeds.ndim > 2:
                inputs_embeds = inputs_embeds.squeeze(0)
            if labels is not None and labels.ndim > 1:
                labels = labels.squeeze(0)
            attention_mask = None

        # Always use return_dict internally so we can reliably access fields.
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logits = compute_lm_head_logits(self.lm_head, hidden_states, logits_to_keep, is_thd=is_thd).logits

        loss = None
        if labels is not None:
            if is_thd and labels.ndim == 1:
                labels = labels.unsqueeze(0)
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
        if return_dict:
            return out
        return out.to_tuple()


ModelClass = LlamaForCausalLM
