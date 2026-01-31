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

This module provides a self-contained Llama implementation with combined QKV and gate_up projections
for improved efficiency. Following HuggingFace's implementation with optimizations.

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
```
"""

from __future__ import annotations

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

from nemo_automodel.components.models.common import (
    BackendConfig,
    CombinedGateUpMLP,
    CombinedQKVAttentionMixin,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.llama.rope_utils import LlamaRotaryEmbedding, apply_rotary_pos_emb
from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter
from nemo_automodel.shared.import_utils import get_check_model_inputs_decorator

check_model_inputs = get_check_model_inputs_decorator()


class LlamaAttention(CombinedQKVAttentionMixin, nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with combined QKV projection."""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Combined QKV projection for improved efficiency
        self.setup_qkv_projection(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=self.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
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
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Compute Q, K, V using mixin (handles fused or separate projection)
        q, k, v = self.compute_qkv(hidden_states)

        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
    """SwiGLU MLP with combined gate_up projection for efficiency."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Combined gate and up projections
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        from transformers.activations import ACT2FN

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project and split into gate and up
        gate_up = self.gate_up_proj(x)
        # Handle tensor parallelism: split based on actual tensor size
        gate_up_size = gate_up.shape[-1]
        local_intermediate_size = gate_up_size // 2
        gate, up = gate_up.split([local_intermediate_size, local_intermediate_size], dim=-1)

        return self.down_proj(self.act_fn(gate) * up)


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
        )

        # ALWAYS use combined gate_up MLP for efficiency
        self.mlp = CombinedGateUpMLP(config=config)

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
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
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

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Cache position (for tracking sequence position with KV cache)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Position IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create proper causal mask (matches HuggingFace implementation)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

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

        # Convert to configured dtype if specified
        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
            self.to(dtype=config.torch_dtype)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[LlamaForCausalLM] Attention implementation: {self.config._attn_implementation}")
            print("[LlamaForCausalLM] Custom implementation with COMBINED QKV and gate_up projections")
            print(f"[LlamaForCausalLM] torch_dtype: {self.config.torch_dtype}")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
            input_ids: (batch_size, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional cached key/values
            inputs_embeds: Optional pre-computed embeddings
            labels: Optional labels for computing loss
            use_cache: Whether to use KV caching
            cache_position: Position in cache
            logits_to_keep: Number of final logits to compute (0=all, N=last N tokens)

        Returns:
            CausalLMOutputWithPast with loss, logits, past_key_values
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        # Only compute necessary logits (optimization for training and generation)
        # DTensor compatibility with pytorch 2.9.0: when logits_to_keep=0, slice(0, None, None) would select all
        # elements but DTensor cannot handle sliced DTensor. Skip slicing when logits_to_keep=0.
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
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
