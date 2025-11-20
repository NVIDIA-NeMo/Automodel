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

"""Custom Qwen2 model implementation for NeMo Automodel.

This module provides a self-contained Qwen2 implementation with combined QKV/gate_up projections.
Uses shared components from common/ for fused projections.

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.components.models.qwen2.build_qwen2_model
  pretrained_model_name_or_path: Qwen/Qwen2.5-7B
  use_fused_qkv: true
  use_fused_gate_up: true
```
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import check_model_inputs

from nemo_automodel.components.models.common.combined_projection import (
    CombinedGateUpMLP,
    CombinedQKVAttentionMixin,
)
from nemo_automodel.components.models.qwen2.state_dict_adapter import Qwen2StateDictAdapter
from nemo_automodel.components.moe.utils import BackendConfig
from nemo_automodel.shared.utils import dtype_from_str

__all__ = ["build_qwen2_model", "Qwen2ForCausalLM"]


class Qwen2Attention(CombinedQKVAttentionMixin, nn.Module):
    """Multi-headed attention with combined QKV projection.

    Uses CombinedQKVAttentionMixin for efficient combined QKV projection.
    ALWAYS uses combined projections - this is the whole point of the custom implementation.
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Setup combined QKV projection using mixin (ALWAYS combined in custom implementation)
        self.setup_qkv_projection(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=self.head_dim,
            bias=True,  # Qwen2 uses bias in attention
        )

        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

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

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Select attention interface based on config
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
            sliding_window=self.sliding_window,  # Qwen2 feature
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    """Single Qwen2 decoder layer with RMSNorm, attention, and combined MLP.

    ALWAYS uses combined projections - this is the whole point of the custom implementation.
    """

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        # ALWAYS use combined QKV in custom implementation
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        # ALWAYS use combined gate_up MLP in custom implementation
        self.mlp = CombinedGateUpMLP(config=config)

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

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


class Qwen2PreTrainedModel(PreTrainedModel):
    """Abstract class for Qwen2 pretrained models."""

    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }


class Qwen2Model(Qwen2PreTrainedModel):
    """Qwen2 transformer model (embeddings + decoder layers + norm).

    ALWAYS uses combined projections - this is the whole point of the custom implementation.
    """

    def __init__(
        self,
        config: Qwen2Config,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # ALWAYS use combined projections in all layers
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create masks (Qwen2 supports sliding window attention)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    """Qwen2 model with causal language modeling head.

    ALWAYS uses combined projections - this is the whole point of the custom implementation.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config: Qwen2Config,
        backend: Optional[BackendConfig] = None,
    ):
        super().__init__(config)
        self.backend = backend or BackendConfig()
        # ALWAYS use combined projections
        self.model = Qwen2Model(config=config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Create state_dict_adapter if enabled (needed to convert HF checkpoints)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen2StateDictAdapter(config=self.config)

        # Initialize weights and apply final processing
        self.post_init()

        # Tie weights if specified in config (standard for Qwen2/Llama)
        # Must be done after post_init() to ensure embed_tokens is initialized
        if getattr(config, "tie_word_embeddings", True):
            self.tie_weights()

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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Forward pass returning CausalLMOutputWithPast."""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # DTensor compatibility with pytorch 2.9.0: when logits_to_keep=0, slice(0, None, None) would select all elements
        # but DTensor cannot handle sliced DTensor, which will raise error message:
        # NotImplementedError: Operator aten.alias.default does not have a sharding strategy registered.
        # Solution: Skip slicing entirely when logits_to_keep=0 to avoid DTensor issues in TP with sequence parallel.
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def save_pretrained_hf_format(self, save_directory: str, **kwargs):
        """Save model in HuggingFace-compatible format by converting combined projections.

        This method converts the custom model's combined projections (qkv_proj, gate_up_proj)
        back to HuggingFace's separate projections format before saving, making the checkpoint
        loadable with AutoModelForCausalLM.from_pretrained().

        Args:
            save_directory: Directory where the model will be saved
            **kwargs: Additional arguments passed to config.save_pretrained and save_file
        """
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Convert state dict to HF format
        if hasattr(self, "state_dict_adapter"):
            custom_state_dict = self.state_dict()
            hf_state_dict = self.state_dict_adapter.to_hf(custom_state_dict)
        else:
            hf_state_dict = self.state_dict()

        # Handle tied weights: remove duplicate tied weights before saving
        # In Qwen2, lm_head.weight is tied to model.embed_tokens.weight
        # HuggingFace expects only model.embed_tokens.weight to be saved
        if "lm_head.weight" in hf_state_dict and "model.embed_tokens.weight" in hf_state_dict:
            # Check if they actually share memory
            if hf_state_dict["lm_head.weight"].data_ptr() == hf_state_dict["model.embed_tokens.weight"].data_ptr():
                # Remove lm_head.weight as it's tied to embed_tokens
                hf_state_dict = {k: v for k, v in hf_state_dict.items() if k != "lm_head.weight"}

        # Save weights in safetensors format
        save_file(hf_state_dict, os.path.join(save_directory, "model.safetensors"), metadata={"format": "pt"})


def build_qwen2_model(pretrained_model_name_or_path: str, **kwargs: Any) -> nn.Module:
    """Build a custom Qwen2 model with combined projections.

    This custom implementation ALWAYS uses combined QKV and gate_up projections
    for better efficiency. The state dict adapter handles conversion from HuggingFace
    checkpoints (which have separate projections) to the combined format.

    Args:
        pretrained_model_name_or_path: HuggingFace model card name (e.g., "Qwen/Qwen2.5-7B")
        **kwargs: Override config parameters. Common parameters include:
                  - torch_dtype: Model dtype ("bf16", "fp32", etc.)
                  - attn_implementation: Attention backend ("eager", "sdpa", "flash_attention_2")
                  - num_hidden_layers: Number of layers (useful for testing)

    Returns:
        Qwen2ForCausalLM model instance with combined projections

    Example:
        # Load custom Qwen2 with combined projections (ALWAYS enabled)
        model = build_qwen2_model("Qwen/Qwen2.5-7B", torch_dtype="bf16")
    """
    # Extract and convert torch_dtype
    torch_dtype = kwargs.pop("torch_dtype", None)
    if torch_dtype is not None and isinstance(torch_dtype, str):
        torch_dtype = dtype_from_str(torch_dtype)
    elif torch_dtype is None:
        torch_dtype = torch.bfloat16

    # Extract attention implementation
    attn_implementation = kwargs.pop("attn_implementation", None)

    # Load config from HuggingFace
    config = Qwen2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Ensure architectures is set for compatibility
    if not hasattr(config, "architectures") or config.architectures is None:
        config.architectures = ["Qwen2ForCausalLM"]

    # Set attention implementation with auto-detection
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    elif not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        try:
            config._attn_implementation = "flash_attention_2"
        except (ImportError, ModuleNotFoundError):
            if hasattr(F, "scaled_dot_product_attention"):
                config._attn_implementation = "sdpa"
            else:
                config._attn_implementation = "eager"

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"[build_qwen2_model] Attention implementation: {config._attn_implementation}")
        print("[build_qwen2_model] Custom implementation with COMBINED QKV and gate_up projections")
        print(f"[build_qwen2_model] torch_dtype: {torch_dtype}")

    # Create backend config with HF state dict adapter enabled
    # This allows loading HuggingFace checkpoints (separate projections) into our combined format
    backend = BackendConfig(enable_hf_state_dict_adapter=True)

    # Create model with combined projections (ALWAYS)
    model = Qwen2ForCausalLM(config=config, backend=backend)

    # Convert to specified dtype
    model = model.to(dtype=torch_dtype)

    return model
