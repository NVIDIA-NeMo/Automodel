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
  _target_: nemo_automodel.components.models.llama.build_llama_model
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
```
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

# Import HuggingFace's Llama components directly to ensure exact same behavior
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
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
from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter
from nemo_automodel.components.moe.utils import BackendConfig
from nemo_automodel.shared.utils import dtype_from_str

__all__ = ["build_llama_model", "LlamaForCausalLM"]


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
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(
            config=config,
            layer_idx=layer_idx,
        )

        # ALWAYS use combined gate_up MLP for efficiency
        self.mlp = CombinedGateUpMLP(config=config)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
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

        # Decoder layers (slice to support partial layer execution like in HF)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    """Llama model with causal language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config: LlamaConfig,
        backend: Optional[BackendConfig] = None,
    ):
        super().__init__(config)
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = LlamaModel(config=config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Create state_dict_adapter if enabled
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = LlamaStateDictAdapter(config=self.config)

        # Initialize weights and apply final processing
        self.post_init()

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
        # In Llama, lm_head.weight is tied to model.embed_tokens.weight
        # HuggingFace expects only model.embed_tokens.weight to be saved
        if "lm_head.weight" in hf_state_dict and "model.embed_tokens.weight" in hf_state_dict:
            # Check if they actually share memory
            if hf_state_dict["lm_head.weight"].data_ptr() == hf_state_dict["model.embed_tokens.weight"].data_ptr():
                # Remove lm_head.weight as it's tied to embed_tokens
                hf_state_dict = {k: v for k, v in hf_state_dict.items() if k != "lm_head.weight"}

        # Save weights
        save_file(hf_state_dict, os.path.join(save_directory, "model.safetensors"))

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

        # Only compute necessary logits (optimization for training and generation)
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


def build_llama_model(pretrained_model_name_or_path: str, **kwargs: Any) -> nn.Module:
    """Build a custom Llama model with combined projections for efficiency.

    This function loads the config from a HuggingFace model card and builds
    a custom Llama model with combined QKV and gate_up projections for improved efficiency.

    Args:
        pretrained_model_name_or_path: HuggingFace model card name (e.g., "meta-llama/Meta-Llama-3-70B")
        **kwargs: Override config parameters. Common parameters include:
                  - vocab_size: Vocabulary size
                  - hidden_size: Hidden dimension size
                  - num_hidden_layers: Number of transformer layers (useful for testing)
                  - num_attention_heads: Number of attention heads
                  - num_key_value_heads: Number of key/value heads for GQA
                  - intermediate_size: MLP intermediate size
                  - max_position_embeddings: Maximum sequence length
                  - rms_norm_eps: RMSNorm epsilon
                  - rope_theta: RoPE base frequency
                  - attention_dropout: Attention dropout probability
                  - pad_token_id: Padding token ID
                  - attn_implementation: Attention backend ("eager", "sdpa", "flash_attention_2")
                  - torch_dtype: Model dtype (default: bfloat16)

    Returns:
        LlamaForCausalLM model instance with combined projections

    Example:
        # Load with default settings (combined projections, bfloat16)
        model = build_llama_model("meta-llama/Meta-Llama-3-70B")

        # Use SDPA for faster attention
        model = build_llama_model("meta-llama/Meta-Llama-3-70B",
                                   attn_implementation="sdpa")

        # Override for testing with fewer layers
        model = build_llama_model("meta-llama/Meta-Llama-3-70B", num_hidden_layers=4)
    """
    # Extract and convert torch_dtype
    torch_dtype = kwargs.pop("torch_dtype", None)
    if torch_dtype is not None and isinstance(torch_dtype, str):
        torch_dtype = dtype_from_str(torch_dtype)
    elif torch_dtype is None:
        torch_dtype = torch.bfloat16  # Default to bf16

    # Extract attention implementation if specified, otherwise auto-detect
    # This matches nemo_automodel/_transformers/auto_model.py approach
    attn_implementation = kwargs.pop("attn_implementation", None)

    # Load config from HuggingFace (with any overrides from kwargs)
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Ensure architectures is set for LoRA compatibility
    if not hasattr(config, "architectures") or config.architectures is None:
        config.architectures = ["LlamaForCausalLM"]

    # Set attention implementation with auto-detection
    # Priority: user-specified > existing in config > auto-detect (flash_attention_2 > sdpa > eager)
    # This matches the logic in nemo_automodel/_transformers/auto_model.py
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    elif not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        # Auto-detect best available implementation (same as nemo_automodel default)
        try:
            # Try flash_attention_2 first (fastest)
            config._attn_implementation = "flash_attention_2"
        except (ImportError, ModuleNotFoundError):
            # Fall back to SDPA if available (PyTorch 2.0+)
            if hasattr(F, "scaled_dot_product_attention"):
                config._attn_implementation = "sdpa"
            else:
                # Final fallback to eager
                config._attn_implementation = "eager"

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"[build_llama_model] Attention implementation: {config._attn_implementation}")
        print(f"[build_llama_model] torch_dtype: {torch_dtype}")

    # Create backend config with HF state dict adapter enabled
    backend = BackendConfig(enable_hf_state_dict_adapter=True)

    # Create model with combined projections
    model = LlamaForCausalLM(config=config, backend=backend)

    # need to convert model manually since LlamaForCausalLM does not support to(dtype=...)
    model = model.to(dtype=torch_dtype)

    return model
