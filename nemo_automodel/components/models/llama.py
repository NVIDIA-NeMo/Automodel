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

This module provides a self-contained Llama implementation with combined QKV/gate_up projections.
Following HuggingFace's implementation.

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.components.models.llama.build_llama_model
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
```
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import Cache, DynamicCache

# Import HuggingFace's Llama components directly to ensure exact same behavior
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP as HFLlamaMLP,  # HuggingFace's standard MLP
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)

__all__ = ["build_llama_model", "LlamaForCausalLM"]


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        use_fused_qkv: bool = True,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_fused_qkv = use_fused_qkv

        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim

        if use_fused_qkv:
            # Combined QKV projection for improved efficiency
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                (config.num_attention_heads + 2 * config.num_key_value_heads) * self.head_dim,
                bias=config.attention_bias,
            )
        else:
            # Separate Q, K, V projections (standard HuggingFace)
            self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=config.attention_bias)
            self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=config.attention_bias)
            self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=config.attention_bias)
        
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.use_fused_qkv:
            # Combined QKV projection and split
            qkv = self.qkv_proj(hidden_states)
            # Compute split sizes based on actual tensor size (handles TP sharding)
            qkv_size = qkv.shape[-1]
            total_size = self.q_size + 2 * self.kv_size
            local_q_size = (self.q_size * qkv_size) // total_size
            local_kv_size = (self.kv_size * qkv_size) // total_size
            q, k, v = qkv.split([local_q_size, local_kv_size, local_kv_size], dim=-1)
        else:
            # Separate Q, K, V projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        
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


class FusedLlamaMLP(nn.Module):
    """SwiGLU MLP with fused gate_up projection for efficiency.
    
    This is an experimental optimization that fuses gate_proj and up_proj into a single projection.
    Note: May not always be faster than separate projections due to split overhead.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Fused gate and up projections
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 
            2 * self.intermediate_size, 
            bias=config.mlp_bias
        )
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


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer with RMSNorm, attention, and MLP."""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        use_fused_qkv: bool = True,
        use_fused_gate_up: bool = True,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(
            config=config,
            layer_idx=layer_idx,
            use_fused_qkv=use_fused_qkv,
        )
        
        # Use HuggingFace's standard MLP or our fused version
        if use_fused_gate_up:
            self.mlp = FusedLlamaMLP(config=config)
        else:
            self.mlp = HFLlamaMLP(config=config)
        
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
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


class LlamaModel(nn.Module):
    """Llama transformer model (embeddings + decoder layers + norm)."""

    def __init__(
        self,
        config: LlamaConfig,
        use_fused_qkv: bool = True,
        use_fused_gate_up: bool = True,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    use_fused_qkv=use_fused_qkv,
                    use_fused_gate_up=use_fused_gate_up,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Cache position (for tracking sequence position with KV cache)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=input_ids.device
            )

        # Position IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create proper causal mask (matches HuggingFace implementation)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,  # Use proper causal mask
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """Llama model with causal language modeling head."""

    def __init__(
        self,
        config: LlamaConfig,
        use_fused_qkv: bool = True,
        use_fused_gate_up: bool = True,
    ):
        super().__init__()
        
        # Store config (required for LoRA and other tools)
        self.config = config
        
        self.model = LlamaModel(
            config=config,
            use_fused_qkv=use_fused_qkv,
            use_fused_gate_up=use_fused_gate_up,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)
        return logits

    def _init_weights(self):
        """Initialize weights following Llama conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)


def build_llama_model(pretrained_model_name_or_path: str, **kwargs: Any) -> nn.Module:
    """Build a custom Llama model with optional fused projections.
    
    This function loads the config from a HuggingFace model card and builds
    a custom Llama model with optional fused QKV and gate_up projections for efficiency.

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
                  - use_fused_qkv: Whether to use fused QKV projection
                  - use_fused_gate_up: Whether to use fused gate_up projection

    Returns:
        LlamaForCausalLM model instance
        
    Example:
        # Load with separate projections (default, matches HuggingFace)
        model = build_llama_model("meta-llama/Meta-Llama-3-70B")
        
        # Use SDPA for faster attention
        model = build_llama_model("meta-llama/Meta-Llama-3-70B", 
                                   attn_implementation="sdpa")
        
        # Load with fused gate_up (experimental, may be slower due to split overhead)
        model = build_llama_model("meta-llama/Meta-Llama-3-70B", 
                                   use_fused_gate_up=True)
        
        # Override for testing with fewer layers
        model = build_llama_model("meta-llama/Meta-Llama-3-70B", num_hidden_layers=4)
    """
    # Extract fusion options (not part of HuggingFace config)
    # Note: HuggingFace uses fused QKV (qkv_proj) but separate gate/up projections
    use_fused_qkv = kwargs.pop('use_fused_qkv', True)
    use_fused_gate_up = kwargs.pop('use_fused_gate_up', True)
    
    # Extract attention implementation if specified, otherwise auto-detect
    # This matches nemo_automodel/_transformers/auto_model.py approach
    attn_implementation = kwargs.pop('attn_implementation', None)
    
    # Load config from HuggingFace (with any overrides from kwargs)
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    
    # Ensure architectures is set for LoRA compatibility
    if not hasattr(config, 'architectures') or config.architectures is None:
        config.architectures = ["LlamaForCausalLM"]
    
    # Set attention implementation with auto-detection
    # Priority: user-specified > existing in config > auto-detect (flash_attention_2 > sdpa > eager)
    # This matches the logic in nemo_automodel/_transformers/auto_model.py
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    elif not hasattr(config, '_attn_implementation') or config._attn_implementation is None:
        # Auto-detect best available implementation (same as nemo_automodel default)
        try:
            # Try flash_attention_2 first (fastest)
            from flash_attn import flash_attn_func
            config._attn_implementation = "flash_attention_2"
        except (ImportError, ModuleNotFoundError):
            # Fall back to SDPA if available (PyTorch 2.0+)
            if hasattr(F, 'scaled_dot_product_attention'):
                config._attn_implementation = "sdpa"
            else:
                # Final fallback to eager
                config._attn_implementation = "eager"
    
    if torch.distributed.get_rank() == 0:
        print(f"[build_llama_model] Attention implementation: {config._attn_implementation}")
        print(f"[build_llama_model] Use fused QKV: {use_fused_qkv}")
        print(f"[build_llama_model] Use fused gate_up: {use_fused_gate_up}")

    
    return LlamaForCausalLM(
        config=config,
        use_fused_qkv=use_fused_qkv,
        use_fused_gate_up=use_fused_gate_up,
    )
