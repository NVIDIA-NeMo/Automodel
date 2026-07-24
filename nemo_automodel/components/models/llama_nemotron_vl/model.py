# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoProcessor, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
)
from transformers.models.llama.modeling_llama import (
    LlamaModel as HFLlamaModel,
)
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.utils import logging

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.llama.rope_utils import (
    LlamaRotaryEmbedding as OptimizedLlamaRotaryEmbedding,
)
from nemo_automodel.components.models.llama_nemotron_vl.config import (
    LlamaBidirectionalConfig,
    LlamaNemotronVLConfig,
)
from nemo_automodel.components.models.llama_nemotron_vl.layers import (
    FusedTESiglipEncoderLayer,
    FusedTESiglipTransformerLayer,
    OptimizedFusedTERMSNormMLP,
    OptimizedFusedTERMSNormQKV,
    OptimizedLlamaAttention,
    OptimizedLlamaDecoderLayer,
    OptimizedLlamaMLP,
    disable_unused_siglip_pooling_head_grad,
    replace_llama_mlp_with_te_fused,
    replace_llama_qkv_with_te_fused,
    replace_siglip_encoder_layers_with_te_fused,
)
from nemo_automodel.components.models.llama_nemotron_vl.state_dict_adapter import (
    LlamaNemotronVLEncoderStateDictAdapter,
)

logger = logging.get_logger(__name__)


# Check if native create_bidirectional_mask exists (transformers >= 5.0)
try:
    from transformers.masking_utils import create_bidirectional_mask

    _HAS_NATIVE_BIDIRECTIONAL_MASK = True
except ImportError:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    _HAS_NATIVE_BIDIRECTIONAL_MASK = False

# Detect API differences via introspection
_decoder_forward_params = inspect.signature(LlamaDecoderLayer.forward).parameters
_dynamic_cache_init_params = inspect.signature(DynamicCache.__init__).parameters

# past_key_value (singular) in < 4.56, past_key_values (plural) in >= 4.56
_USE_PLURAL_CACHE_PARAM = "past_key_values" in _decoder_forward_params
# DynamicCache accepts config parameter in >= 4.56
_DYNAMIC_CACHE_ACCEPTS_CONFIG = "config" in _dynamic_cache_init_params


def _create_bidirectional_attention_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    bidirectional_mask: torch.Tensor | None = None,
    bidirectional_mask_precomputed: bool = False,
) -> torch.Tensor | None:
    if bidirectional_mask_precomputed:
        return bidirectional_mask
    if attention_mask is None:
        return None

    if _HAS_NATIVE_BIDIRECTIONAL_MASK:
        return create_bidirectional_mask(
            config,
            input_embeds,
            attention_mask=attention_mask,
        )

    if getattr(config, "_attn_implementation", None) == "flash_attention_2":
        has_masked_tokens = (attention_mask == 0).any()
        return attention_mask if has_masked_tokens else None

    return _prepare_4d_attention_mask(attention_mask, input_embeds.dtype)


def split_model(model_path, device):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    print("world_size", world_size)
    num_layers_per_gpu_ = math.floor(num_layers / (world_size - 1))
    num_layers_per_gpu = [num_layers_per_gpu_] * world_size
    num_layers_per_gpu[device] = num_layers - num_layers_per_gpu_ * (world_size - 1)
    print(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = device
    device_map["mlp1"] = device
    device_map["language_model.model.tok_embeddings"] = device
    device_map["language_model.model.embed_tokens"] = device
    device_map["language_model.output"] = device
    device_map["language_model.model.norm"] = device
    device_map["language_model.lm_head"] = device
    device_map["language_model.model.rotary_emb"] = device
    device_map[f"language_model.model.layers.{num_layers - 1}"] = device
    return device_map


def pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str) -> torch.Tensor:
    """Pool token-level hidden states into sequence embeddings using the given strategy."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weighted_avg":
        emb = last_hidden.sum(dim=1)
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    elif pool_type == "cls_last":
        emb = last_hidden[:, 0]
    elif pool_type == "colbert":
        emb = last_hidden
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def _replace_image_token_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
    img_context_token_id: int,
    image_token_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace image placeholder token embeddings with vision embeddings."""
    batch_size, seq_len, hidden_size = input_embeds.shape
    flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
    if image_token_indices is None:
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected_indices = (flat_input_ids == img_context_token_id).nonzero(as_tuple=False).squeeze(1)
    else:
        selected_indices = image_token_indices.reshape(-1).to(device=flat_embeds.device, dtype=torch.long)
    vit_embeds = vit_embeds.reshape(-1, hidden_size).to(dtype=flat_embeds.dtype)

    n_token = selected_indices.numel()
    if n_token != vit_embeds.shape[0]:
        logger.warning(
            "image token count mismatch: selected=%s, vit_embeds=%s; truncating vision embeddings",
            n_token,
            vit_embeds.shape,
        )
        vit_embeds = vit_embeds[:n_token]

    flat_embeds = flat_embeds.index_copy(0, selected_indices, vit_embeds)
    return flat_embeds.reshape(batch_size, seq_len, hidden_size)


def _filter_vision_embeddings_by_image_flags(
    vit_embeds: torch.Tensor,
    image_flags: Optional[torch.Tensor],
) -> torch.Tensor:
    """Keep only vision embeddings marked as real images."""
    if image_flags is None or isinstance(image_flags, list):
        return vit_embeds

    image_flags = image_flags.squeeze(-1)
    return vit_embeds[image_flags == 1]


# ============================================================================
# Bidirectional LLaMA Model
# ============================================================================


class LlamaBidirectionalModel(HFLlamaModel):
    """
    LlamaModel modified to use bidirectional (non-causal) attention.
    Supports transformers 4.44+ through 5.x with a unified forward() implementation.
    See https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2 for version notes.
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaBidirectionalConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def _create_bidirectional_mask(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        bidirectional_mask: torch.Tensor | None = None,
        bidirectional_mask_precomputed: bool = False,
    ) -> torch.Tensor | None:
        return _create_bidirectional_attention_mask(
            self.config,
            input_embeds,
            attention_mask,
            bidirectional_mask,
            bidirectional_mask_precomputed,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        bidirectional_mask: torch.Tensor | None = None,
        bidirectional_mask_precomputed: bool = False,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            if _DYNAMIC_CACHE_ACCEPTS_CONFIG:
                past_key_values = DynamicCache(config=self.config)
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        bidirectional_mask = self._create_bidirectional_mask(
            inputs_embeds,
            attention_mask,
            bidirectional_mask,
            bidirectional_mask_precomputed,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        # Build decoder layer kwargs with correct cache parameter name
        layer_kwargs = {
            "attention_mask": bidirectional_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        if _USE_PLURAL_CACHE_PARAM:
            layer_kwargs["past_key_values"] = past_key_values
        else:
            layer_kwargs["past_key_value"] = past_key_values

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(hidden_states, **layer_kwargs)

            # Decoder returns tuple in < 4.54, tensor in >= 4.54
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class OptimizedLlamaBidirectionalModel(PreTrainedModel):
    """Self-contained optimized LLaMA stack for Nemotron VL retrieval only."""

    config_class = LlamaBidirectionalConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OptimizedLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True  # transformers < 4.54
    _supports_flash_attn = True  # transformers >= 4.54
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OptimizedLlamaDecoderLayer,
        "attentions": OptimizedLlamaAttention,
    }

    def __init__(self, config: LlamaBidirectionalConfig, backend: BackendConfig | None = None):
        super().__init__(config)
        backend = backend or BackendConfig()
        self.backend = backend
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                OptimizedLlamaDecoderLayer(config=config, layer_idx=layer_idx, backend=backend)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.rotary_emb = OptimizedLlamaRotaryEmbedding(config=config, rope_fusion=backend.rope_fusion)
        self.gradient_checkpointing = False
        self.post_init()

    def _create_bidirectional_mask(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        bidirectional_mask: torch.Tensor | None = None,
        bidirectional_mask_precomputed: bool = False,
    ) -> torch.Tensor | None:
        return _create_bidirectional_attention_mask(
            self.config,
            input_embeds,
            attention_mask,
            bidirectional_mask,
            bidirectional_mask_precomputed,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        bidirectional_mask: torch.Tensor | None = None,
        bidirectional_mask_precomputed: bool = False,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Run the optimized bidirectional LLaMA stack.

        Args:
            input_ids: Optional token IDs of shape [batch, sequence]. Mutually
                exclusive with ``inputs_embeds``.
            attention_mask: Optional padding mask of shape [batch, sequence].
            position_ids: Optional absolute position IDs of shape [batch, sequence].
            past_key_values: Optional cache whose per-layer key and value tensors
                have shape [batch, key_value_heads, cached_sequence, head_dim].
            inputs_embeds: Optional token embeddings of shape [batch, sequence,
                hidden]. Mutually exclusive with ``input_ids``.
            cache_position: Optional tensor of shape [sequence] containing absolute
                positions for cache updates.
            bidirectional_mask: Optional precomputed mask of shape [batch, sequence]
                for FlashAttention 2, or [batch, 1, sequence, key_value_sequence]
                for eager and SDPA attention.
            bidirectional_mask_precomputed: Whether ``bidirectional_mask`` is ready
                for the selected attention backend.
            use_cache: Whether to create or update ``past_key_values``.
            output_hidden_states: Whether to return the embedding output and every
                decoder-layer output.
            return_dict: Whether to return ``BaseModelOutputWithPast`` instead of a tuple.
            **kwargs: Additional decoder-layer arguments. Tensor-valued entries use
                the layouts required by the selected attention backend.

        Returns:
            ``BaseModelOutputWithPast`` whose ``last_hidden_state`` has shape
            [batch, sequence, hidden], whose cached key and value tensors have shape
            [batch, key_value_heads, cached_sequence, head_dim], and whose optional
            ``hidden_states`` entries each have shape [batch, sequence, hidden]. If
            ``return_dict`` is false, these populated fields are returned as a tuple
            in the same order.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            if _DYNAMIC_CACHE_ACCEPTS_CONFIG:
                past_key_values = DynamicCache(config=self.config)
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        bidirectional_mask = self._create_bidirectional_mask(
            inputs_embeds,
            attention_mask,
            bidirectional_mask,
            bidirectional_mask_precomputed,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=bidirectional_mask,
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
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


def replace_language_model_with_custom_llama(
    model: nn.Module,
    *,
    backend: BackendConfig | None = None,
) -> bool:
    """Replace a loaded LlamaNemotronVL language tower with the local optimized LLaMA stack."""
    if not isinstance(model, LlamaNemotronVLModel):
        return False

    old_language_model = model.language_model
    if not isinstance(old_language_model, LlamaBidirectionalModel):
        logger.info(
            "Skipping custom LLaMA language replacement: expected LlamaBidirectionalModel, got %s",
            type(old_language_model).__name__,
        )
        return False

    parameter = next(old_language_model.parameters(), None)
    buffer = next(old_language_model.buffers(), None)
    device = parameter.device if parameter is not None else buffer.device if buffer is not None else None
    dtype = parameter.dtype if parameter is not None else buffer.dtype if buffer is not None else None

    new_language_model = OptimizedLlamaBidirectionalModel(old_language_model.config, backend=backend)
    # assign=True adopts the old tower's tensors in place of the freshly
    # initialized ones, so the swap never holds a second copy of the weights
    # on device; the .to() below only moves buffers absent from the state dict.
    missing, unexpected = new_language_model.load_state_dict(old_language_model.state_dict(), strict=False, assign=True)
    if device is not None:
        if dtype is not None and dtype.is_floating_point:
            new_language_model = new_language_model.to(device=device, dtype=dtype)
        else:
            new_language_model = new_language_model.to(device=device)
    extra_state_missing = [key for key in missing if key.endswith("._extra_state")]
    real_missing = [key for key in missing if key not in extra_state_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            "Custom LLaMA language replacement state dict mismatch: "
            f"missing={real_missing[:8]} unexpected={list(unexpected)[:8]}"
        )

    model.language_model = new_language_model
    model._nemo_use_custom_llama_backend = True
    logger.info(
        "Replaced LlamaNemotronVL language model with local optimized LLaMA stack "
        "(backend=%s, ignored_te_extra_state=%d)",
        new_language_model.backend,
        len(extra_state_missing),
    )
    return True


@dataclass(frozen=True, kw_only=True)
class LlamaNemotronVLRetrievalOptimizationConfig:
    """Model-owned optimizations for Llama Nemotron VL retrieval training.

    Attributes:
        use_custom_llama_backend: Replace the Hugging Face Llama tower with the optimized local implementation.
        use_te_fused_mlp: Fuse Llama RMSNorm and MLP projections with Transformer Engine.
        use_te_fused_qkv: Fuse Llama RMSNorm and QKV projections with Transformer Engine.
        use_te_fused_siglip_layer: Replace SigLIP encoder layers with Transformer Engine layers.
        disable_unused_siglip_pooling_head: Disable gradients and execution for the unused SigLIP pooling head.
    """

    use_custom_llama_backend: bool = False
    use_te_fused_mlp: bool = False
    use_te_fused_qkv: bool = False
    use_te_fused_siglip_layer: bool = False
    disable_unused_siglip_pooling_head: bool = False

    def __post_init__(self) -> None:
        """Validate dependencies between the model-specific optimizations."""
        if self.use_te_fused_mlp and not self.use_custom_llama_backend:
            raise ValueError("use_te_fused_mlp requires use_custom_llama_backend=True")
        if self.use_te_fused_qkv and not self.use_custom_llama_backend:
            raise ValueError("use_te_fused_qkv requires use_custom_llama_backend=True")

    def build(self, *, model: "LlamaNemotronVLModel") -> "LlamaNemotronVLModel":
        """Apply the requested optimizations to a loaded Llama Nemotron VL model.

        Args:
            model: Loaded model to optimize before distributed wrapping and optimizer construction.

        Returns:
            The optimized model.

        Raises:
            TypeError: If ``model`` is not a ``LlamaNemotronVLModel``.
            RuntimeError: If a requested optimization cannot be applied.
        """
        if not isinstance(model, LlamaNemotronVLModel):
            raise TypeError(f"Expected LlamaNemotronVLModel, got {type(model).__name__}")

        if self.use_custom_llama_backend:
            replaced = replace_language_model_with_custom_llama(model)
            if not replaced:
                raise RuntimeError("use_custom_llama_backend requested but the loaded backbone was not replaced")
            if self.use_te_fused_mlp:
                fused = replace_llama_mlp_with_te_fused(model)
                if fused == 0:
                    raise RuntimeError("use_te_fused_mlp requested but no custom LLaMA MLP layers were fused")
            if self.use_te_fused_qkv:
                fused = replace_llama_qkv_with_te_fused(model)
                if fused == 0:
                    raise RuntimeError("use_te_fused_qkv requested but no custom LLaMA QKV layers were fused")

        if self.use_te_fused_siglip_layer:
            replaced = replace_siglip_encoder_layers_with_te_fused(model)
            if replaced == 0:
                raise RuntimeError("use_te_fused_siglip_layer requested but no SigLIP encoder layers were replaced")
        if self.disable_unused_siglip_pooling_head:
            disabled = disable_unused_siglip_pooling_head_grad(model)
            if disabled == 0:
                raise RuntimeError(
                    "disable_unused_siglip_pooling_head requested but no SigLIP pooling-head parameters were disabled"
                )

        return model


# ============================================================================
# LlamaNemotronVL Model Classes
# ============================================================================


class LlamaNemotronVLModel(PreTrainedModel):
    """
    LlamaNemotron VL model for vision-language reranking.
    Combines a vision encoder (SigLIP) with a bidirectional language model (LLaMA)
    for cross-modal reranking tasks.
    """

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    config_class = LlamaNemotronVLConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["LlamaDecoderLayer"]
    _supports_flash_attn_2 = True  # transformers < 4.54
    _supports_flash_attn = True  # transformers >= 4.54
    _supports_sdpa = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        optimization_config: Optional[LlamaNemotronVLRetrievalOptimizationConfig] = None,
        **kwargs: Any,
    ) -> "LlamaNemotronVLModel":
        """Load the model and apply model-specific retrieval optimizations.

        Args:
            pretrained_model_name_or_path: Hugging Face model identifier or local checkpoint path.
            *model_args: Positional arguments forwarded to Hugging Face model loading.
            optimization_config: Optional typed configuration for model-specific optimizations.
            **kwargs: Keyword arguments forwarded to Hugging Face model loading.

        Returns:
            The loaded model, with requested optimizations applied before it is returned.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if optimization_config is not None:
            model = optimization_config.build(model=model)
        return model

    def __init__(
        self,
        config: LlamaNemotronVLConfig,
        vision_model: Optional[PreTrainedModel] = None,
        language_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)

        # Propagate attn_implementation to sub-configs for transformers < 4.56
        # which lacks set_attn_implementation. In 4.56+, set_attn_implementation
        # handles this automatically using the sub_configs declared on the config class.
        if not hasattr(PreTrainedModel, "set_attn_implementation"):
            parent_attn = getattr(config, "_attn_implementation", None)
            if parent_attn is not None:
                for sub_config in (config.vision_config, config.llm_config):
                    if getattr(sub_config, "_attn_implementation_autoset", False):
                        sub_config._attn_implementation = parent_attn
                        sub_config._attn_implementation_autoset = False

        # Calculate image token count
        image_size = config.force_image_size or config.vision_config.image_size
        if hasattr(config.vision_config, "grid_size"):
            grid_size = config.vision_config.grid_size
            self.patch_size = 14
            self.num_image_token = int((grid_size * config.downsample_ratio) ** 2)
        else:
            patch_size = config.vision_config.patch_size
            self.patch_size = patch_size
            self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))

        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio

        logger.info(f"num_image_token: {self.num_image_token}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == "siglip_vision_model":
                self.vision_model = SiglipVisionModel(config.vision_config)
            else:
                raise NotImplementedError(f"Unsupported vision model type: {config.vision_config.model_type}")

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaBidirectionalModel":
                self.language_model = LlamaBidirectionalModel(config.llm_config)
            else:
                raise NotImplementedError(f"{config.llm_config.architectures[0]} is not implemented.")

        # Vision-to-language projection
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                llm_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.img_context_token_id = None

        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(config.name_or_path, trust_remote_code=True)

        self.post_init()

        # transformers 4.54.0-4.55.x have a bug in the flash attention
        # infrastructure where a parameter name mismatch ('implementation' vs
        # 'attn_implementation') in _flash_attention_forward causes FA3 pad/unpad
        # functions to be used with FA2 attention kernels, producing nondeterministic
        # and numerically different results for batched vision+text sequences.
        # This was fixed in 4.56.0.
        from packaging.version import Version

        _tv = Version(transformers.__version__)
        if Version("4.54.0") <= _tv < Version("4.56.0"):
            raise RuntimeError(
                f"transformers {transformers.__version__} is not supported by this model. "
                f"Versions 4.54.0-4.55.x have a flash attention bug that produces "
                f"nondeterministic and incorrect image embeddings. "
                f"Please use transformers <=4.53.x or >=4.56.0."
            )

    def get_encoder_state_dict_adapter(self):
        return LlamaNemotronVLEncoderStateDictAdapter(self)

    def _embed_batch(self, inputs: Dict[str, Any], pool_type: Optional[str] = None):
        """
        Encodes the inputs into a tensor of embeddings.
        Args:
            inputs: A dictionary of inputs to the model. You can prepare the inputs using the processor.process_queries and processor.process_documents methods.
            pool_type: The type of pooling to use. If None, the pooling type is set to the pooling type configured in the model.
        Returns:
            A tensor of embeddings.
        """
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        outputs = self.forward(**inputs, output_hidden_states=True, return_dict=True)
        if not pool_type:
            pool_type = self.config.pooling
        embeddings = pool(
            last_hidden_states=outputs.hidden_states[-1], attention_mask=inputs["attention_mask"], pool_type=pool_type
        )
        return embeddings

    def encode_queries(self, queries: List[str], **kwargs):
        """
        Encodes the input queries into a tensor of embeddings.
        Args:
            queries: A list of queries.
        Returns:
            A tensor of embeddings.
        """
        queries_dict = self.processor.process_queries(queries)
        queries_embeddings = self._embed_batch(inputs=queries_dict, **kwargs)
        return queries_embeddings

    def encode_documents(self, images: Optional[List[Any]] = None, texts: Optional[List[str]] = None, **kwargs):
        """
        Encodes the input document images and texts into a tensor of embeddings.
        Args:
            images: A list of PIL.Image of document pages images.
            texts: A list of document page texts.
        Returns:
            A tensor of embeddings.
        """
        if images and texts:
            examples = [{"image": image, "text": doc_text} for image, doc_text in zip(images, texts)]

        elif images:
            examples = [{"image": image, "text": ""} for image in images]

        elif texts:
            examples = [{"image": "", "text": doc_text} for doc_text in texts]
        else:
            raise ValueError("At least docs_images or docs_texts need to be provided")

        docs_dict = self.processor.process_documents(examples)
        docs_embeddings = self._embed_batch(inputs=docs_dict, **kwargs)
        return docs_embeddings

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_patches_list: Optional[List[torch.Tensor]] = None,
        run_dummy_vision: Optional[bool] = None,
        image_token_indices: torch.LongTensor | None = None,
        bidirectional_mask: torch.Tensor | None = None,
        bidirectional_mask_precomputed: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get text embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Process and inject vision embeddings if present
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values).to(device=input_embeds.device)
            vit_embeds = _filter_vision_embeddings_by_image_flags(vit_embeds, image_flags)

            input_embeds = _replace_image_token_embeddings(
                input_embeds,
                input_ids,
                vit_embeds,
                self.config.img_context_token_id,
                image_token_indices=image_token_indices,
            )

        elif self.training and run_dummy_vision is not False:
            # If there is no image in the batch, adds a dummy image to the batch
            # to ensure multi-GPU synchronization when there are batches with only text samples and others with image samples
            image_size = self.config.force_image_size or self.config.vision_config.image_size
            dtype = next(self.vision_model.parameters()).dtype
            dummy_pixels = torch.zeros(1, 3, image_size, image_size, device=input_embeds.device, dtype=dtype)
            dummy_output = self.extract_feature(dummy_pixels)
            input_embeds = input_embeds + dummy_output.sum().to(input_embeds.dtype) * 0.0

        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            bidirectional_mask=bidirectional_mask,
            bidirectional_mask_precomputed=bidirectional_mask_precomputed,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = None
        loss = None

        if hasattr(outputs, "logits"):
            logits = outputs.logits
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        hidden_states = outputs.hidden_states
        if hidden_states is None and hasattr(outputs, "last_hidden_state"):
            hidden_states = (outputs.last_hidden_state,)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.shape
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """Extract and project vision features to language model space."""
        # Extract features from vision encoder
        if self.select_layer == -1:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
            if hasattr(vit_embeds, "last_hidden_state"):
                vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]

        # Remove CLS token if not using SigLIP
        if not isinstance(self.vision_model, SiglipVisionModel):
            vit_embeds = vit_embeds[:, 1:, :]

        # Apply pixel shuffle and MLP projection
        _, n, c = vit_embeds.shape
        h = w = int(n**0.5)
        vit_embeds = vit_embeds.reshape(-1, h, w, c)  # (B, H, W, C)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)  # (B, H/s, W/s, C*s*s)
        _, h_s, w_s, c_s = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(-1, h_s * w_s, c_s)  # (B, (H/s)*(W/s), C*s*s)
        vit_embeds = self.mlp1(vit_embeds)

        return vit_embeds

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def build_collator(self, processor=None, **kwargs):
        return processor or self.processor

    def post_loss(self, loss, inputs):
        # Add Dummy Gradients for Vision Encoder to ensure multi-GPU synchronization when there are batches with only text samples
        # and other batches with images.
        if "pixel_values" in inputs and inputs["pixel_values"] is None:
            dummy_pixels = torch.zeros(1, 3, 512, 512, device=loss.device, dtype=self.vision_model.dtype)
            dummy_output = self.extract_feature(dummy_pixels)
            loss = loss + dummy_output.sum() * 0.0
        return loss


# Export for ModelRegistry auto-discovery
ModelClass = [LlamaNemotronVLModel]


def _register_with_hf_auto_classes():
    """Register bidirectional models with HuggingFace Auto classes.

    This is needed so that AutoModel.from_config(LlamaBidirectionalConfig)
    works inside LlamaForSequenceClassification.__init__.
    """
    from transformers import AutoConfig, AutoModel

    try:
        AutoConfig.register(LlamaNemotronVLConfig.model_type, LlamaNemotronVLConfig)
    except ValueError:
        pass  # Already registered
    try:
        AutoModel.register(LlamaNemotronVLConfig, LlamaNemotronVLModel)
    except ValueError:
        pass  # Already registered


_register_with_hf_auto_classes()

__all__ = [
    "FusedTESiglipEncoderLayer",
    "FusedTESiglipTransformerLayer",
    "LlamaBidirectionalConfig",
    "LlamaBidirectionalModel",
    "LlamaNemotronVLConfig",
    "LlamaNemotronVLEncoderStateDictAdapter",
    "LlamaNemotronVLModel",
    "LlamaNemotronVLRetrievalOptimizationConfig",
    "ModelClass",
    "OptimizedFusedTERMSNormMLP",
    "OptimizedFusedTERMSNormQKV",
    "OptimizedLlamaAttention",
    "OptimizedLlamaBidirectionalModel",
    "OptimizedLlamaDecoderLayer",
    "OptimizedLlamaMLP",
    "disable_unused_siglip_pooling_head_grad",
    "pool",
    "replace_language_model_with_custom_llama",
    "replace_llama_mlp_with_te_fused",
    "replace_llama_qkv_with_te_fused",
    "replace_siglip_encoder_layers_with_te_fused",
]
