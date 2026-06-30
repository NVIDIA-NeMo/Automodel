# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoProcessor, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    eager_attention_forward,
)
from transformers.models.llama.modeling_llama import (
    LlamaModel as HFLlamaModel,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

logger = logging.get_logger(__name__)


def _delete_module_attr(module: nn.Module, name: str) -> None:
    if hasattr(module, name):
        delattr(module, name)


# ============================================================================
# Bidirectional LLaMA Configuration
# ============================================================================


class LlamaBidirectionalConfig(LlamaConfig):
    """Configuration for bidirectional (non-causal) LLaMA model."""

    model_type = "llama_bidirec"

    def __init__(
        self,
        pooling="avg",
        temperature=1.0,
        **kwargs,
    ):
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(
            **kwargs,
        )


# ============================================================================
# LlamaNemotronVL Configuration Classes
# ============================================================================


class LlamaNemotronVLConfig(PretrainedConfig):
    """
    Base configuration for vision-language models combining vision and language components.
    This serves as the foundation for LlamaNemotronVL configurations.
    """

    model_type = "llama_nemotron_vl"
    is_composition = True
    # is_composition was renamed to has_no_defaults_at_init in transformers 4.52.1
    # In PR https://github.com/huggingface/transformers/pull/36263
    has_no_defaults_at_init = True
    # Declare sub-configs so transformers can propagate per-backbone attn_implementation
    # e.g. from_pretrained(attn_implementation={"vision_config": "sdpa", "llm_config": "flash_attention_2"})
    sub_configs = {"vision_config": SiglipVisionConfig, "llm_config": LlamaBidirectionalConfig}

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        mlp_checkpoint=True,
        pre_feature_reduction=False,
        keep_aspect_ratio=False,
        vocab_size=-1,
        q_max_length: Optional[int] = 512,
        p_max_length: Optional[int] = 10240,
        query_prefix: str = "query:",
        passage_prefix: str = "passage:",
        pooling: str = "last",
        bidirectional_attention: bool = False,
        max_input_tiles: int = 2,
        img_context_token_id: int = 128258,  # tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        **kwargs,
    ):
        if vision_config is not None:
            if vision_config["model_type"] == "siglip_vision_model":
                self.vision_config = SiglipVisionConfig(**vision_config)
            else:
                raise ValueError("Unsupported model_type: {}".format(vision_config["model_type"]))

        if llm_config is not None:
            if llm_config["architectures"][0] in {
                "LlamaBidirectionalModel",
                "LlamaBidirectionalForSequenceClassification",
            }:
                self.llm_config = LlamaBidirectionalConfig(**llm_config)
            else:
                raise ValueError("Unsupported architecture: {}".format(llm_config["architectures"][0]))
            self.vocab_size = self.llm_config.vocab_size
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.mlp_checkpoint = mlp_checkpoint
        self.pre_feature_reduction = pre_feature_reduction
        self.keep_aspect_ratio = keep_aspect_ratio

        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.pooling = pooling
        self.bidirectional_attention = bidirectional_attention
        self.img_context_token_id = img_context_token_id
        self.max_input_tiles = max_input_tiles
        super().__init__(**kwargs)


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
    image_token_indices: Optional[torch.Tensor] = None,
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
        if bidirectional_mask_precomputed:
            return bidirectional_mask
        if attention_mask is None:
            return None

        if _HAS_NATIVE_BIDIRECTIONAL_MASK:
            return create_bidirectional_mask(
                self.config,
                input_embeds,
                attention_mask=attention_mask,
            )

        # Fallback for transformers < 5.0
        if getattr(self.config, "_attn_implementation", None) == "flash_attention_2":
            has_masked_tokens = (attention_mask == 0).any()
            return attention_mask if has_masked_tokens else None

        return _prepare_4d_attention_mask(attention_mask, input_embeds.dtype)

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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _apply_rotary_pos_emb_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb as te_apply_rope

    q_bshd = q.permute(0, 2, 1, 3)
    k_bshd = k.permute(0, 2, 1, 3)
    q_bshd = te_apply_rope(q_bshd, freqs_cis, tensor_format="bshd", fused=True)
    k_bshd = te_apply_rope(k_bshd, freqs_cis, tensor_format="bshd", fused=True)
    return q_bshd.permute(0, 2, 1, 3), k_bshd.permute(0, 2, 1, 3)


def _get_rope_config(config) -> tuple[float, dict]:
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        rope_params = config.rope_parameters
        base = rope_params.get("rope_theta", 10000.0)
        rope_scaling = rope_params
    else:
        base = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", {}) or {}
    return base, rope_scaling


def _compute_default_inv_freq(config, device: torch.device | None = None) -> tuple[torch.Tensor, float]:
    base, _ = _get_rope_config(config)
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
    return inv_freq, 1.0


def _compute_llama3_inv_freq(config, device: torch.device | None = None) -> tuple[torch.Tensor, float]:
    inv_freq, _ = _compute_default_inv_freq(config, device)
    _, rope_scaling = _get_rope_config(config)
    factor = rope_scaling.get("factor", 1.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings", config.max_position_embeddings)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama, 1.0


class OptimizedLlamaRotaryEmbedding(nn.Module):
    """Local RoPE helper for the optimized VL-only LLaMA backend."""

    inv_freq: torch.Tensor

    def __init__(self, config, device: torch.device | None = None, rope_fusion: bool = False):
        super().__init__()
        self.max_seq_len_cached = 0
        self.rope_fusion = rope_fusion
        self.dtype = getattr(config, "torch_dtype", None) or torch.float32

        rope_functions = {
            "default": _compute_default_inv_freq,
            "llama3": _compute_llama3_inv_freq,
        }
        _, rope_scaling = _get_rope_config(config)
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        compute_fn = rope_functions.get(rope_type)
        if compute_fn is None:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            compute_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
            if compute_fn is None:
                supported = sorted(set(rope_functions) | set(ROPE_INIT_FUNCTIONS))
                raise ValueError(f"Unsupported RoPE rope_type={rope_type!r}; expected one of {supported}.")

        inv_freq, self.attention_scaling = compute_fn(config, device)
        self._rope_config = config
        self._inv_freq_compute_fn = compute_fn
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cache", None, persistent=False)
        self.register_buffer("_sin_cache", None, persistent=False)
        self.register_buffer("_freqs_cache", None, persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        self.max_seq_len_cached = seq_len
        inv_freq, attention_scaling = self._inv_freq_compute_fn(self._rope_config, device)
        self.attention_scaling = attention_scaling
        self.inv_freq = inv_freq
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        self._cos_cache = cos.to(dtype=self.dtype)
        self._sin_cache = sin.to(dtype=self.dtype)
        if self.rope_fusion:
            self._freqs_cache = emb.to(dtype=self.dtype).unsqueeze(1).unsqueeze(1).contiguous()

    def _ensure_cache(self, seq_len: int, device: torch.device) -> None:
        if self._cos_cache is None or seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_fusion:
            seq_len = position_ids.shape[-1]
            self._ensure_cache(seq_len, x.device)
            cos = self._cos_cache[:seq_len].unsqueeze(0).expand(position_ids.shape[0], -1, -1)
            sin = self._sin_cache[:seq_len].unsqueeze(0).expand(position_ids.shape[0], -1, -1)
            return cos, sin, self._freqs_cache[:seq_len]

        needed = max(int(position_ids.max().item()) + 1, position_ids.shape[-1])
        self._ensure_cache(needed, x.device)
        cos = self._cos_cache[position_ids]
        sin = self._sin_cache[position_ids]
        return cos, sin


class OptimizedLlamaAttention(nn.Module):
    """Local LLaMA attention used only by the optimized Nemotron VL retriever path."""

    def __init__(self, config: LlamaConfig, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.rope_fusion = getattr(backend, "rope_fusion", False)
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.q_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.k_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.v_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            backend.linear,
            self.num_key_value_groups * config.num_key_value_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.pre_attention_qkv: nn.Module | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.pre_attention_qkv is not None:
            q, k, v = self.pre_attention_qkv(hidden_states)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings[:2]
        if self.rope_fusion and len(position_embeddings) == 3:
            query_states, key_states = _apply_rotary_pos_emb_fused(query_states, key_states, position_embeddings[2])
        else:
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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


class OptimizedLlamaMLP(nn.Module):
    """Local SwiGLU MLP for the optimized VL-only LLaMA backend."""

    def __init__(self, config: LlamaConfig, backend: BackendConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.gate_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=dtype
        )
        self.up_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=dtype
        )
        self.down_proj = initialize_linear_module(
            backend.linear, self.intermediate_size, self.hidden_size, bias=config.mlp_bias, dtype=dtype
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class OptimizedFusedTERMSNormMLP(nn.Module):
    """Transformer Engine RMSNorm + SwiGLU replacement for the local optimized LLaMA MLP."""

    def __init__(self, source_norm: nn.Module, source_mlp: OptimizedLlamaMLP, eps: float):
        super().__init__()
        from transformer_engine.pytorch import LayerNormMLP

        norm_weight = source_norm.weight
        gate_proj = source_mlp.gate_proj
        up_proj = source_mlp.up_proj
        down_proj = source_mlp.down_proj
        gate_bias = getattr(gate_proj, "bias", None)
        use_bias = gate_bias is not None and gate_bias.numel() > 0
        self.fused = LayerNormMLP(
            hidden_size=source_mlp.hidden_size,
            ffn_hidden_size=source_mlp.intermediate_size,
            eps=eps,
            normalization="RMSNorm",
            activation="swiglu",
            bias=use_bias,
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )
        with torch.no_grad():
            self.fused.layer_norm_weight.copy_(norm_weight)
            self.fused.fc1_weight.copy_(torch.cat([gate_proj.weight, up_proj.weight], dim=0))
            self.fused.fc2_weight.copy_(down_proj.weight)
            if use_bias:
                self.fused.fc1_bias.copy_(torch.cat([gate_proj.bias, up_proj.bias], dim=0))
                self.fused.fc2_bias.copy_(down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fused(x)


class OptimizedFusedTERMSNormQKV(nn.Module):
    """Transformer Engine RMSNorm + concatenated QKV projection for the local optimized LLaMA attention."""

    def __init__(self, source_norm: nn.Module, source_attn: OptimizedLlamaAttention, eps: float):
        super().__init__()
        from transformer_engine.pytorch import LayerNormLinear

        norm_weight = source_norm.weight
        q_proj = source_attn.q_proj
        k_proj = source_attn.k_proj
        v_proj = source_attn.v_proj
        q_size = q_proj.out_features
        k_size = k_proj.out_features
        v_size = v_proj.out_features
        q_bias = getattr(q_proj, "bias", None)
        use_bias = q_bias is not None and q_bias.numel() > 0
        self.sizes = (q_size, k_size, v_size)
        self.fused = LayerNormLinear(
            in_features=source_attn.config.hidden_size,
            out_features=q_size + k_size + v_size,
            eps=eps,
            normalization="RMSNorm",
            bias=use_bias,
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )
        with torch.no_grad():
            self.fused.layer_norm_weight.copy_(norm_weight)
            self.fused.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
            if use_bias:
                self.fused.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.fused(x)
        return qkv.split(self.sizes, dim=-1)


class OptimizedLlamaDecoderLayer(nn.Module):
    """Local LLaMA decoder layer used only by the optimized Nemotron VL retriever path."""

    def __init__(self, config: LlamaConfig, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OptimizedLlamaAttention(config=config, layer_idx=layer_idx, backend=backend)
        self.mlp = OptimizedLlamaMLP(config=config, backend=backend)
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if getattr(self, "self_attn", None) is None:
            return self.mlp(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        residual = hidden_states
        if getattr(self, "input_layernorm", None) is not None:
            hidden_states = self.input_layernorm(hidden_states)
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

        residual = hidden_states
        if getattr(self, "post_attention_layernorm", None) is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


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
        backend = backend or BackendConfig(rms_norm="te", rope_fusion=True)
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
        if bidirectional_mask_precomputed:
            return bidirectional_mask
        if attention_mask is None:
            return None

        if _HAS_NATIVE_BIDIRECTIONAL_MASK:
            return create_bidirectional_mask(
                self.config,
                input_embeds,
                attention_mask=attention_mask,
            )

        if getattr(self.config, "_attn_implementation", None) == "flash_attention_2":
            has_masked_tokens = (attention_mask == 0).any()
            return attention_mask if has_masked_tokens else None

        return _prepare_4d_attention_mask(attention_mask, input_embeds.dtype)

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
    if device is not None:
        if dtype is not None and dtype.is_floating_point:
            new_language_model = new_language_model.to(device=device, dtype=dtype)
        else:
            new_language_model = new_language_model.to(device=device)

    missing, unexpected = new_language_model.load_state_dict(old_language_model.state_dict(), strict=False)
    extra_state_missing = [key for key in missing if key.endswith("._extra_state")]
    real_missing = [key for key in missing if key not in extra_state_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            "Custom LLaMA language replacement state dict mismatch: "
            f"missing={real_missing[:8]} unexpected={list(unexpected)[:8]}"
        )

    model.language_model = new_language_model
    logger.info(
        "Replaced LlamaNemotronVL language model with local optimized LLaMA stack "
        "(backend=%s, ignored_te_extra_state=%d)",
        new_language_model.backend,
        len(extra_state_missing),
    )
    return True


def replace_llama_mlp_with_te_fused(model: nn.Module) -> int:
    """Fuse post-attention RMSNorm and SwiGLU in the custom LLaMA stack."""
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return 0

    fused_count = 0
    for layer in layers:
        norm = getattr(layer, "post_attention_layernorm", None)
        mlp = getattr(layer, "mlp", None)
        if norm is None or mlp is None or isinstance(mlp, OptimizedFusedTERMSNormMLP):
            continue
        if not hasattr(mlp, "gate_proj") or not hasattr(mlp, "up_proj") or not hasattr(mlp, "down_proj"):
            continue

        fused = OptimizedFusedTERMSNormMLP(norm, mlp, eps=language_model.config.rms_norm_eps)
        _delete_module_attr(layer, "post_attention_layernorm")
        layer.mlp = fused
        fused_count += 1

    logger.info("Fused custom LLaMA post-attention RMSNorm + SwiGLU layers: %d", fused_count)
    return fused_count


def replace_llama_qkv_with_te_fused(model: nn.Module) -> int:
    """Fuse custom LLaMA input RMSNorm and Q/K/V projections with TE."""
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return 0

    fused_count = 0
    for layer in layers:
        norm = getattr(layer, "input_layernorm", None)
        attention = getattr(layer, "self_attn", None)
        if norm is None or attention is None or getattr(attention, "pre_attention_qkv", None) is not None:
            continue
        if not all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue

        fused = OptimizedFusedTERMSNormQKV(norm, attention, eps=language_model.config.rms_norm_eps)
        _delete_module_attr(layer, "input_layernorm")
        attention.pre_attention_qkv = fused
        attention.q_proj = None
        attention.k_proj = None
        attention.v_proj = None
        fused_count += 1

    logger.info("Fused custom LLaMA input RMSNorm + QKV layers: %d", fused_count)
    return fused_count


def replace_llama_layers_with_te_fused(model: nn.Module) -> int:
    """Replace custom bidirectional LLaMA layers with TE transformer layers."""
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return 0

    fused_count = 0
    for index, layer in enumerate(layers):
        if getattr(layer, "self_attn", None) is None and isinstance(
            getattr(layer, "mlp", None), FusedTELlamaTransformerLayer
        ):
            continue
        required = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        if not all(getattr(layer, name, None) is not None for name in required):
            continue
        fused = FusedTELlamaTransformerLayer(layer, layer_number=index + 1)
        _delete_module_attr(layer, "input_layernorm")
        _delete_module_attr(layer, "self_attn")
        _delete_module_attr(layer, "post_attention_layernorm")
        layer.mlp = fused
        fused_count += 1

    logger.info("Replaced custom LLaMA layers with TE TransformerLayer: %d", fused_count)
    return fused_count


class FusedTESiglipQKV(nn.Module):
    """Transformer Engine LayerNorm + fused QKV projection for SigLIP."""

    def __init__(self, source_norm: nn.Module, source_attn: nn.Module):
        super().__init__()
        from transformer_engine.pytorch import LayerNormLinear

        norm_weight = source_norm.weight
        self.fused = LayerNormLinear(
            in_features=source_attn.embed_dim,
            out_features=3 * source_attn.embed_dim,
            eps=source_norm.eps,
            normalization="LayerNorm",
            bias=True,
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )
        with torch.no_grad():
            self.fused.layer_norm_weight.copy_(norm_weight)
            self.fused.layer_norm_bias.copy_(source_norm.bias)
            self.fused.weight.copy_(
                torch.cat([source_attn.q_proj.weight, source_attn.k_proj.weight, source_attn.v_proj.weight], dim=0)
            )
            self.fused.bias.copy_(
                torch.cat([source_attn.q_proj.bias, source_attn.k_proj.bias, source_attn.v_proj.bias], dim=0)
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fused(hidden_states)


class FusedQKVSiglipAttention(nn.Module):
    """SigLIP attention with one fused QKV projection and optional TE paths."""

    def __init__(
        self,
        source_attn: nn.Module,
        source_norm: nn.Module | None = None,
        use_te: bool = False,
        use_te_qkv: bool = False,
    ):
        super().__init__()
        self.config = source_attn.config
        self.embed_dim = source_attn.embed_dim
        self.num_heads = source_attn.num_heads
        self.head_dim = source_attn.head_dim
        self.dropout = source_attn.dropout
        device = source_attn.q_proj.weight.device
        dtype = source_attn.q_proj.weight.dtype
        if use_te_qkv:
            if source_norm is None:
                raise ValueError("use_te_qkv requires the SigLIP input LayerNorm")
            self.te_qkv = FusedTESiglipQKV(source_norm, source_attn)
            self.qkv_proj = None
        else:
            self.te_qkv = None
            self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, device=device, dtype=dtype)
            self._load_qkv_from_source(source_attn)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, device=device, dtype=dtype)
        with torch.no_grad():
            self.out_proj.weight.copy_(source_attn.out_proj.weight)
            self.out_proj.bias.copy_(source_attn.out_proj.bias)
        self._use_te = use_te
        if use_te:
            from transformer_engine.pytorch.attention import DotProductAttention

            object.__setattr__(
                self,
                "_te_attention",
                DotProductAttention(
                    num_attention_heads=self.num_heads,
                    kv_channels=(self.head_dim, self.head_dim),
                    attn_mask_type="no_mask",
                    qkv_format="bshd",
                    softmax_scale=self.head_dim**-0.5,
                ),
            )

    def _load_qkv_from_source(self, source_attn: nn.Module) -> None:
        with torch.no_grad():
            self.qkv_proj.weight.copy_(
                torch.cat(
                    [
                        source_attn.q_proj.weight,
                        source_attn.k_proj.weight,
                        source_attn.v_proj.weight,
                    ],
                    dim=0,
                )
            )
            self.qkv_proj.bias.copy_(
                torch.cat(
                    [
                        source_attn.q_proj.bias,
                        source_attn.k_proj.bias,
                        source_attn.v_proj.bias,
                    ],
                    dim=0,
                )
            )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        qkv_projection = self.te_qkv(hidden_states) if self.te_qkv is not None else self.qkv_proj(hidden_states)
        qkv = qkv_projection.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(dim=2)
        if self._use_te:
            with torch.cuda.device(hidden_states.device):
                attn_output = self._te_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask_type="no_mask",
                    window_size=(-1, -1),
                )
            attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class FusedQKVSiglipMLP(nn.Module):
    """SigLIP MLP mirror for local class-name consistency in fused-QKV layers."""

    def __init__(self, source_mlp: nn.Module):
        super().__init__()
        self.config = source_mlp.config
        self.activation_fn = ACT2FN[self.config.hidden_act]
        self.fc1 = source_mlp.fc1
        self.fc2 = source_mlp.fc2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class FusedTESiglipMLP(nn.Module):
    """Transformer Engine LayerNorm + GELU MLP replacement for SigLIP."""

    def __init__(self, source_norm: nn.Module, source_mlp: nn.Module):
        super().__init__()
        from transformer_engine.pytorch import LayerNormMLP

        norm_weight = source_norm.weight
        norm_bias = getattr(source_norm, "bias", None)
        fc1 = source_mlp.fc1
        fc2 = source_mlp.fc2
        fc1_bias = getattr(fc1, "bias", None)
        fc2_bias = getattr(fc2, "bias", None)
        use_bias = fc1_bias is not None and fc2_bias is not None
        self.fused = LayerNormMLP(
            hidden_size=fc1.in_features,
            ffn_hidden_size=fc1.out_features,
            eps=source_norm.eps,
            normalization="LayerNorm",
            activation="gelu",
            bias=use_bias,
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )

        with torch.no_grad():
            self.fused.layer_norm_weight.copy_(norm_weight)
            if norm_bias is not None and hasattr(self.fused, "layer_norm_bias"):
                self.fused.layer_norm_bias.copy_(norm_bias)
            self.fused.fc1_weight.copy_(fc1.weight)
            self.fused.fc2_weight.copy_(fc2.weight)
            if use_bias:
                self.fused.fc1_bias.copy_(fc1_bias)
                self.fused.fc2_bias.copy_(fc2_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fused(hidden_states)


class FusedTEVisionProjection(nn.Module):
    """Fuse the vision-to-language projection's LayerNorm and first Linear."""

    def __init__(self, source_mlp: nn.Sequential):
        super().__init__()
        from transformer_engine.pytorch import LayerNormLinear

        source_norm, source_fc1, source_fc2 = source_mlp[0], source_mlp[1], source_mlp[3]
        self.norm_fc1 = LayerNormLinear(
            in_features=source_fc1.in_features,
            out_features=source_fc1.out_features,
            eps=source_norm.eps,
            normalization="LayerNorm",
            bias=True,
            params_dtype=source_norm.weight.dtype,
            device=source_norm.weight.device,
        )
        self.fc2 = source_fc2
        with torch.no_grad():
            self.norm_fc1.layer_norm_weight.copy_(source_norm.weight)
            self.norm_fc1.layer_norm_bias.copy_(source_norm.bias)
            self.norm_fc1.weight.copy_(source_fc1.weight)
            self.norm_fc1.bias.copy_(source_fc1.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="none")
        return self.fc2(hidden_states)


class FusedTELlamaTransformerLayer(nn.Module):
    """Transformer Engine replacement for one custom bidirectional LLaMA layer."""

    def __init__(self, source_layer: nn.Module, layer_number: int):
        super().__init__()
        from transformer_engine.pytorch import TransformerLayer

        source_attn = source_layer.self_attn
        source_mlp = source_layer.mlp
        source_norm = source_layer.input_layernorm
        source_post_norm = source_layer.post_attention_layernorm
        config = source_attn.config
        q_proj = source_attn.q_proj
        k_proj = source_attn.k_proj
        v_proj = source_attn.v_proj
        dtype = source_norm.weight.dtype
        device = source_norm.weight.device

        self.fused = TransformerLayer(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_key_value_heads,
            kv_channels=source_attn.head_dim,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layer_number=layer_number,
            self_attn_mask_type="arbitrary",
            normalization="RMSNorm",
            activation="swiglu",
            bias=False,
            fuse_qkv_params=True,
            qkv_weight_interleaved=False,
            attn_input_format="bshd",
            params_dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            self.fused.self_attention.layernorm_qkv.layer_norm_weight.copy_(source_norm.weight)
            self.fused.self_attention.layernorm_qkv.weight.copy_(
                torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
            )
            self.fused.self_attention.proj.weight.copy_(source_attn.o_proj.weight)
            self.fused.layernorm_mlp.layer_norm_weight.copy_(source_post_norm.weight)
            self.fused.layernorm_mlp.fc1_weight.copy_(
                torch.cat([source_mlp.gate_proj.weight, source_mlp.up_proj.weight], dim=0)
            )
            self.fused.layernorm_mlp.fc2_weight.copy_(source_mlp.down_proj.weight)

    @staticmethod
    def _normalize_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dtype == torch.bool:
            if attention_mask.ndim == 2:
                return ~attention_mask[:, None, None, :]
            return ~attention_mask
        if attention_mask.ndim == 2:
            return attention_mask == 0
        return attention_mask < 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings=None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.get("past_key_values") is not None or kwargs.get("past_key_value") is not None:
            raise RuntimeError("FusedTELlamaTransformerLayer does not support KV cache updates")
        if position_embeddings is None or len(position_embeddings) != 3:
            raise RuntimeError("FusedTELlamaTransformerLayer requires fused raw RoPE frequencies")
        mask = self._normalize_attention_mask(attention_mask)
        return self.fused(
            hidden_states,
            attention_mask=mask,
            self_attn_mask_type="arbitrary" if mask is not None else "no_mask",
            rotary_pos_emb=position_embeddings[2],
        )


class FusedTESiglipTransformerLayer(nn.Module):
    """Transformer Engine implementation of one complete SigLIP encoder layer."""

    def __init__(self, source_layer: nn.Module):
        super().__init__()
        from transformer_engine.pytorch import TransformerLayer

        source_attn = source_layer.self_attn
        source_norm = source_layer.layer_norm1
        source_mlp = source_layer.mlp
        norm_weight = source_norm.weight
        self.fused = TransformerLayer(
            hidden_size=source_layer.embed_dim,
            ffn_hidden_size=source_mlp.fc1.out_features,
            num_attention_heads=source_attn.num_heads,
            kv_channels=source_attn.head_dim,
            layernorm_epsilon=source_norm.eps,
            hidden_dropout=0.0,
            attention_dropout=source_attn.dropout,
            self_attn_mask_type="no_mask",
            normalization="LayerNorm",
            activation="gelu",
            bias=True,
            fuse_qkv_params=True,
            qkv_weight_interleaved=False,
            attn_input_format="bshd",
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )

        with torch.no_grad():
            fused_qkv = self.fused.self_attention.layernorm_qkv
            fused_mlp = self.fused.layernorm_mlp
            fused_qkv.layer_norm_weight.copy_(source_norm.weight)
            fused_qkv.layer_norm_bias.copy_(source_norm.bias)
            fused_qkv.weight.copy_(
                torch.cat(
                    [source_attn.q_proj.weight, source_attn.k_proj.weight, source_attn.v_proj.weight],
                    dim=0,
                )
            )
            fused_qkv.bias.copy_(
                torch.cat(
                    [source_attn.q_proj.bias, source_attn.k_proj.bias, source_attn.v_proj.bias],
                    dim=0,
                )
            )
            self.fused.self_attention.proj.weight.copy_(source_attn.out_proj.weight)
            self.fused.self_attention.proj.bias.copy_(source_attn.out_proj.bias)
            fused_mlp.layer_norm_weight.copy_(source_layer.layer_norm2.weight)
            fused_mlp.layer_norm_bias.copy_(source_layer.layer_norm2.bias)
            fused_mlp.fc1_weight.copy_(source_mlp.fc1.weight)
            fused_mlp.fc1_bias.copy_(source_mlp.fc1.bias)
            fused_mlp.fc2_weight.copy_(source_mlp.fc2.weight)
            fused_mlp.fc2_bias.copy_(source_mlp.fc2.bias)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fused(
            hidden_states,
            attention_mask=None,
            self_attn_mask_type="no_mask",
            window_size=(-1, -1),
        )


class FusedQKVSiglipEncoderLayer(nn.Module):
    """Stock-compatible SigLIP encoder layer with fused QKV projection."""

    def __init__(
        self,
        source_layer: nn.Module,
        use_te: bool = False,
        use_te_mlp: bool = False,
        use_te_qkv: bool = False,
        use_te_layer: bool = False,
    ):
        super().__init__()
        self.embed_dim = source_layer.embed_dim
        if use_te_layer:
            # Keep the active fused block under a standard child name so the
            # generic DDP/FSDP activation-checkpointing path can wrap it.
            self.mlp = FusedTESiglipTransformerLayer(source_layer)
            return

        self.self_attn = FusedQKVSiglipAttention(
            source_layer.self_attn,
            source_norm=source_layer.layer_norm1,
            use_te=use_te,
            use_te_qkv=use_te_qkv,
        )
        if not use_te_qkv:
            self.layer_norm1 = source_layer.layer_norm1
        if use_te_mlp:
            self.mlp = FusedTESiglipMLP(source_layer.layer_norm2, source_layer.mlp)
        else:
            self.mlp = FusedQKVSiglipMLP(source_layer.mlp)
            self.layer_norm2 = source_layer.layer_norm2

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if getattr(self, "self_attn", None) is None:
            return self.mlp(hidden_states, *args, **kwargs)

        residual = hidden_states
        if getattr(self, "layer_norm1", None) is not None:
            hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        if getattr(self, "layer_norm2", None) is not None:
            hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def replace_siglip_attention_with_fused_qkv(
    model: nn.Module,
    use_te: bool = False,
    use_te_mlp: bool = False,
    use_te_qkv: bool = False,
    use_te_layer: bool = False,
) -> int:
    """Replace loaded HF SigLIP encoder layers with trainability-preserving fused-QKV equivalents."""
    vision_model = getattr(model, "vision_model", None)
    vision_transformer = getattr(vision_model, "vision_model", None)
    encoder = getattr(vision_transformer, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return 0

    replaced = 0
    for idx, layer in enumerate(layers):
        if not all(hasattr(layer.self_attn, name) for name in ("q_proj", "k_proj", "v_proj", "out_proj")):
            continue
        layers[idx] = FusedQKVSiglipEncoderLayer(
            layer,
            use_te=use_te,
            use_te_mlp=use_te_mlp,
            use_te_qkv=use_te_qkv,
            use_te_layer=use_te_layer,
        )
        replaced += 1

    logger.info("Replaced SigLIP encoder layers with fused-QKV attention: layers=%d", replaced)
    return replaced


def replace_vision_projection_with_te(model: nn.Module) -> bool:
    """Replace the vision-to-language projection with a trainable TE prefix."""
    projection = getattr(model, "mlp1", None)
    if not isinstance(projection, nn.Sequential) or len(projection) != 4:
        return False
    model.mlp1 = FusedTEVisionProjection(projection)
    logger.info("Fused vision projection LayerNorm + first Linear")
    return True


class _FusedLinearProjection(nn.Module):
    """One projection whose output is exposed through several Linear views.

    Llama attention and SwiGLU call their projections sequentially with the
    same hidden-state tensor. Computing the concatenated projection once keeps
    the original module interfaces while removing redundant small GEMM launches.
    """

    def __init__(self, sources: tuple[nn.Linear, ...]):
        super().__init__()
        if not sources or not all(isinstance(source, nn.Linear) for source in sources):
            raise TypeError("fused projections require nn.Linear sources")
        first = sources[0]
        if any(source.in_features != first.in_features for source in sources):
            raise ValueError("fused projections require matching input dimensions")

        self.sizes = tuple(source.out_features for source in sources)
        self.offsets = (0, *torch.tensor(self.sizes, dtype=torch.int64).cumsum(0).tolist())
        self.linear = nn.Linear(
            first.in_features,
            sum(self.sizes),
            bias=first.bias is not None,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        with torch.no_grad():
            self.linear.weight.copy_(torch.cat([source.weight for source in sources], dim=0))
            if self.linear.bias is not None:
                self.linear.bias.copy_(torch.cat([source.bias for source in sources], dim=0))

        self._cached_input = None
        self._cached_output = None
        self._remaining_views = 0

    def _view(self, hidden_states: torch.Tensor, index: int) -> torch.Tensor:
        if self._cached_input is not hidden_states:
            self._cached_input = hidden_states
            self._cached_output = self.linear(hidden_states)
            self._remaining_views = len(self.sizes)

        start, end = self.offsets[index], self.offsets[index + 1]
        output = self._cached_output[..., start:end]
        self._remaining_views -= 1
        if self._remaining_views == 0:
            self._cached_input = None
            self._cached_output = None
        return output


class _FusedLinearView(nn.Module):
    """A parameter-free view retaining the source module's call contract."""

    def __init__(self, projection: _FusedLinearProjection, index: int):
        super().__init__()
        object.__setattr__(self, "_projection", projection)
        self._index = index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._projection._view(hidden_states, self._index)


def replace_llama_language_projections(model: nn.Module) -> tuple[int, int]:
    """Fuse trainable Llama QKV and SwiGLU gate/up projections.

    The replacement is shape-preserving and copies the original weights. The
    fused parameters remain trainable; only the number of projection launches
    changes. This is applied before distributed wrapping and optimizer creation.
    """

    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        logger.warning("Llama projection fusion requested but no language-model layers were found")
        return 0, 0

    fused_qkv = 0
    fused_gate_up = 0
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None and all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")):
            sources = (attention.q_proj, attention.k_proj, attention.v_proj)
            if all(isinstance(source, nn.Linear) for source in sources):
                projection = _FusedLinearProjection(sources)
                attention.qkv_proj = projection
                attention.q_proj = _FusedLinearView(projection, 0)
                attention.k_proj = _FusedLinearView(projection, 1)
                attention.v_proj = _FusedLinearView(projection, 2)
                fused_qkv += 1

        mlp = getattr(layer, "mlp", None)
        if mlp is not None and all(hasattr(mlp, name) for name in ("gate_proj", "up_proj")):
            sources = (mlp.gate_proj, mlp.up_proj)
            if all(isinstance(source, nn.Linear) for source in sources):
                projection = _FusedLinearProjection(sources)
                mlp.gate_up_proj = projection
                mlp.gate_proj = _FusedLinearView(projection, 0)
                mlp.up_proj = _FusedLinearView(projection, 1)
                fused_gate_up += 1

    logger.info(
        "Fused Llama language projections: qkv_layers=%d gate_up_layers=%d",
        fused_qkv,
        fused_gate_up,
    )
    return fused_qkv, fused_gate_up


def disable_unused_siglip_pooling_head_grad(model: nn.Module) -> int:
    """Disable the unused SigLIP pooling head for retrieval feature extraction."""
    vision_model = getattr(model, "vision_model", None)
    vision_transformer = getattr(vision_model, "vision_model", None)
    pooling_head = getattr(vision_transformer, "head", None)
    if pooling_head is None:
        return 0

    # Retrieval consumes only ``last_hidden_state``. Avoid launching the
    # pooling head while retaining its parameters for checkpoint compatibility.
    if hasattr(vision_transformer, "use_head"):
        vision_transformer.use_head = False

    disabled = 0
    for param in pooling_head.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            disabled += param.numel()

    logger.info("Disabled unused SigLIP pooling head gradients: parameters=%d", disabled)
    return disabled


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
        image_token_indices: Optional[torch.LongTensor] = None,
        bidirectional_mask: Optional[torch.Tensor] = None,
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
    "LlamaNemotronVLModel",
    "LlamaNemotronVLConfig",
    "ModelClass",
]
