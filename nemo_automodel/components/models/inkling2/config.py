# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Configuration for the inspected Inkling2 checkpoint and text reference."""

import json
import os
import re
from typing import Any, Literal, TypedDict

from transformers import PretrainedConfig


class Inkling2QuantizationConfig(TypedDict):
    """Expert storage of a released checkpoint: NVFP4 for the first layers, MXFP8 after them."""

    quant_method: Literal["inkling2"]
    nvfp4_layer_count: int


class _ParameterGroup(TypedDict):
    """One ``hf_quant_config.json`` group; only the keys the loader reads."""

    compute_dtype: str
    tensors: list[str]


_EXPERT_WEIGHT = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)")
_EXPERT_PROJECTIONS = ("w13_weight", "w2_weight")


def _checked_quantization_config(
    declaration: Inkling2QuantizationConfig, num_hidden_layers: int
) -> Inkling2QuantizationConfig:
    if declaration.get("quant_method") != "inkling2":
        raise ValueError("Inkling2 checkpoint import requires quant_method='inkling2'")
    if set(declaration) != {"quant_method", "nvfp4_layer_count"}:
        raise ValueError("Unsupported Inkling2 quantization metadata fields")
    boundary = declaration["nvfp4_layer_count"]
    if not isinstance(boundary, int) or not 0 <= boundary <= num_hidden_layers:
        raise ValueError("nvfp4_layer_count must be an integer within the layer count")
    return declaration


def load_quantization_config(name_or_path: str, num_hidden_layers: int) -> Inkling2QuantizationConfig | None:
    """Read the released ``hf_quant_config.json`` stored next to ``config.json``.

    Args:
        name_or_path: Local checkpoint directory or Hub repository id.
        num_hidden_layers: Decoder depth the expert declaration must cover.

    Returns:
        The expert storage layout, or None for a checkpoint without the sidecar,
        such as the floating checkpoints that training saves.
    """
    local = os.path.join(name_or_path, "hf_quant_config.json")
    if not os.path.isfile(local):
        if os.path.isdir(name_or_path):
            return None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            local = hf_hub_download(name_or_path, "hf_quant_config.json")
        except EntryNotFoundError:
            return None
    with open(local, encoding="utf-8") as stream:
        sidecar = json.load(stream)
    if sidecar["quant_method"] != "inkling2":
        raise ValueError("Inkling2 checkpoint import requires quant_method='inkling2'")
    groups: list[_ParameterGroup] = sidecar["model"]["parameter_groups"]
    layout: dict[str, set[tuple[int, str]]] = {"nvfp4": set(), "mxfp8": set()}
    for group in groups:
        for name in group["tensors"]:
            match = _EXPERT_WEIGHT.fullmatch(name)
            if match is None:
                continue
            if group["compute_dtype"] not in layout:
                raise ValueError(f"Unsupported expert storage declaration: {group['compute_dtype']}")
            layout[group["compute_dtype"]].add((int(match[1]), match[2]))
    boundary = len({layer for layer, _ in layout["nvfp4"]})
    if layout["nvfp4"] != {(i, p) for i in range(boundary) for p in _EXPERT_PROJECTIONS} or layout["mxfp8"] != {
        (i, p) for i in range(boundary, num_hidden_layers) for p in _EXPERT_PROJECTIONS
    }:
        raise ValueError("Expected a contiguous NVFP4 expert prefix followed by MXFP8 experts, with both projections")
    return {"quant_method": "inkling2", "nvfp4_layer_count": boundary}


def resolve_quantization_config(config: "Inkling2TextConfig | Inkling2Config") -> None:
    """Fill ``quantization_config`` from the checkpoint sidecar for a config loaded from a checkpoint."""
    if config.quantization_config is not None or not config.name_or_path:
        return
    text = config.text_config if isinstance(config, Inkling2Config) else config
    declaration = load_quantization_config(config.name_or_path, text.num_hidden_layers)
    if declaration is not None:
        config.quantization_config = declaration
        text.quantization_config = declaration


class Inkling2TextConfig(PretrainedConfig):
    """Text configuration; defaults follow the inspected checkpoint, not serving defaults."""

    model_type = "inkling2_model"
    base_config_key = "text_config"
    attribute_map = {
        "model_max_length": "max_position_embeddings",
        "n_layers_per_attnres_block": "n_layers_per_attn_res_block",
    }

    def __init__(
        self,
        *,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        sliding_window_size: int = 1024,
        local_layer_ids: list[int] | None = None,
        max_position_embeddings: int = 1048576,
        model_max_length: int | None = None,
        log_scaling_n_floor: int = 8192,
        log_scaling_alpha: float = 0.1,
        rms_norm_eps: float = 1e-6,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        sconv_kernel_size: int = 4,
        n_layers_per_attn_res_block: int | None = None,
        n_layers_per_attnres_block: int | None = None,
        attn_res_normalize_values: bool = True,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        q_bias: bool = False,
        o_bias: bool = False,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 0,
        intermediate_size: int = 2304,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        final_logit_softcapping: float = 30.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        quantization_config: Inkling2QuantizationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Hugging Face owns the open-ended serialized metadata at this boundary.
        kwargs["tie_word_embeddings"] = tie_word_embeddings
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = (
            200058
            if unpadded_vocab_size is None and vocab_size == 201024
            else vocab_size
            if unpadded_vocab_size is None
            else unpadded_vocab_size
        )
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = (
            num_attention_heads if swa_num_attention_heads is None else swa_num_attention_heads
        )
        self.swa_num_key_value_heads = (
            num_key_value_heads if swa_num_key_value_heads is None else swa_num_key_value_heads
        )
        self.swa_head_dim = head_dim if swa_head_dim is None else swa_head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.sliding_window_size = sliding_window_size
        self.local_layer_ids = (
            list(local_layer_ids)
            if local_layer_ids is not None
            else [i for i in range(num_hidden_layers) if i % 6 != 5]
        )
        self.max_position_embeddings = max_position_embeddings if model_max_length is None else model_max_length
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.rms_norm_eps = rms_norm_eps
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.sconv_kernel_size = sconv_kernel_size
        if (
            n_layers_per_attn_res_block is not None
            and n_layers_per_attnres_block is not None
            and n_layers_per_attn_res_block != n_layers_per_attnres_block
        ):
            raise ValueError("Conflicting attention residual block sizes in current and legacy config fields")
        self.n_layers_per_attn_res_block = (
            n_layers_per_attn_res_block
            if n_layers_per_attn_res_block is not None
            else n_layers_per_attnres_block
            if n_layers_per_attnres_block is not None
            else 6
        )
        self.attn_res_normalize_values = attn_res_normalize_values
        self.pre_norm_pre_affine = pre_norm_pre_affine
        self.final_norm_pre_affine = final_norm_pre_affine
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.final_logit_softcapping = final_logit_softcapping
        self.initializer_range = initializer_range
        # Released checkpoints declare this in hf_quant_config.json; the model
        # fills it in from the checkpoint directory when it is built.
        self.quantization_config = (
            None
            if quantization_config is None
            else _checked_quantization_config(quantization_config, num_hidden_layers)
        )
        self.use_cache = False
        if (
            min(
                hidden_size,
                num_hidden_layers,
                head_dim,
                d_rel,
                rel_extent,
                sliding_window_size,
                sconv_kernel_size,
                intermediate_size,
                self.max_position_embeddings,
                log_scaling_n_floor,
            )
            <= 0
        ):
            raise ValueError("Inkling2 dimensions, window sizes and scaling floor must be positive")
        for query_heads, kv_heads, width in (
            (num_attention_heads, num_key_value_heads, head_dim),
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            ),
        ):
            if min(query_heads, kv_heads, width) <= 0 or query_heads % kv_heads:
                raise ValueError("Inkling2 query heads must be a positive multiple of KV heads")
        if not 0 < self.unpadded_vocab_size <= vocab_size:
            raise ValueError("unpadded_vocab_size must be within the embedding vocabulary")
        if not 1 < num_experts_per_tok <= n_routed_experts:
            raise ValueError("The Inkling2 reference requires 1 < num_experts_per_tok <= n_routed_experts")
        if not 0 < self.n_layers_per_attn_res_block <= num_hidden_layers:
            raise ValueError("Attention residual block size must be within the layer count")
        if len(set(self.local_layer_ids)) != len(self.local_layer_ids) or any(
            i < 0 or i >= num_hidden_layers for i in self.local_layer_ids
        ):
            raise ValueError("local_layer_ids must contain unique valid layer indices")
        if n_shared_experts != 0 or pre_norm_pre_affine or final_norm_pre_affine or tie_word_embeddings:
            raise ValueError(
                "The inspected Inkling2 reference uses no shared experts, no pre-affine scales and untied weights"
            )
        if (
            min(
                rms_norm_eps,
                qk_l2_clip_threshold,
                rel_l2_clip_threshold,
                mlp_gate_softcap,
                mlp_up_softcap,
                route_scale,
                final_logit_softcapping,
                initializer_range,
            )
            <= 0
        ):
            raise ValueError("Inkling2 epsilon, thresholds, softcaps and initialization scale must be positive")


class Inkling2VisionConfig(PretrainedConfig):
    """Released image encoder dimensions, including the spatial merge projector."""

    model_type = "inkling2_vision_model"

    def __init__(
        self,
        *,
        temporal_patch_size: int = 2,
        native_patch_size: int = 20,
        adapter_merge_size: int = 2,
        projection_hidden_dim: int = 4096,
        projector_bias: bool = False,
        hidden_size: int = 1024,
        num_hidden_layers: int = 23,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
        posemb_grid_size: int = 32,
        layer_norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if (
            min(
                temporal_patch_size,
                native_patch_size,
                adapter_merge_size,
                projection_hidden_dim,
                hidden_size,
                num_hidden_layers,
                num_attention_heads,
                posemb_grid_size,
            )
            <= 0
        ):
            raise ValueError("Inkling2 vision dimensions must be positive")
        if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 4:
            raise ValueError("Inkling2 vision head width must be divisible by four")
        if mlp_ratio <= 0 or hidden_size * mlp_ratio != int(hidden_size * mlp_ratio) or layer_norm_eps <= 0:
            raise ValueError("Inkling2 vision requires a positive integral MLP width and positive epsilon")
        self.temporal_patch_size = temporal_patch_size
        self.native_patch_size = native_patch_size
        self.adapter_merge_size = adapter_merge_size
        self.projection_hidden_dim = projection_hidden_dim
        self.projector_bias = projector_bias
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.posemb_grid_size = posemb_grid_size
        self.layer_norm_eps = layer_norm_eps


class Inkling2Config(PretrainedConfig):
    """Released image/text model configuration and optional source quantization."""

    model_type = "inkling2"
    sub_configs = {"text_config": Inkling2TextConfig, "vision_config": Inkling2VisionConfig}

    def __init__(
        self,
        *,
        text_config: Inkling2TextConfig | dict[str, Any] | None = None,
        vision_config: Inkling2VisionConfig | dict[str, Any] | None = None,
        mtp_config: dict[str, Any] | None = None,
        quantization_config: Inkling2QuantizationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Serialized dictionaries are accepted only at Hugging Face's config boundary.
        self.text_config = (
            text_config if isinstance(text_config, Inkling2TextConfig) else Inkling2TextConfig(**(text_config or {}))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, Inkling2VisionConfig)
            else Inkling2VisionConfig(**(vision_config or {}))
        )
        self.mtp_config = mtp_config
        self.quantization_config = (
            self.text_config.quantization_config
            if quantization_config is None
            else _checked_quantization_config(quantization_config, self.text_config.num_hidden_layers)
        )
        self.text_config.quantization_config = self.quantization_config
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class Inkling2LegacyConfig(Inkling2Config):
    """Compatibility with the original download's immutable configuration."""

    model_type = "inkling2_mm_model"

    def __init__(self, **kwargs: Any) -> None:
        # Transformers dataclass-wraps subclasses without an explicit initializer.
        # Delegate so nested text/vision dictionaries retain their typed conversion.
        super().__init__(**kwargs)
