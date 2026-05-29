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

"""Typed configuration classes for the MiniMax M3 VL family.

The released checkpoint ships ``configuration_minimax_m3_vl.py`` which coerces
the ``vision_config``/``text_config`` sub-dicts into generic ``PretrainedConfig``
instances (the text backbone's ``model_type="minimax_m2"`` is not in HF's
``CONFIG_MAPPING``).  For the native AutoModel implementation we declare typed
sub-configs so that fields such as ``sparse_attention_config``, ``moe_layer_freq``
and the SwiGLU-OAI parameters are real, defaulted attributes.

Mirrors the canonical sglang reference ``sglang.srt.configs.minimax_vl`` and the
field set in the checkpoint's ``config.json``; keep them in sync.
"""

from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig


def _json_safe_value(value: Any) -> Any:
    """Convert config values that are valid in-memory but not JSON serializable."""
    if value.__class__.__module__ == "torch" and value.__class__.__name__ == "dtype":
        return str(value).removeprefix("torch.")
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


class MiniMaxM3VLTextConfig(PretrainedConfig):
    """Configuration for the MiniMax M3 (mixed sparse/dense MoE) text backbone."""

    model_type = "minimax_m3"
    architectures = ["MiniMaxM3SparseForCausalLM"]

    def __init__(
        self,
        hidden_size: int = 6144,
        intermediate_size: int = 3072,
        dense_intermediate_size: int = 12288,
        shared_intermediate_size: int = 3072,
        num_hidden_layers: int = 60,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        vocab_size: int = 200064,
        max_position_embeddings: int = 524288,
        rms_norm_eps: float = 1e-6,
        use_gemma_norm: bool = True,
        attention_output_gate: bool = False,
        rope_theta: float = 5000000.0,
        rotary_dim: int = 64,
        partial_rotary_factor: float = 0.5,
        hidden_act: str = "swigluoai",
        use_qk_norm: bool = True,
        qk_norm_type: str = "per_head",
        tie_word_embeddings: bool = False,
        # MoE
        num_local_experts: int = 128,
        num_experts_per_tok: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "sigmoid",
        use_routing_bias: bool = True,
        routed_scaling_factor: float = 2.0,
        moe_layer_freq: Optional[list[int]] = None,
        # SwiGLU-OAI (GPT-OSS style) activation parameters
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        # Sparse attention (DeepSeek-style index branch, block-level top-k)
        sparse_attention_config: Optional[dict] = None,
        # Multi-token prediction
        num_mtp_modules: int = 1,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_gemma_norm = use_gemma_norm
        self.attention_output_gate = attention_output_gate
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.hidden_act = hidden_act
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.routed_scaling_factor = routed_scaling_factor
        # moe_layer_freq[i] == 0 -> dense MLP at layer i, != 0 -> sparse MoE.
        self.moe_layer_freq = moe_layer_freq
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.sparse_attention_config = sparse_attention_config
        self.num_mtp_modules = num_mtp_modules
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

    def to_dict(self):
        return _json_safe_value(super().to_dict())


class MiniMaxM3VLVisionConfig(PretrainedConfig):
    """Configuration for the MiniMax M3 VL CLIP-style vision tower (Conv3d + 3D RoPE)."""

    model_type = "minimax_m3_vision"

    def __init__(
        self,
        hidden_size: int = 1280,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 32,
        intermediate_size: int = 5120,
        patch_size: int = 14,
        image_size: int = 672,
        projection_dim: int = 6144,
        num_channels: int = 3,
        position_embedding_type: str = "rope",
        rope_mode: str = "3d",
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        img_token_compression_config: Optional[dict] = None,
        vision_segment_max_frames: int = 4,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.projection_dim = projection_dim
        self.num_channels = num_channels
        self.position_embedding_type = position_embedding_type
        self.rope_mode = rope_mode
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.img_token_compression_config = img_token_compression_config or {
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
        self.vision_segment_max_frames = vision_segment_max_frames
        super().__init__(**kwargs)

    def to_dict(self):
        return _json_safe_value(super().to_dict())


class MiniMaxM3VLConfig(PretrainedConfig):
    """Top-level configuration for MiniMax M3 vision-language checkpoints."""

    model_type = "minimax_m3_vl"
    sub_configs = {"text_config": MiniMaxM3VLTextConfig, "vision_config": MiniMaxM3VLVisionConfig}

    def __init__(
        self,
        vision_config: Optional[Union[dict, MiniMaxM3VLVisionConfig]] = None,
        text_config: Optional[Union[dict, MiniMaxM3VLTextConfig]] = None,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        image_seq_length: int = 576,
        process_image_mode: str = "dynamic_res",
        projector_hidden_act: str = "gelu",
        projector_hidden_size: int = 6144,
        multimodal_projector_bias: bool = True,
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "full",
        img_token_compression_config: Optional[dict] = None,
        image_grid_pinpoints: Optional[str] = None,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = MiniMaxM3VLVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = MiniMaxM3VLVisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = MiniMaxM3VLTextConfig()
        elif isinstance(text_config, dict):
            text_config = MiniMaxM3VLTextConfig(**text_config)
        self.text_config = text_config

        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.image_seq_length = image_seq_length
        self.process_image_mode = process_image_mode
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.multimodal_projector_bias = multimodal_projector_bias
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.img_token_compression_config = img_token_compression_config or {}
        self.image_grid_pinpoints = image_grid_pinpoints
        # Convenience mirror used by recipe/PP helpers.
        self.hidden_size = text_config.hidden_size
        self.max_position_embeddings = text_config.max_position_embeddings
        super().__init__(**kwargs)

    def to_dict(self):
        return _json_safe_value(super().to_dict())
