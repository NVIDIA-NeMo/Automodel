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

"""Inkling 2 configuration classes for Automodel's registry-first config lookup.

Attribute names, defaults and aliases follow the Transformers ``inkling2`` configuration
(huggingface/transformers commit 109a7d19bf667168837f32929e00b791e195d14d,
``src/transformers/models/inkling2/configuration_inkling2.py``, Apache-2.0), so a config
written by either library loads in the other. The pinned Transformers release has no
``inkling2`` model type, so these classes are registered through
``_CUSTOM_CONFIG_REGISTRATIONS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    import torch
    from torch import nn

    from nemo_automodel.components.distributed.config import DistributedSetup
    from nemo_automodel.components.models.inkling2.dspark import Inkling2DSparkModel, Inkling2DSparkOptions

_SLIDING_LAYER_TYPE = "hybrid_sliding"
_FULL_LAYER_TYPE = "hybrid"
_LAYER_TYPES = (_SLIDING_LAYER_TYPE, _FULL_LAYER_TYPE)


def _default_layer_types(num_hidden_layers: int, local_layer_ids: list[int] | None) -> list[str]:
    """Return per-layer attention types.

    Released checkpoints list their sliding-window layers in ``local_layer_ids``; without
    it every sixth layer is a full-attention layer, as in Transformers.
    """
    if local_layer_ids is None:
        return [_FULL_LAYER_TYPE if (i + 1) % 6 == 0 else _SLIDING_LAYER_TYPE for i in range(num_hidden_layers)]
    local = set(local_layer_ids)
    return [_SLIDING_LAYER_TYPE if i in local else _FULL_LAYER_TYPE for i in range(num_hidden_layers)]


class Inkling2TextConfig(PretrainedConfig):
    """Configuration for the Inkling 2 text backbone.

    ``layer_types`` holds ``"hybrid_sliding"`` for sliding-window layers and ``"hybrid"`` for
    full-attention layers, the strings Transformers uses.
    """

    model_type = "inkling2_text"
    base_config_key = "text_config"
    attribute_map = {
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
        "sconv_kernel_size": "conv_kernel_size",
        "model_max_length": "max_position_embeddings",
        "intermediate_size": "moe_intermediate_size",
        "routed_scaling_factor": "route_scale",
    }

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int = 200058,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int = 48,
        swa_num_key_value_heads: int = 8,
        swa_head_dim: int = 128,
        sliding_window_size: int = 1024,
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int = 128000,
        log_scaling_alpha: float = 0.1,
        layer_types: list[str] | None = None,
        local_layer_ids: list[int] | None = None,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        hidden_act: str = "silu",
        moe_intermediate_size: int = 2304,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        route_scale: float = 8.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 200006,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        n_layers_per_attn_res_block: int = 6,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        final_logit_softcapping: float = 30.0,
        scoring_func: str = "sigmoid",
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        if layer_types is None:
            layer_types = _default_layer_types(num_hidden_layers, local_layer_ids)
        if len(layer_types) != num_hidden_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries for {num_hidden_layers} layers")
        unknown_types = sorted(set(layer_types) - set(_LAYER_TYPES))
        if unknown_types:
            raise ValueError(f"Unsupported Inkling 2 layer_types {unknown_types}; expected entries of {_LAYER_TYPES}")
        if scoring_func != "sigmoid":
            raise ValueError(f"Inkling 2 routes with sigmoid scores, got scoring_func={scoring_func!r}")

        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.local_layer_ids = local_layer_ids
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.hidden_act = hidden_act
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = route_scale
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.final_logit_softcapping = final_logit_softcapping
        self.scoring_func = scoring_func
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # Set after the base initializer, which may reset ``layer_types``.
        self.layer_types = layer_types


class Inkling2VisionConfig(PretrainedConfig):
    """Configuration for the Inkling 2 ViT vision tower and its projector."""

    model_type = "inkling2_vision"
    base_config_key = "vision_config"
    attribute_map = {"native_patch_size": "patch_size", "adapter_merge_size": "spatial_merge_size"}
    default_rope_type = "axial"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 23,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        patch_size: int = 20,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 0.1,
        rope_parameters: dict[str, Any] | None = None,
        spatial_merge_size: int = 2,
        posemb_grid_size: int = 32,
        projection_hidden_dim: int = 4096,
        projector_bias: bool = False,
        text_hidden_size: int = 4608,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * mlp_ratio) if intermediate_size is None else intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.mlp_ratio = mlp_ratio
        self.layer_scale_init_value = layer_scale_init_value
        self.rope_parameters = rope_parameters
        self.spatial_merge_size = spatial_merge_size
        self.posemb_grid_size = posemb_grid_size
        self.projection_hidden_dim = projection_hidden_dim
        self.projector_bias = projector_bias
        self.text_hidden_size = text_hidden_size
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Inkling2Config(PretrainedConfig):
    """Top-level Inkling 2 configuration.

    ``vision_config=None`` builds a text-only model. ``mtp_config`` keeps the drafter
    section of the released ``config.json`` as a plain mapping for the DSpark drafter.
    """

    model_type = "inkling2"
    sub_configs = {"text_config": Inkling2TextConfig, "vision_config": Inkling2VisionConfig}

    def __init__(
        self,
        text_config: Inkling2TextConfig | dict[str, Any] | None = None,
        vision_config: Inkling2VisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 200054,
        mtp_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Inkling2TextConfig(**text_config)
        elif text_config is None:
            text_config = Inkling2TextConfig()
        if isinstance(vision_config, dict):
            vision_config = Inkling2VisionConfig(**vision_config)
        if vision_config is not None and vision_config.text_hidden_size != text_config.hidden_size:
            raise ValueError(
                f"vision_config.text_hidden_size ({vision_config.text_hidden_size}) must match "
                f"text_config.hidden_size ({text_config.hidden_size})"
            )
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.mtp_config = mtp_config
        super().__init__(**kwargs)

    def build_dspark_draft(self, options: Inkling2DSparkOptions) -> Inkling2DSparkModel:
        """Build the DSpark draft in the released ``mtp.safetensors`` layout.

        Args:
            options: Declarative draft settings supplied by the DSpark recipe; they must agree
                with ``mtp_config``.

        Returns:
            Newly initialized draft ready for device placement.

        Raises:
            ValueError: If the checkpoint has no ``mtp_config`` or an option disagrees with it.
        """
        from nemo_automodel.components.models.inkling2.dspark import Inkling2DSparkConfig

        if self.mtp_config is None:
            raise ValueError("This Inkling 2 config has no mtp_config, so it defines no DSpark draft.")
        return Inkling2DSparkConfig.from_options(self.text_config, self.mtp_config, options).build()

    def build_dspark_target(
        self,
        *,
        target_path: str,
        trust_remote_code: bool,
        device: torch.device,
        compute_dtype: torch.dtype,
        distributed_setup: DistributedSetup,
        load_metadata_with_gloo: bool,
    ) -> nn.Module:
        """Load the frozen DSpark target on the recipe's expert-parallel / FSDP mesh.

        Args:
            target_path: Hub id or local directory of this checkpoint.
            trust_remote_code: Forwarded to the checkpoint loaders.
            device: Resolved execution device.
            compute_dtype: Precision used to load and run the frozen target.
            distributed_setup: Runtime parallelism configuration composed by the recipe.
            load_metadata_with_gloo: Read checkpoint metadata over Gloo.

        Returns:
            The pretrained target, frozen and in eval mode.
        """
        from nemo_automodel.components.models.inkling2.dspark_target import Inkling2DSparkTargetConfig

        return Inkling2DSparkTargetConfig(target_path=target_path, trust_remote_code=trust_remote_code).build(
            device=device,
            compute_dtype=compute_dtype,
            distributed_setup=distributed_setup,
            load_metadata_with_gloo=load_metadata_with_gloo,
        )


__all__ = ["Inkling2Config", "Inkling2TextConfig", "Inkling2VisionConfig"]
