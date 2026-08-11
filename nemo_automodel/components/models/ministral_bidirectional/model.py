# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.
"""
Bidirectional Ministral3 model for embedding tasks.

This module provides a modified Ministral3Model that uses bidirectional (non-causal)
attention, suitable for generating embeddings where each token should attend
to all other tokens in the sequence.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model, Mistral3MultiModalProjector
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Ministral3BidirectionalConfig(Ministral3Config):
    """Configuration for Ministral3BidirectionalModel with pooling and temperature settings."""

    model_type = "ministral3_bidirec"

    def __init__(self, pooling: str = "avg", temperature: float = 1.0, **kwargs) -> None:
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(**kwargs)


class Ministral3BidirectionalModel(Ministral3Model):
    """
    Ministral3Model modified to use bidirectional (non-causal) attention.

    In standard Ministral3, each token can only attend to previous tokens (causal
    attention). This model removes that restriction, allowing each token to attend
    to all tokens in the sequence, which is useful for embedding tasks.

    The key modifications are:
        1. Setting is_causal=False on all attention layers
        2. Using a bidirectional attention mask instead of causal mask

    Loading a Mistral3 VLM checkpoint (e.g. ``mistralai/Ministral-3-3B-Base-2512``
    or ``mistralai/Ministral-3-3B-Instruct-2512``) requires extracting the language
    tower; this is driven by the recipe YAML via
    ``extract_submodel: language_model`` and handled by
    :func:`nemo_automodel._transformers.retrieval.build_encoder_backbone`.

    Text-only checkpoints (e.g. ``mistralai/Ministral-3B-Instruct``) load directly
    via the standard ``from_pretrained`` path with no extraction needed.
    """

    config_class = Ministral3BidirectionalConfig

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Forward pass with bidirectional attention.

        Identical to Ministral3Model.forward() except the causal mask is replaced
        with a bidirectional mask, allowing all tokens to attend to each other.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=bidirectional_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Mistral3BidirectionalConfig(Mistral3Config):
    """Configuration for Mistral3BidirectionalModel with pooling and temperature settings."""

    model_type = "mistral3_bidirec"

    @property
    def pooling(self):
        return self._pooling

    @pooling.setter
    def pooling(self, value):
        self._pooling = value
        if self.text_config is not None:
            if isinstance(self.text_config, dict):
                self.text_config["pooling"] = value
            else:
                self.text_config.pooling = value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling = kwargs.get("pooling", "avg")

    def __post_init__(self, **kwargs):
        if self.text_config is not None:
            if isinstance(self.text_config, dict):
                self.text_config = Ministral3BidirectionalConfig(**self.text_config)
        super().__post_init__(**kwargs)

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.pop("_pooling", None)
        return config_dict


class Mistral3BidirectionalModel(Mistral3Model):
    """
    Mistral3Model modified to use bidirectional (non-causal) attention (Ministral3BidirectionalModel).

    In standard Mistral3, each token can only attend to previous tokens (causal
    attention). This model removes that restriction, allowing each token to attend
    to all tokens in the sequence, which is useful for embedding tasks.
    """

    config_class = Mistral3BidirectionalConfig

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Any:
        """Load stock Mistral3 weights into the bidirectional VL layout.

        Caller-provided key mappings are preserved unless they conflict with
        the structural mapping required by this model's fixed module layout.
        """
        key_mapping = dict(kwargs.pop("key_mapping", None) or {})
        # Stock Mistral3 wraps the text tower in an extra ``model`` module. This
        # custom base model owns the same tower directly, so mirror Transformers'
        # native Mistral3Model checkpoint conversion during loading only. This
        # required mapping intentionally overrides a conflicting caller entry.
        key_mapping[r"^language_model\.model\."] = "language_model."
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            key_mapping=key_mapping,
            **kwargs,
        )

    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.language_model = Ministral3BidirectionalModel(config.text_config)
        self.post_init()

    def _nemo_apply_liger_kernel(self, liger_kernel_transformers) -> None:
        """Apply compatible Liger kernels to the Ministral text tower only."""
        liger_kernel_transformers.apply_liger_kernel_to_mistral(
            model=self.language_model,
            rope=False,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

    def _dummy_vision_sum(self) -> torch.Tensor:
        """Run the smallest valid Pixtral image through the sharded vision tower."""
        vision_tower = self.vision_tower
        parameter = next(vision_tower.parameters())
        patch_size = getattr(vision_tower, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(vision_tower.config, "patch_size", 16)
        spatial_merge_size = getattr(self.config, "spatial_merge_size", 1)

        if isinstance(patch_size, (list, tuple)):
            patch_height, patch_width = patch_size
        else:
            patch_height = patch_width = patch_size
        if isinstance(spatial_merge_size, (list, tuple)):
            merge_height, merge_width = spatial_merge_size
        else:
            merge_height = merge_width = spatial_merge_size

        # Mistral3 squeezes projected image features, so retain more than one
        # merged token before its split-by-image step.
        image_height = int(patch_height) * int(merge_height) * 2
        image_width = int(patch_width) * int(merge_width) * 2
        dummy_pixels = torch.zeros(
            1,
            3,
            image_height,
            image_width,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        dummy_image_sizes = torch.tensor(
            [[image_height, image_width]],
            device=parameter.device,
            dtype=torch.long,
        )
        dummy_outputs = self.get_image_features(
            pixel_values=dummy_pixels,
            image_sizes=dummy_image_sizes,
            return_dict=True,
        )
        image_features = dummy_outputs.pooler_output
        if isinstance(image_features, (list, tuple)):
            return sum(feature.sum() for feature in image_features)
        return image_features.sum()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        run_dummy_vision: bool | None = None,
        **kwargs,
    ):
        """Keep dummy and real vision calls inside the same FSDP root forward."""
        if self.training and run_dummy_vision is not False and pixel_values is None:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                input_ids = None
            dummy_vision_sum = self._dummy_vision_sum()
            inputs_embeds = inputs_embeds + dummy_vision_sum.to(inputs_embeds.dtype) * 0.0

        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def build_collator(self, processor=None, **kwargs):
        """Dummy method that matches the interface for evaluation, returning the processor argunment as the model collator for data loading."""
        return processor


# Export for ModelRegistry auto-discovery
ModelClass = [Ministral3BidirectionalModel, Mistral3BidirectionalModel]


def _register_with_hf_auto_classes() -> None:
    """Register bidirectional Ministral3 with HuggingFace Auto classes.

    Needed so ``AutoModel.from_config(Ministral3BidirectionalConfig)`` and checkpoint
    reload paths that use Auto resolution work consistently.
    """
    try:
        AutoConfig.register(Ministral3BidirectionalConfig.model_type, Ministral3BidirectionalConfig)
    except ValueError:
        pass  # Already registered
    try:
        AutoModel.register(Ministral3BidirectionalConfig, Ministral3BidirectionalModel)
    except ValueError:
        pass  # Already registered

    try:
        AutoConfig.register(Mistral3BidirectionalConfig.model_type, Mistral3BidirectionalConfig)
    except ValueError:
        pass  # Already registered
    try:
        AutoModel.register(Mistral3BidirectionalConfig, Mistral3BidirectionalModel)
    except ValueError:
        pass  # Already registered


_register_with_hf_auto_classes()

__all__ = [
    "Ministral3BidirectionalModel",
    "Ministral3BidirectionalConfig",
    "Mistral3BidirectionalModel",
    "ModelClass",
]
