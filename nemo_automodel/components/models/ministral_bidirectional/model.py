# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.
"""
Custom bidirectional Ministral3 model for explicit and legacy use.

This module provides a modified Ministral3Model that uses bidirectional (non-causal)
attention, suitable for generating embeddings where each token should attend
to all other tokens in the sequence. Standard ``ministral3`` embedding checkpoints
use the stock HuggingFace model with ``is_causal=False``; this custom architecture
remains available for ``ministral3_bidirec`` checkpoints.
"""

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3Model,
    Mistral3MultiModalProjector,
    Mistral3PreTrainedModel,
)
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

    This class is not selected automatically for standard ``ministral3`` embedding
    checkpoints. It remains registered for explicit and legacy use.

    In causal Ministral3, each token can only attend to previous tokens (causal
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
            past_key_values=past_key_values,
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
    def pooling(self) -> str:
        return self._pooling

    @pooling.setter
    def pooling(self, value: str) -> None:
        self._pooling = value
        if self.text_config is not None:
            if isinstance(self.text_config, dict):
                self.text_config["pooling"] = value
            else:
                self.text_config.pooling = value

    def __init__(self, pooling: str = "avg", temperature: float = 1.0, **kwargs) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        super().__init__(**kwargs)
        self.pooling = pooling
        self.temperature = temperature

    def __post_init__(self, **kwargs: Any) -> None:
        if self.text_config is not None:
            if isinstance(self.text_config, dict):
                self.text_config = Ministral3BidirectionalConfig(**self.text_config)
        super().__post_init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize pooling under its public configuration key."""
        config_dict = super().to_dict()
        config_dict.pop("_pooling", None)
        config_dict["pooling"] = self.pooling
        return config_dict


class Mistral3BidirectionalModel(Mistral3Model):
    """
    Mistral3Model modified to use bidirectional (non-causal) attention (Ministral3BidirectionalModel).

    In standard Mistral3, each token can only attend to previous tokens (causal
    attention). This model removes that restriction, allowing each token to attend
    to all tokens in the sequence, which is useful for embedding tasks.
    """

    config_class = Mistral3BidirectionalConfig

    _export_as_stock_model = True
    _sentence_transformer_input_mode = "structured_multimodal"

    def get_hf_export_config(self) -> Mistral3Config:
        """Return the stock Transformers config used for portable inference export.

        Subclasses must explicitly re-assert ``_export_as_stock_model = True``.
        This prevents inherited export from silently discarding custom forward
        behavior that a stock ``Mistral3Model`` cannot reproduce.
        """
        if type(self).__dict__.get("_export_as_stock_model") is not True:
            raise TypeError(
                f"{type(self).__name__} must explicitly opt in to stock model export after verifying that "
                "its forward behavior is representable by Mistral3Model configuration and weights."
            )

        config_dict = self.config.to_dict()
        config_dict.pop("model_type", None)
        config_dict.pop("auto_map", None)
        config_dict.pop("task_instructions", None)
        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict):
            text_config.pop("auto_map", None)
            text_config["model_type"] = "ministral3"
            text_config["is_causal"] = False
        vision_config = config_dict.get("vision_config")
        if isinstance(vision_config, dict):
            vision_config.pop("auto_map", None)

        export_config = Mistral3Config.from_dict(config_dict)
        export_config.architectures = ["Mistral3Model"]
        export_config.is_causal = False
        export_config.text_config.is_causal = False
        return export_config

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Any:
        """Load stock Mistral3 weights into the bidirectional VL layout.

        Caller-provided key mappings take precedence over this model's default
        stock-checkpoint conversion.
        """
        key_mapping = dict(kwargs.pop("key_mapping", None) or {})
        # Stock Mistral3 wraps the text tower in an extra ``model`` module. This
        # custom base model owns the same tower directly, so mirror Transformers'
        # native Mistral3Model checkpoint conversion during loading only. An
        # explicit caller mapping for the same pattern takes precedence.
        key_mapping.setdefault(r"^language_model\.model\.", "language_model.")
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
        image_outputs = self.vision_tower(
            dummy_pixels,
            image_sizes=dummy_image_sizes,
            output_hidden_states=True,
            return_dict=True,
        )
        vision_feature_layer = self.config.vision_feature_layer
        if isinstance(vision_feature_layer, int):
            selected_image_features = image_outputs.hidden_states[vision_feature_layer]
        else:
            selected_image_features = torch.cat(
                [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer],
                dim=-1,
            )
        image_features = self.multi_modal_projector(selected_image_features.squeeze(0), dummy_image_sizes)
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
        force_dummy_vision: bool = False,
        **kwargs,
    ):
        """Keep dummy and real vision calls inside the same FSDP root forward.

        ``run_dummy_vision`` only ever applies in training mode. A frozen FSDP teacher runs in
        eval mode yet still needs the vision tower's all-gather issued on every rank, so
        ``force_dummy_vision`` opts into the same path regardless of ``self.training``.
        """
        if pixel_values is None and (force_dummy_vision or (self.training and run_dummy_vision is not False)):
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


def pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pool_type: str,
) -> torch.Tensor:
    """Pool token hidden states into one representation per input sequence.

    Args:
        last_hidden_states: Tensor of shape [batch, sequence, hidden].
        attention_mask: Tensor of shape [batch, sequence].
        pool_type: Pooling strategy. Supported values are ``"avg"``, ``"cls"``,
            and ``"last"``.

    Returns:
        Tensor of shape [batch, hidden].
    """
    masked_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        return masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pool_type == "cls":
        return masked_hidden_states[:, 0]
    if pool_type == "last":
        if attention_mask[:, -1].all():
            return masked_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(masked_hidden_states.shape[0], device=masked_hidden_states.device)
        return masked_hidden_states[batch_indices, sequence_lengths]
    raise ValueError(f"Unsupported pooling strategy: {pool_type!r}. Expected 'avg', 'cls', or 'last'.")


class Mistral3VLBidirectionalForSequenceClassification(Mistral3PreTrainedModel):
    """Bidirectional Mistral3 VLM with a sequence-scoring head."""

    config_class = Mistral3BidirectionalConfig
    base_model_prefix = "model"

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args: Any,
        **kwargs: Any,
    ) -> "Mistral3VLBidirectionalForSequenceClassification":
        """Load stock Mistral3 VLM weights into the reranker layout.

        Stock conditional-generation checkpoints nest the text backbone under
        ``model.language_model.model``. This reranker owns that backbone directly
        under ``model.language_model``, so checkpoint loading must remove the
        extra ``model`` segment.
        """
        key_mapping = dict(kwargs.pop("key_mapping", None) or {})
        key_mapping.setdefault(r"^language_model\.model\.", "language_model.")
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            key_mapping=key_mapping,
            **kwargs,
        )

    def __init__(self, config: Mistral3BidirectionalConfig) -> None:
        """Initialize the full vision-language cross-encoder.

        Args:
            config: Mistral3 vision-language configuration.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Mistral3BidirectionalModel(config)
        self.score = nn.Linear(config.text_config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def _nemo_apply_liger_kernel(self, liger_kernel_transformers) -> None:
        """Apply compatible Liger kernels to the nested Mistral text tower."""
        self.model._nemo_apply_liger_kernel(liger_kernel_transformers)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the scoring head in its configured dtype.

        Args:
            module: Module whose parameters are initialized in place.
        """
        if module is self.score:
            std = getattr(self.config.text_config, "initializer_range", 0.02) or 0.02
            init.normal_(module.weight, mean=0.0, std=std)
            return
        super()._init_weights(module)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        run_dummy_vision: bool | None = None,
        force_dummy_vision: bool = False,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast | tuple[torch.Tensor, ...]:
        """Score each multimodal question-document sequence.

        Args:
            input_ids: Tensor of shape [batch, sequence] containing text and image
                placeholder token IDs.
            pixel_values: Tensor of shape [images, channels, height, width].
            image_sizes: Tensor of shape [images, 2] containing image height and
                width before padding.
            attention_mask: Tensor of shape [batch, sequence].
            position_ids: Tensor of shape [batch, sequence].
            past_key_values: Optional transformer key/value cache.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            labels: Tensor of shape [query_batch] containing the relevant
                candidate index for each query. Accepted for compatibility but
                intentionally unused because the recipe computes the grouped
                cross-entropy loss.
            use_cache: Whether to return updated key/value cache state.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return intermediate hidden states.
            return_dict: Whether to return a structured model output.
            run_dummy_vision: Whether text-only training batches should run the
                zero-weighted dummy vision path for distributed gradient coverage.
            force_dummy_vision: Run that same path for text-only batches even in eval mode,
                which a frozen FSDP teacher needs to keep its all-gathers aligned across ranks.
            **kwargs: Additional arguments forwarded to ``Mistral3BidirectionalModel``.

        Returns:
            A sequence-classifier output whose ``logits`` tensor has shape
            [batch, num_labels], or the equivalent tuple when ``return_dict`` is
            false.
        """
        if attention_mask is None:
            raise ValueError("attention_mask is required for sequence pooling.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
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
            run_dummy_vision=run_dummy_vision,
            force_dummy_vision=force_dummy_vision,
            **kwargs,
        )
        pooled_hidden_states = pool(
            last_hidden_states=outputs[0],
            attention_mask=attention_mask,
            pool_type=self.config.pooling,
        )
        logits = self.score(pooled_hidden_states) / self.config.temperature

        if not return_dict:
            return (logits,) + outputs[1:]

        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Export for ModelRegistry auto-discovery
ModelClass = [
    Ministral3BidirectionalModel,
    Mistral3BidirectionalModel,
    Mistral3VLBidirectionalForSequenceClassification,
]


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
    try:
        AutoModelForSequenceClassification.register(
            Mistral3BidirectionalConfig,
            Mistral3VLBidirectionalForSequenceClassification,
        )
    except ValueError:
        pass  # Already registered


_register_with_hf_auto_classes()

__all__ = [
    "Ministral3BidirectionalModel",
    "Ministral3BidirectionalConfig",
    "Mistral3BidirectionalModel",
    "Mistral3BidirectionalConfig",
    "Mistral3VLBidirectionalForSequenceClassification",
    "ModelClass",
]
