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
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, PretrainedConfig
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3Model,
    Mistral3ModelOutputWithPast,
    Mistral3PreTrainedModel,
)
from transformers.utils import logging

if TYPE_CHECKING:
    from .reranker_export import Mistral3RerankerMetadataExporter


logger = logging.get_logger(__name__)


class Ministral3BidirectionalConfig(Ministral3Config):
    """Configuration for Ministral3BidirectionalModel with pooling and temperature settings."""

    model_type = "ministral3_bidirec"

    def __init__(
        self,
        pooling: str = "avg",
        temperature: float = 1.0,
        is_causal: bool = False,
        **kwargs: Any,
    ) -> None:
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(is_causal=is_causal, **kwargs)


class Ministral3BidirectionalModel(Ministral3Model):
    """
    Ministral3 retrieval model with a configurable attention mode.

    The historical class identity is retained for legacy checkpoint compatibility.
    """

    config_class = Ministral3BidirectionalConfig

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: Ministral3BidirectionalConfig) -> None:
        super().__init__(config)
        # Transformers mask builders read this per-attention flag; dual-mode parity tests guard that upstream contract.
        is_causal = getattr(config, "is_causal", False)
        config.is_causal = is_causal
        for layer in self.layers:
            layer.self_attn.is_causal = is_causal

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Forward pass with the attention mode stored in the model config.

        Causal mode delegates to the Hugging Face parent implementation. Non-causal
        mode uses the retrieval-specific bidirectional mask.

        Args:
            input_ids: Integer token IDs [batch, sequence], exclusive with inputs_embeds.
            attention_mask: Optional padding mask [batch, sequence], with zero for padding.
            position_ids: Integer positions [batch, sequence] or [1, sequence].
            past_key_values: Optional Transformers cache in its native layout.
            inputs_embeds: Floating-point embeddings [batch, sequence, hidden].
            use_cache: Whether to populate the returned cache.
            cache_position: Integer cache positions [sequence].
            **kwargs: Additional Hugging Face forward options.

        Returns:
            Output with floating-point last_hidden_state [batch, sequence, hidden],
            where hidden is config.hidden_size, and the optional Transformers cache.
            In causal mode, optional hidden states and attentions follow the Hugging
            Face Ministral3Model.forward output contract.
        """
        if getattr(self.config, "is_causal", False):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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
                cache_position=cache_position,
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

    def __init__(
        self,
        pooling: str = "avg",
        temperature: float | None = None,
        score_temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        # Released rerankers separate scoring from generation temperature; legacy checkpoints use temperature.
        temperature = score_temperature if score_temperature is not None else temperature
        temperature = 1.0 if temperature is None else temperature
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
    Mistral3 retrieval model with a configurable Ministral3 text tower.

    The historical class identity is retained while the text tower honors either policy.
    """

    config_class = Mistral3BidirectionalConfig
    # Resolve only when mining/evaluating: portable model exports must not import AutoModel.
    retrieval_processor_target: str = (
        "nemo_automodel.components.models.ministral_bidirectional.processor.Mistral3BiEncoderProcessor"
    )

    _export_as_stock_model = True
    _sentence_transformer_input_mode = "structured_multimodal"

    @classmethod
    def supports_config(cls, config: PretrainedConfig) -> bool:
        """Select this implementation only for Ministral3 text towers.

        Args:
            config: Checkpoint configuration before any retrieval conversion.

        Returns:
            Whether the checkpoint has a supported text architecture.
        """
        return config.get_text_config().model_type in {"ministral3", "ministral3_bidirec"}

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
        # Causality is a text-tower policy. A top-level value is also forwarded to Pixtral vision attention.
        config_dict.pop("is_causal", None)
        text_config = config_dict.get("text_config")
        is_causal = False
        if isinstance(text_config, dict):
            text_config.pop("auto_map", None)
            text_config["model_type"] = "ministral3"
            is_causal = bool(text_config.get("is_causal", False))
        vision_config = config_dict.get("vision_config")
        if isinstance(vision_config, dict):
            vision_config.pop("auto_map", None)

        export_config = Mistral3Config.from_dict(config_dict)
        export_config.architectures = ["Mistral3Model"]
        export_config.text_config.is_causal = is_causal
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
    ) -> "Mistral3BidirectionalModel":
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

    def __init__(self, config: Mistral3BidirectionalConfig) -> None:
        """Build the Pixtral vision tower, projector, and retrieval text tower.

        Args:
            config: Composite Mistral3 vision-language configuration.
        """
        super().__init__(config)
        # AutoModel already resolves our registered text config to the retrieval tower.
        # Only callers supplying a stock text config object need a replacement.
        if not isinstance(self.language_model, Ministral3BidirectionalModel):
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
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...] | Mistral3ModelOutputWithPast:
        """Run text or multimodal inputs through one FSDP root forward.

        Args:
            input_ids: Tensor of shape [batch, sequence] containing text and image
                placeholder token IDs.
            pixel_values: Tensor of shape [images, channels, height, width].
            image_sizes: Tensor of shape [images, 2] containing image height and width.
            attention_mask: Tensor of shape [batch, sequence].
            position_ids: Tensor of shape [batch, sequence].
            past_key_values: Optional cache with per-layer key and value tensors of
                shape [batch, kv_heads, cached_sequence, head_dim], where kv_heads
                is the number of key/value heads and cached_sequence is the stored
                token count. The cache is updated in place when used.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            use_cache: Whether to return updated key/value cache state.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return intermediate hidden states.
            return_dict: Whether to return a structured model output.
            run_dummy_vision: Whether to execute a zero-weighted vision path for
                text-only inputs. Defaults to enabled in training. Sharded vision
                layers always participate during evaluation, even when false, to
                preserve collective ordering across ranks with different modalities.
            **kwargs: Additional arguments forwarded to the stock Mistral3 model.

        Returns:
            A ``Mistral3ModelOutputWithPast`` with these fields:

            - ``last_hidden_state``: Tensor of shape [batch, sequence, hidden].
            - ``past_key_values``: Optional updated cache with per-layer key and
              value tensors of shape [batch, kv_heads, cached_sequence, head_dim].
            - ``hidden_states``: Optional tuple of embedding and layer-output
              tensors, each of shape [batch, sequence, hidden].
            - ``attentions``: Optional tuple of per-layer attention tensors of
              shape [batch, heads, sequence, key_sequence], where key_sequence is
              the number of attended key positions, including cached positions.
            - ``image_hidden_states``: Optional projected image tensor of shape
              [image_tokens, hidden], where image_tokens is the total token count
              flattened across all images. Present only for real image inputs.

            When ``return_dict`` is false, return these fields in the listed order,
            omitting fields whose value is ``None``.
        """
        needs_dummy_vision = self.training and run_dummy_vision is not False
        if pixel_values is None and not self.training:
            needs_dummy_vision = run_dummy_vision is True or any(
                isinstance(module, FSDPModule) for module in self.vision_tower.modules()
            )
        if needs_dummy_vision and pixel_values is None:
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


class _Float32ScoringHead(nn.Linear):
    """Keep scoring arithmetic in FP32 while retaining checkpoint parameter dtypes."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project pooled states without autocast rounding the scores.

        Args:
            input: Pooled hidden states of shape [batch, hidden].

        Returns:
            FP32 scores of shape [batch, num_labels]. Gradients flow to the
            original input and weight tensors in their respective dtypes.
        """
        with torch.autocast(device_type=input.device.type, enabled=False):
            return nn.functional.linear(input.float(), self.weight.float())


class Mistral3VLBidirectionalForSequenceClassification(Mistral3PreTrainedModel):
    """Bidirectional Mistral3 VLM with a sequence-scoring head."""

    config_class = Mistral3BidirectionalConfig
    base_model_prefix = "model"

    @classmethod
    def supports_config(cls, config: PretrainedConfig) -> bool:
        """Return whether the checkpoint uses the supported Ministral3 text tower.

        Args:
            config: Checkpoint configuration before retrieval conversion.

        Returns:
            Whether the checkpoint can use this scoring backbone.
        """
        return Mistral3BidirectionalModel.supports_config(config)

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    @property
    def effective_score_temperature(self) -> float:
        """Return the temperature applied to score logits by this model."""
        return float(self.config.temperature)

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
        key_mapping.setdefault(r"^language_model\.score\.weight$", "score.weight")
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            key_mapping=key_mapping,
            **kwargs,
        )

    _sentence_transformer_input_mode = "structured_multimodal"

    def __init__(self, config: Mistral3BidirectionalConfig) -> None:
        """Initialize the full vision-language cross-encoder.

        Args:
            config: Mistral3 vision-language configuration.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Mistral3BidirectionalModel(config)
        self.score = _Float32ScoringHead(config.text_config.hidden_size, self.num_labels, bias=False)
        self.post_init()
        from .reranker_export import Mistral3RerankerStateDictAdapter

        self.state_dict_adapter = Mistral3RerankerStateDictAdapter()

    def _get_consolidated_hf_metadata_exporter(
        self, *, tokenizer: object, original_model_path: str | None
    ) -> "Mistral3RerankerMetadataExporter":
        """Select the portable reranker export for direct and distributed checkpoint saves."""
        from .reranker_export import Mistral3RerankerMetadataExporter

        return Mistral3RerankerMetadataExporter(self)

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs: Any) -> None:
        """Save trained weights using the shared Transformers/vLLM reranker head name."""
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
        super().save_pretrained(save_directory, state_dict=self.state_dict_adapter.to_hf(state_dict), **kwargs)
        from .reranker_export import Mistral3RerankerMetadataExporter

        Mistral3RerankerMetadataExporter(self).save_model_assets(save_directory)

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
        **kwargs: Any,
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
            past_key_values: Optional cache with per-layer key and value tensors of
                shape [batch, kv_heads, cached_sequence, head_dim], where kv_heads
                is the number of key/value heads and cached_sequence is the stored
                token count. The cache is updated in place when used.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            labels: Tensor of shape [query_batch] containing the relevant
                candidate index for each query. Accepted for compatibility but
                intentionally unused because the recipe computes the grouped
                cross-entropy loss.
            use_cache: Whether to return updated key/value cache state.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return intermediate hidden states.
            return_dict: Whether to return a structured model output.
            run_dummy_vision: Whether text-only inputs should run dummy vision.
                Sharded vision always participates during evaluation to keep
                collectives aligned across ranks with different modalities.
            **kwargs: Additional arguments forwarded to ``Mistral3BidirectionalModel``.

        Returns:
            A ``SequenceClassifierOutputWithPast`` with these fields:

            - ``loss``: Always ``None``; the recipe computes the loss.
            - ``logits``: FP32 tensor of shape [batch, num_labels].
            - ``past_key_values``: Optional updated cache with per-layer key and
              value tensors of shape [batch, kv_heads, cached_sequence, head_dim].
            - ``hidden_states``: Optional tuple of embedding and layer-output
              tensors, each of shape [batch, sequence, hidden].
            - ``attentions``: Optional tuple of per-layer attention tensors of
              shape [batch, heads, sequence, key_sequence], where key_sequence is
              the number of attended key positions, including cached positions.

            When ``return_dict`` is false, return ``logits`` followed by the
            backbone's optional ``past_key_values``, ``hidden_states``,
            ``attentions``, and ``image_hidden_states``, in that order, omitting
            ``None`` values. Unlike the structured output, this tuple includes
            projected image features for real image inputs: a tensor of shape
            [image_tokens, hidden], flattened across all images.
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
            **kwargs,
        )
        pooled_hidden_states = pool(
            last_hidden_states=outputs[0],
            attention_mask=attention_mask,
            pool_type=self.config.pooling,
        )
        # The reference checkpoint projects and scales scores in FP32, including under autocast.
        # The recipe must not apply a second non-unit temperature.
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
