# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.
"""
Bidirectional Ministral3 model for embedding tasks.

This module provides a modified Ministral3Model that uses bidirectional (non-causal)
attention, suitable for generating embeddings where each token should attend
to all other tokens in the sequence.
"""

import torch
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Ministral3BidirectionalConfig(Ministral3Config):
    """Configuration for Ministral3BidirectionalModel with pooling and temperature settings."""

    model_type = "ministral3_bidirec"

    def __init__(
        self, pooling: str = "avg", temperature: float = 1.0, **kwargs
    ) -> None:
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

    Loading from a Mistral3 VLM checkpoint (e.g. mistralai/Ministral-3-3B-Base-2512 or
    mistralai/Ministral-3-3B-Instruct-2512) is handled automatically: the language
    model weights are extracted from the VLM.

    Text-only checkpoints (e.g. mistralai/Ministral-3B-Instruct) load with the usual
    ``from_pretrained`` path when the Hub config has no nested ``text_config``.
    """

    config_class = Ministral3BidirectionalConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load from either a bidirectional text checkpoint or a Mistral3 VLM checkpoint.

        When the checkpoint is a VLM (config has text_config), the language model
        weights are extracted automatically. Otherwise, the checkpoint is loaded
        directly as a text-only model.
        """
        logger.info(f"Loading Ministral3BidirectionalModel from {pretrained_model_name_or_path}")
        hub_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

        text_cfg = getattr(hub_config, "text_config", None)
        if text_cfg is not None:
            return cls._from_vlm_checkpoint(
                pretrained_model_name_or_path, hub_config, **kwargs
            )
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    @classmethod
    def _from_vlm_checkpoint(cls, model_name_or_path, vlm_config, **kwargs):
        """Extract and load the language model from a VLM checkpoint."""
        torch_dtype = kwargs.get("torch_dtype", None)
        attn_implementation = kwargs.get("attn_implementation", None)

        logger.info(
            f"Loading VLM from {model_name_or_path} to extract language model weights"
        )
        vlm = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        state_dict = vlm.language_model.state_dict()
        del vlm

        text_config_dict = vlm_config.text_config.to_dict()
        text_config = cls.config_class(**text_config_dict)
        if attn_implementation is not None:
            text_config._attn_implementation = attn_implementation

        model = cls(text_config)
        model.load_state_dict(state_dict)
        if torch_dtype is not None:
            dtype = (
                getattr(torch, torch_dtype)
                if isinstance(torch_dtype, str)
                else torch_dtype
            )
            model = model.to(dtype)

        return model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Forward pass with bidirectional attention.

        Identical to Ministral3Model.forward() except the causal mask is replaced
        with a bidirectional mask, allowing all tokens to attend to each other.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=inputs_embeds,
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


# Export for ModelRegistry auto-discovery
ModelClass = [Ministral3BidirectionalModel]


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


_register_with_hf_auto_classes()

__all__ = [
    "Ministral3BidirectionalModel",
    "Ministral3BidirectionalConfig",
    "ModelClass",
]
