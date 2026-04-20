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
"""LLaVA-OneVision-1.5 model implementation.

Matches the layout of lmms-lab/LLaVA-OneVision-1.5-*-{Base,Instruct} so that
HF safetensors load into this module tree via LlavaOneVisionStateDictAdapter
with only regex-renames (no tensor transforms).

In-memory tree:
  model.visual.*           (RiceTransformer)
  model.language_model.*   (transformers.Qwen3Model — LLaVA-OV-1.5's text backbone is Qwen3)
  lm_head.*                (nn.Linear)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.llava_onevision.rice_vit import RiceTransformer

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class RiceConfig(PretrainedConfig):
    """Configuration for the Rice ViT vision tower."""

    model_type = "rice_vit"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 24,
        embed_dim: int = 1024,
        hidden_size: int = 1024,
        hidden_act: str = "gelu",
        intermediate_size: int = 4096,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        text_hidden_size: int = 2560,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.text_hidden_size = text_hidden_size


class Llavaonevision1_5Config(PretrainedConfig):
    """Top-level config for LLaVA-OneVision-1.5.

    ``model_type`` matches the on-hub value exactly so ``AutoConfig.from_pretrained``
    resolves to this class without ``trust_remote_code`` once registered.
    """

    model_type = "llavaonevision1_5"
    sub_configs = {"vision_config": RiceConfig}

    def __init__(
        self,
        text_config: Optional[Union[Dict, PretrainedConfig]] = None,
        vision_config: Optional[Union[Dict, RiceConfig]] = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        vocab_size: int = 152064,
        architectures: Optional[List[str]] = None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = RiceConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = RiceConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = _build_text_config(text_config)
        elif text_config is None:
            self.text_config = _build_text_config({})
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vocab_size = vocab_size

        if architectures is None:
            architectures = ["LLaVAOneVision1_5_ForConditionalGeneration"]

        super().__init__(architectures=architectures, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output


def _build_text_config(data: Dict[str, Any]) -> PretrainedConfig:
    """Coerce a text_config dict from HF (or user) into a Qwen3Config.

    LLaVA-OV-1.5's text backbone is Qwen3 (q/k norm, GQA, standard SiLU MLP).
    On-hub ``model_type`` is ``LLaVAOneVision1_5_text``; we drop it so Qwen3Config
    doesn't reject the kwargs.
    """
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    data = dict(data)
    # ``model_type`` would conflict (Qwen3Config sets its own); the rest of the
    # fields (layer_types, use_sliding_window, rope_*) are accepted as kwargs.
    data.pop("model_type", None)
    data.pop("base_config_key", None)
    # These live on the top-level LLaVAOV config, not the text backbone.
    data.pop("image_token_id", None)
    data.pop("video_token_id", None)
    return Qwen3Config(**data)


def _coerce_text_config(tc: Any) -> PretrainedConfig:
    """Accept a raw HF remote-code text config and return a Qwen3Config.

    The constructor path for NeMo custom models is ``cls(hf_config)`` where
    ``hf_config`` may be the remote-code ``Llavaonevision1_5Config`` whose
    ``text_config`` is a ``LLaVAOneVision1_5_TextConfig`` instance. Normalize
    to Qwen3Config so the inner ``Qwen3Model`` gets fields it understands.
    """
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    if isinstance(tc, Qwen3Config):
        return tc
    if isinstance(tc, PretrainedConfig):
        return _build_text_config(tc.to_dict())
    if isinstance(tc, dict):
        return _build_text_config(tc)
    raise TypeError(f"Unsupported text_config type: {type(tc)!r}")


def _coerce_vision_config(vc: Any) -> RiceConfig:
    if isinstance(vc, RiceConfig):
        return vc
    if isinstance(vc, PretrainedConfig):
        data = vc.to_dict()
        data.pop("model_type", None)
        data.pop("base_config_key", None)
        return RiceConfig(**data)
    if isinstance(vc, dict):
        data = dict(vc)
        data.pop("model_type", None)
        data.pop("base_config_key", None)
        return RiceConfig(**data)
    raise TypeError(f"Unsupported vision_config type: {type(vc)!r}")


# =============================================================================
# Model
# =============================================================================


class LLaVAOneVision1_5_Model(nn.Module):
    """Combined vision + language backbone. Returns last_hidden_state."""

    def __init__(self, config: Llavaonevision1_5Config, attn_implementation: str = "eager"):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

        self.config = config
        self.vision_config = _coerce_vision_config(config.vision_config)
        self.text_config = _coerce_text_config(config.text_config)

        # Propagate attn_implementation to the text model; leave ViT's internal
        # choice up to RiceTransformer (the HF checkpoint forward shape matches
        # whichever variant is used — only the mask/FA2 kernel differs).
        if attn_implementation is not None:
            try:
                self.text_config._attn_implementation = attn_implementation
            except Exception:
                pass

        self.visual = RiceTransformer(self.vision_config, attn_implementation=attn_implementation or "eager")
        self.language_model = Qwen3Model(self.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor) -> torch.Tensor:
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)

                image_token_id = self.config.image_token_id
                n_image_tokens = (input_ids == image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: "
                        f"tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                raise NotImplementedError("Video inputs are not yet supported in the NeMo LLaVA-OV-1.5 impl.")

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        return outputs


class LLaVAOneVision1_5_ForConditionalGeneration(HFCheckpointingMixin, nn.Module):
    """LLaVA-OneVision-1.5 for conditional generation (Rice ViT + Qwen3 text)."""

    config_class = Llavaonevision1_5Config

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        if attn_implementation is None:
            attn_implementation = getattr(config, "_attn_implementation", None) or "eager"
        self.model = LLaVAOneVision1_5_Model(config, attn_implementation=attn_implementation)
        self.lm_head = nn.Linear(self.model.text_config.hidden_size, self.model.text_config.vocab_size, bias=False)
        self.vocab_size = self.model.text_config.vocab_size
        self.image_token_id = getattr(config, "image_token_id", 151655)
        self.video_token_id = getattr(config, "video_token_id", 151656)

        # Lazy-initialized on first state_dict IO
        self.state_dict_adapter = None

        # Initialize adapter lazily — import is cheap but keeps module import clean
        from nemo_automodel.components.models.llava_onevision.state_dict_adapter import (
            LlavaOneVisionStateDictAdapter,
        )

        self.state_dict_adapter = LlavaOneVisionStateDictAdapter(config)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def visual(self):
        return self.model.visual

    @property
    def language_model(self):
        return self.model.language_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Labels arrive pre-shifted from the collate_fn; no additional shift here.
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1).to(logits.device),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
