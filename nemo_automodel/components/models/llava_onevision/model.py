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
"""LLaVA-OneVision-1.5 model implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.llava_onevision.rice_vit import (
    RiceBlock,
    RicePatchEmbed,
    RicePatchMerger,
    RiceRotaryEmbedding,
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================


class RiceConfig(PretrainedConfig):
    """Configuration for Rice ViT encoder."""

    model_type = "rice_vit"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        text_hidden_size: int = 2560,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.text_hidden_size = text_hidden_size


class LlavaOneVisionConfig(PretrainedConfig):
    """Configuration for LLaVA-OneVision-1.5 model."""

    model_type = "llava_onevision"

    def __init__(
        self,
        vision_config: Optional[Union[Dict, RiceConfig]] = None,
        text_config: Optional[Union[Dict, PretrainedConfig]] = None,
        ignore_index: int = -100,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        pad_token_id: int = 0,
        architectures: Optional[List[str]] = None,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = RiceConfig()
        elif isinstance(vision_config, dict):
            vision_config = RiceConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

            text_config = Qwen2Config()
        elif isinstance(text_config, dict):
            from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

            text_config = Qwen2Config(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        if architectures is None:
            architectures = ["LlavaOneVisionForConditionalGeneration"]

        super().__init__(pad_token_id=pad_token_id, architectures=architectures, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output


# =============================================================================
# Rice Vision Transformer
# =============================================================================


class RiceTransformer(nn.Module):
    """Rice ViT transformer with 2D RoPE and patch merging."""

    def __init__(self, config: RiceConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.patch_embed = RicePatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = RiceRotaryEmbedding(head_dim // 2)

        scale = config.hidden_size**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(config.hidden_size))
        self.class_pos_emb = nn.Parameter(torch.randn(1, head_dim // 2))

        self.blocks = nn.ModuleList(
            [RiceBlock(config) for _ in range(config.depth)]
        )
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.merger = RicePatchMerger(
            dim=config.text_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute 2D rotary position embeddings for variable-size grids."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for Rice ViT.

        Args:
            pixel_values: Flattened pixel values [num_patches, C*P*P]
            grid_thw: Grid dimensions [num_images, 3] as (T, H, W)

        Returns:
            Image embeddings [total_merged_patches, text_hidden_size]
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # Add class token embedding
        batch_size = hidden_states.shape[0]
        class_emb = self.class_embedding.expand(batch_size, -1)
        class_pos = self.class_pos_emb.expand(batch_size, -1)
        hidden_states = torch.cat([class_emb, hidden_states], dim=0)

        # Compute cu_seqlens for block-diagonal attention
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=grid_thw.device),
                grid_thw[:, 1].prod(dim=-1).cumsum(dim=0),
            ]
        ).to(dtype=torch.int32)

        # Compute rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Forward through blocks
        hidden_states = self.pre_layernorm(hidden_states)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        # Merge patches and project to text hidden size
        image_embeds = self.merger(hidden_states)
        return image_embeds


# =============================================================================
# LLaVA-OneVision Model
# =============================================================================


class LlavaOneVisionModel(nn.Module):
    """Base LLaVA-OneVision model without LM head."""

    def __init__(self, config: LlavaOneVisionConfig):
        super().__init__()
        self.config = config
        self.vision_tower = RiceTransformer(config.vision_config)
        self.language_model = self._build_language_model(config.text_config)

    def _build_language_model(self, text_config):
        """Build the Qwen3 language model."""
        # Import here to avoid circular imports
        try:
            from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM

            return Qwen2ForCausalLM.from_config(text_config)
        except ImportError:
            # Fallback to HF implementation
            from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

            return Qwen2ForCausalLM(text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, object]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                # Get image features
                image_embeds = self.vision_tower(pixel_values, image_grid_thw)

                # Scatter image features into input embeddings
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                # Similar for video (TODO: implement when needed)
                raise NotImplementedError("Video support not yet implemented")

        # Forward through language model
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return outputs


# =============================================================================
# LLaVA-OneVision For Conditional Generation
# =============================================================================


class LlavaOneVisionForConditionalGeneration(HFCheckpointingMixin, nn.Module):
    """LLaVA-OneVision-1.5 for conditional generation with Rice ViT + Qwen3."""

    @classmethod
    def from_config(
        cls,
        config: LlavaOneVisionConfig,
        **kwargs,
    ):
        return cls(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = LlavaOneVisionConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: LlavaOneVisionConfig,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        self.model = LlavaOneVisionModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

        self.vocab_size = config.text_config.vocab_size
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.image_token_id = getattr(config, "image_token_id", 151655)
        self.video_token_id = getattr(config, "video_token_id", 151656)

        # For pipeline parallelism chunking
        self._vlm_pixel_values_chunks = None
        self._vlm_image_grid_hws_chunks = None
        self._vlm_chunk_idx = 0

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @property
    def lm_head(self):
        return self._lm_head

    @lm_head.setter
    def lm_head(self, value):
        self._lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Handle pipeline parallelism chunks
        if (
            pixel_values is None
            and hasattr(self, "_vlm_pixel_values_chunks")
            and self._vlm_pixel_values_chunks is not None
        ):
            has_media_tokens = (
                input_ids is not None
                and self.image_token_id is not None
                and (input_ids == self.image_token_id).any()
            )
            if has_media_tokens:
                chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
                if chunk_idx < len(self._vlm_pixel_values_chunks):
                    pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                    image_grid_thw = self._vlm_image_grid_hws_chunks[chunk_idx]
                    self._vlm_chunk_idx = chunk_idx + 1

        # Forward through model
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:]
                shift_logits = shift_logits[shift_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_mask.to(labels.device) != 0].contiguous()

            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if return_dict is False:
            return (logits,)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
