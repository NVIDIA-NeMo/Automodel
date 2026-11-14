# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#
# The vision encoder, projector and multimodal merge are adapted from Hugging Face Transformers
# (Apache-2.0), https://github.com/huggingface/transformers at v5.23.0
# (52b080b6416a7897e2af90febeb4dfad22b84d19): src/transformers/models/lfm3_vl/modeling_lfm3_vl.py.
# Copyright 2026 The HuggingFace Team. All rights reserved.

"""LFM3-VL: the LFM3 decoder fed by a convolution-and-attention vision encoder through an MLP projector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.lfm2.modeling_lfm2 import Lfm2Attention, Lfm2RMSNorm, eager_attention_forward
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeMLP
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.vision_utils import get_vision_position_ids

from nemo_automodel.components.models.common import compute_lm_head_logits
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.lfm3.config import Lfm3VlConfig, Lfm3VlVisionConfig
from nemo_automodel.components.models.lfm3.model import Lfm3Model
from nemo_automodel.components.models.llama.rope_utils import apply_rotary_pos_emb

__all__ = ["Lfm3VlForConditionalGeneration"]


def axial_rotary_factors(position_ids: torch.Tensor, theta: float, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 2D rotary factors in float32: half of each head rotates by row, half by column.

    Args:
        position_ids: Long Tensor of shape [patches, 2] holding each patch's (row, column).
        theta: RoPE base.
        head_dim: Channels per attention head.

    Returns:
        Pair of float32 cosine and sine Tensors, each of shape [patches, head_dim].
    """
    axis_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float) / axis_dim))
    angles = (position_ids[..., None].float() * inv_freq.to(position_ids.device)).flatten(1)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


class Lfm3VlVisionAttention(Lfm2Attention):
    """LFM2 grouped-query attention made bidirectional, attending within one frame at a time."""

    def __init__(self, config: Lfm3VlVisionConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        frame_lengths: list[int],
    ) -> torch.Tensor:
        """Attend among the patches of each frame.

        Args:
            hidden_states: Tensor of shape [patches, hidden], the frames of every image
                and video concatenated.
            position_embeddings: Pair of float32 cosine and sine Tensors, each of shape
                [patches, head_dim].
            frame_lengths: Patch count of each frame, in order; sums to ``patches``.

        Returns:
            Tensor of shape [patches, hidden].
        """
        patches = hidden_states.shape[0]
        query = self.q_layernorm(self.q_proj(hidden_states).view(patches, -1, self.head_dim))
        key = self.k_layernorm(self.k_proj(hidden_states).view(patches, -1, self.head_dim))
        value = self.v_proj(hidden_states).view(patches, -1, self.head_dim)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        # [patches, heads, head_dim] -> one [1, heads, frame_patches, head_dim] group per frame
        frames = zip(
            *(states.transpose(0, 1).unsqueeze(0).split(frame_lengths, dim=2) for states in (query, key, value))
        )
        attended = [
            attention_interface(self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0, is_causal=False)[
                0
            ]
            for q, k, v in frames
        ]
        return self.out_proj(torch.cat(attended, dim=1).reshape(patches, -1))


class Lfm3VlVisionShortConv(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(self, config: Lfm3VlVisionConfig) -> None:
        super().__init__()
        self.merge_size = config.spatial_merge_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        """Gate, convolve each frame over its patch grid, and gate again.

        Args:
            hidden_states: Tensor of shape [patches, hidden]. Within a frame the patches
                are ordered merge block by merge block, as the processor emits them.
            grid_thw: Per image or video, its patch grid as (frames, height, width).

        Returns:
            Tensor of shape [patches, hidden] in the input order.
        """
        input_gate, output_gate, values = self.in_proj(hidden_states).chunk(3, dim=-1)
        gated = input_gate * values
        merge, channels = self.merge_size, gated.shape[-1]
        mixed = []
        lengths = [frames * height * width for frames, height, width in grid_thw]
        for item, (frames, height, width) in zip(gated.split(lengths), grid_thw):
            # merge-block order -> [frames, channels, height, width] rasters, and back after the convolution
            item = item.view(frames, height // merge, width // merge, merge, merge, channels)
            item = self.spatial_conv(item.permute(0, 5, 1, 3, 2, 4).reshape(frames, channels, height, width))
            item = item.view(frames, channels, height // merge, merge, width // merge, merge)
            mixed.append(item.permute(0, 2, 4, 3, 5, 1).reshape(-1, channels))
        return self.out_proj(output_gate * torch.cat(mixed))


class Lfm3VlVisionLayer(GradientCheckpointingLayer):
    """Pre-norm operator (per-frame attention or 2D convolution) followed by a pre-norm MLP."""

    def __init__(self, config: Lfm3VlVisionConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] != "conv"
        if self.is_attention_layer:
            self.self_attn = Lfm3VlVisionAttention(config, layer_idx)
        else:
            self.conv = Lfm3VlVisionShortConv(config)
        self.feed_forward = Lfm2MoeMLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        grid_thw: list[list[int]],
        frame_lengths: list[int],
    ) -> torch.Tensor:
        """Run one vision layer.

        Args:
            hidden_states: Tensor of shape [patches, hidden].
            position_embeddings: Pair of float32 cosine and sine Tensors, each of shape
                [patches, head_dim].
            grid_thw: Per image or video, its patch grid as (frames, height, width).
            frame_lengths: Patch count of each frame, in order; sums to ``patches``.

        Returns:
            Tensor of shape [patches, hidden].
        """
        operator_input = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            operator_output = self.self_attn(operator_input, position_embeddings, frame_lengths)
        else:
            operator_output = self.conv(operator_input, grid_thw)
        hidden_states = hidden_states + operator_output
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm3VlPreTrainedModel(PreTrainedModel):
    """Weight initialization and loading shared by the LFM3-VL models."""

    config_class = Lfm3VlConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm3VlVisionLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True


class Lfm3VlVisionModel(Lfm3VlPreTrainedModel):
    """LFM3-VL vision encoder over the flattened patches of images and video frames."""

    config_class = Lfm3VlVisionConfig
    input_modalities = ("image", "video")

    def __init__(self, config: Lfm3VlVisionConfig) -> None:
        super().__init__(config)
        patch_features = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patchifier = nn.Linear(patch_features, config.hidden_size, bias=False)
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = nn.ModuleList(Lfm3VlVisionLayer(config, idx) for idx in range(config.num_hidden_layers))
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode patches.

        Args:
            pixel_values: Tensor of shape [patches, in_channels * temporal_patch_size * patch_size**2],
                the flattened patches of every image or video. Within a frame they are
                ordered merge block by merge block.
            grid_thw: Long Tensor of shape [items, 3] giving each image's or video's
                patch grid as (frames, height, width).

        Returns:
            Tensor of shape [patches, hidden] in the input order.
        """
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        position_ids = get_vision_position_ids(grid_thw, self.config.spatial_merge_size)
        position_embeddings = axial_rotary_factors(position_ids, self.config.rope_parameters["rope_theta"], head_dim)
        grid = grid_thw.tolist()
        frame_lengths = [height * width for frames, height, width in grid for _ in range(frames)]

        hidden_states = self.embedding_norm(self.patchifier(pixel_values))
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, grid, frame_lengths)
        return hidden_states


class Lfm3VlMultiModalProjector(nn.Module):
    """Normalize vision tokens, merge each ``spatial_merge_size`` square of them, and map to the text width."""

    def __init__(self, config: Lfm3VlConfig) -> None:
        super().__init__()
        vision = config.vision_config
        self.merged_size = vision.hidden_size * vision.spatial_merge_size**2
        self.layer_norm = nn.LayerNorm(vision.hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(self.merged_size, config.projector_hidden_size, bias=config.projector_bias)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project vision tokens to language tokens.

        Args:
            image_features: Tensor of shape [patches, vision_hidden], where consecutive runs
                of ``spatial_merge_size**2`` patches form one merge block. Cast to the
                projector's dtype: a frozen vision tower may be stored in lower precision.

        Returns:
            Tensor of shape [patches / spatial_merge_size**2, text_hidden].
        """
        image_features = image_features.to(self.layer_norm.weight.dtype)
        merged = self.layer_norm(image_features).reshape(-1, self.merged_size)
        return self.linear_2(self.act(self.linear_1(merged)))


class Lfm3VlModel(Lfm3VlPreTrainedModel):
    """LFM3 decoder whose image and video placeholder tokens are replaced by projected vision features."""

    def __init__(self, config: Lfm3VlConfig) -> None:
        super().__init__(config)
        self.language_model = Lfm3Model(config.text_config)
        self.vision_tower = Lfm3VlVisionModel(config.vision_config)
        self.multi_modal_projector = Lfm3VlMultiModalProjector(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding table."""
        return self.language_model.embed_tokens

    @can_return_tuple
    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> BaseModelOutputWithPooling:
        """Encode images into language-model tokens.

        Args:
            pixel_values: Tensor of shape [patches, in_channels * temporal_patch_size * patch_size**2].
            image_grid_thw: Long Tensor of shape [images, 3] giving each patch grid as
                (frames, height, width).

        Returns:
            Output whose ``last_hidden_state`` is the vision Tensor of shape
            [patches, vision_hidden] and whose ``pooler_output`` is a tuple with one
            Tensor of shape [image_tokens, text_hidden] per image.
        """
        hidden_states = self.vision_tower(pixel_values, image_grid_thw)
        tokens = (image_grid_thw.prod(-1) // self.config.vision_config.spatial_merge_size**2).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states, pooler_output=self.multi_modal_projector(hidden_states).split(tokens)
        )

    @can_return_tuple
    def get_video_features(
        self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor
    ) -> BaseModelOutputWithPooling:
        """Encode videos into language-model tokens; same layouts as ``get_image_features``, per video."""
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def _fill_placeholders(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor | None,
        token_id: int,
        features: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Write vision features over the embeddings of one modality's placeholder tokens.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            input_ids: Long Tensor of shape [batch, sequence] locating the placeholders.
            token_id: The modality's placeholder token.
            features: One Tensor of shape [tokens, hidden] per image or video, in the
                order their placeholders appear row by row.

        Returns:
            New Tensor of shape [batch, sequence, hidden]; ``inputs_embeds`` is not modified.
        """
        if input_ids is None:
            raise ValueError("Image and video inputs need `input_ids` to locate their placeholder tokens.")
        placeholders = input_ids == token_id
        stacked = torch.cat(features).to(inputs_embeds.device, inputs_embeds.dtype)
        if placeholders.sum() != stacked.shape[0]:
            raise ValueError(
                f"Found {placeholders.sum()} placeholder tokens with id {token_id} for {stacked.shape[0]} vision features."
            )
        return inputs_embeds.masked_scatter(placeholders.unsqueeze(-1), stacked)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Encode any images and videos, splice them into the text, and run the decoder.

        Args:
            input_ids: Long Tensor of shape [batch, sequence]. Exclusive with ``inputs_embeds``.
            attention_mask: Optional padding Tensor of shape [batch, key_sequence], where
                ``key_sequence`` also counts cached tokens.
            position_ids: Optional long Tensor of shape [batch, sequence].
            past_key_values: Optional generation cache, updated in place.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            pixel_values: Image patches, Tensor of shape
                [patches, in_channels * temporal_patch_size * patch_size**2].
            image_grid_thw: Long Tensor of shape [images, 3], each patch grid as (frames, height, width).
            pixel_values_videos: Video patches, laid out like ``pixel_values``.
            video_grid_thw: Long Tensor of shape [videos, 3], laid out like ``image_grid_thw``.
            use_cache: Whether to create and return a generation cache.
            mm_encoder_outputs: Results of ``get_image_features`` and ``get_video_features``
                under ``"image"`` and ``"video"``, as generation precomputes them. Exclusive
                with the pixel inputs.
            **kwargs: Hugging Face output-capture and attention arguments for the decoder.

        Returns:
            Output whose ``last_hidden_state`` is a Tensor of shape [batch, sequence, hidden].
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if mm_encoder_outputs is not None and (pixel_values is not None or pixel_values_videos is not None):
            raise ValueError("Pass either pixel inputs or `mm_encoder_outputs`, not both.")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        encoded = dict(mm_encoder_outputs or {})
        if pixel_values is not None:
            encoded["image"] = self.get_image_features(pixel_values, image_grid_thw)
        if pixel_values_videos is not None:
            encoded["video"] = self.get_video_features(pixel_values_videos, video_grid_thw)
        for modality, token_id in (("image", self.config.image_token_id), ("video", self.config.video_token_id)):
            if encoded.get(modality) is not None:
                inputs_embeds = self._fill_placeholders(
                    inputs_embeds, input_ids, token_id, encoded[modality].pooler_output
                )

        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class Lfm3VlForConditionalGeneration(HFCheckpointingMixin, Lfm3VlPreTrainedModel, GenerationMixin):
    """LFM3-VL with a vocabulary head, loadable from LiquidAI's released checkpoints."""

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallelism is undeclared until corresponding recipes are verified."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: Lfm3VlConfig) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.model = Lfm3VlModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def tie_weights(self, **kwargs: bool | set[str] | None) -> None:
        """Share the vocabulary head with the token embeddings when the config asks for it."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding table."""
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the token embedding table and retie the head."""
        self.model.language_model.embed_tokens = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        """Return the vocabulary head."""
        return self.lm_head

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Predict next-token logits over text interleaved with images and videos.

        Args:
            input_ids: Long Tensor of shape [batch, sequence]. Exclusive with ``inputs_embeds``.
            attention_mask: Optional padding Tensor of shape [batch, key_sequence], where
                ``key_sequence`` also counts cached tokens.
            position_ids: Optional long Tensor of shape [batch, sequence].
            past_key_values: Optional generation cache, updated in place.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            pixel_values: Image patches, Tensor of shape
                [patches, in_channels * temporal_patch_size * patch_size**2].
            image_grid_thw: Long Tensor of shape [images, 3], each patch grid as (frames, height, width).
            pixel_values_videos: Video patches, laid out like ``pixel_values``.
            video_grid_thw: Long Tensor of shape [videos, 3], laid out like ``image_grid_thw``.
            labels: Optional long Tensor of shape [batch, sequence]; ``-100`` is ignored.
                Shifted internally.
            use_cache: Whether to create and return a generation cache.
            logits_to_keep: Project only the last ``logits_to_keep`` positions (0 keeps
                all), or the positions in a long Tensor of shape [kept].
            mm_encoder_outputs: Precomputed vision features, as generation passes them.
            **kwargs: Hugging Face output-capture, attention and loss arguments.

        Returns:
            Output whose ``logits`` is a Tensor of shape [batch, kept, vocab] and whose
            ``loss`` is a scalar Tensor when labels are given.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )
        logits = compute_lm_head_logits(self.lm_head, outputs.last_hidden_state, logits_to_keep).logits
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


ModelClass = Lfm3VlForConditionalGeneration
