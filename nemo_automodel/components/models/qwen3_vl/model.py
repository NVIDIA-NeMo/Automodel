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

"""Dense Qwen3-VL integration with context-parallel multimodal pre-embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
)

from nemo_automodel.components.distributed.cp_vision_shard import maybe_distribute_visual
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)

from .cp_batch import make_qwen3_vl_cp_batch


class _Qwen3VLCpInputs(TypedDict, total=False):
    """Internal batch fields produced by Qwen3-VL's CP pre-embed."""

    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    visual_pos_masks: torch.Tensor
    _deepstack_visual_embeds: list[torch.Tensor]
    _cp_make_batch_fn: Any
    _qwen3_vl_cp_preembedded: bool


class Qwen3VLForConditionalGeneration(HFCheckpointingMixin, HFQwen3VLForConditionalGeneration):
    """Dense Qwen3-VL with a DeepStack-aware context-parallel pre-embed path."""

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for dense Qwen3-VL."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = False
        supports_thd: bool = False

    def __init__(self, config: Qwen3VLConfig) -> None:
        """Construct the Hugging Face-compatible dense Qwen3-VL model.

        Args:
            config: Qwen3-VL configuration containing the text and vision sub-configs.
        """
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        """Tie the language head to the active text embedding when configured."""
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def _prepare_visual_inputs_for_cp(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Encode and scatter Qwen3-VL visual features before sequence sharding.

        Images and videos share one vision-tower call when both are present. This
        preserves collective ordering under FSDP and lets CP vision sharding
        partition their combined frame stream once.

        Args:
            input_ids: Token ids of shape ``[batch, sequence]`` containing image
                and video placeholder ids.
            inputs_embeds: Text embeddings of shape ``[batch, sequence, hidden]``.
            pixel_values: Optional image patch rows of shape
                ``[image_patch_rows, patch_dim]`` in image-entry order.
            pixel_values_videos: Optional video patch rows of shape
                ``[video_patch_rows, patch_dim]`` in video-entry/frame order.
            image_grid_thw: Optional image grid tensor of shape ``[num_images, 3]``.
            video_grid_thw: Optional video grid tensor of shape ``[num_videos, 3]``.

        Returns:
            A tuple containing updated embeddings of shape ``[batch, sequence,
            hidden]``, an optional boolean visual-position mask of shape ``[batch,
            sequence]``, and optional DeepStack tensors each shaped
            ``[visual_tokens, hidden]`` in sequence visual-token order.

        Raises:
            ValueError: If patch rows are provided without matching grid metadata,
                or grid metadata is provided without patch rows.
        """
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("Qwen3-VL image pixel_values and image_grid_thw must be provided together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("Qwen3-VL video pixel_values_videos and video_grid_thw must be provided together")

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        has_images = pixel_values is not None
        has_videos = pixel_values_videos is not None

        if has_images and has_videos:
            image_pixels = pixel_values.type(self.model.visual.dtype)
            video_pixels = pixel_values_videos.type(self.model.visual.dtype)
            merged_output = maybe_distribute_visual(
                self.model.visual,
                torch.cat((image_pixels, video_pixels), dim=0),
                torch.cat((image_grid_thw, video_grid_thw), dim=0),
            )
            spatial_merge_size_sq = self.model.visual.spatial_merge_size**2
            num_image_tokens = int((image_grid_thw.prod(-1) // spatial_merge_size_sq).sum().item())
            image_embeds = merged_output.pooler_output[:num_image_tokens]
            video_embeds = merged_output.pooler_output[num_image_tokens:]
            merged_deepstack = merged_output.deepstack_features or []
            deepstack_image_embeds = [features[:num_image_tokens] for features in merged_deepstack]
            deepstack_video_embeds = [features[num_image_tokens:] for features in merged_deepstack]
        elif has_images:
            image_output = maybe_distribute_visual(
                self.model.visual,
                pixel_values.type(self.model.visual.dtype),
                image_grid_thw,
            )
            image_embeds = image_output.pooler_output
            deepstack_image_embeds = image_output.deepstack_features or []
        elif has_videos:
            video_output = maybe_distribute_visual(
                self.model.visual,
                pixel_values_videos.type(self.model.visual.dtype),
                video_grid_thw,
            )
            video_embeds = video_output.pooler_output
            deepstack_video_embeds = video_output.deepstack_features or []

        if has_images:
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if has_videos:
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for image_features, video_features in zip(deepstack_image_embeds, deepstack_video_embeds):
                joint_features = image_features.new_zeros(visual_pos_masks.sum(), image_features.shape[-1])
                joint_features[image_mask_joint] = image_features
                joint_features[video_mask_joint] = video_features
                deepstack_visual_embeds.append(joint_features)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds
        else:
            visual_pos_masks = None
            deepstack_visual_embeds = None

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    def prepare_model_inputs_for_cp(
        self,
        input_ids: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> _Qwen3VLCpInputs:
        """Build full-sequence Qwen3-VL multimodal inputs before CP sharding.

        Args:
            input_ids: Token ids of shape ``[batch, sequence]``.
            attention_mask: Optional padding mask of shape ``[batch, sequence]``.
            position_ids: Optional precomputed mRoPE positions of shape
                ``[position_channels, batch, sequence]``.
            pixel_values: Optional image patch rows of shape
                ``[image_patch_rows, patch_dim]``.
            pixel_values_videos: Optional video patch rows of shape
                ``[video_patch_rows, patch_dim]``.
            image_grid_thw: Optional image grids of shape ``[num_images, 3]``.
            video_grid_thw: Optional video grids of shape ``[num_videos, 3]``.
            mm_token_type_ids: Optional modality ids of shape ``[batch, sequence]``.
            **_kwargs: Additional upstream-compatible inputs ignored by this pre-embed path.

        Returns:
            Mapping containing ``inputs_embeds`` of shape ``[batch, sequence,
            hidden]`` and, when available, ``position_ids`` of shape
            ``[position_channels, batch, sequence]``. Media batches also carry
            ``visual_pos_masks`` of shape ``[batch, sequence]`` and one DeepStack
            tensor of shape ``[visual_tokens, hidden]`` per injection layer.

        Raises:
            ValueError: If ``input_ids`` is missing or media/grid pairs are incomplete.
        """
        if input_ids is None:
            raise ValueError("Qwen3-VL CP pre-embedding requires input_ids")

        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._prepare_visual_inputs_for_cp(
            input_ids,
            inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )

        prepared: _Qwen3VLCpInputs = {
            "inputs_embeds": inputs_embeds,
            "_qwen3_vl_cp_preembedded": True,
        }
        if position_ids is not None:
            prepared["position_ids"] = position_ids
        if visual_pos_masks is not None:
            prepared["visual_pos_masks"] = visual_pos_masks
            prepared["_deepstack_visual_embeds"] = deepstack_visual_embeds or []
            prepared["_cp_make_batch_fn"] = make_qwen3_vl_cp_batch
        return prepared

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> tuple | Qwen3VLCausalLMOutputWithPast | _Qwen3VLCpInputs:
        """Run Qwen3-VL, including the internal CP pre-embed and local DeepStack path.

        Named tensor arguments preserve the Hugging Face
        ``Qwen3VLForConditionalGeneration.forward`` layouts. The internal CP path
        consumes ``inputs_embeds`` of shape ``[batch, local_sequence, hidden]``,
        ``visual_pos_masks`` of shape ``[batch, local_sequence]``, and DeepStack
        tensors of shape ``[local_visual_tokens, hidden]``.
        """
        if kwargs.pop("_pre_embed_only", False):
            return self.prepare_model_inputs_for_cp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )

        cp_preembedded = bool(kwargs.pop("_qwen3_vl_cp_preembedded", False))
        if not cp_preembedded:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        deepstack_visual_embeds = kwargs.pop("_deepstack_visual_embeds", None)
        text_outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        hidden_states = text_outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=text_outputs.past_key_values,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
            rope_deltas=self.model.rope_deltas,
        )


ModelClass = Qwen3VLForConditionalGeneration
