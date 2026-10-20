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

"""Compatibility fixes for Hugging Face Nemotron Omni remote-code models."""

from functools import wraps
from types import MethodType
from typing import Protocol, cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.nemotron_omni.model import NemotronOmniForConditionalGeneration


class _HFNemotronOmni(Protocol):
    """Remote-code attributes used by the variable-resolution compatibility path."""

    img_context_token_id: int
    vision_model: nn.Module
    vision_projector: nn.Module

    def get_input_embeddings(self) -> nn.Module:
        """Return the language model's token embedding module."""


def _merge_variable_resolution_images(
    model: _HFNemotronOmni,
    input_ids: torch.Tensor,
    pixel_values: list[torch.Tensor] | tuple[torch.Tensor, ...],
    image_flags: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> torch.Tensor:
    """Encode and scatter a ragged image batch into the text embeddings.

    Args:
        model: Hugging Face Nemotron Omni remote-code model.
        input_ids: Tensor of shape [batch, sequence].
        pixel_values: Per-image tensors of shape [channels, height, width] or
            [images, channels, height, width]. Spatial dimensions may differ.
        image_flags: Optional tensor of shape [images, 1], where one marks a
            real image and zero marks padding.
        inputs_embeds: Optional tensor of shape [batch, sequence, hidden].

    Returns:
        Tensor of shape [batch, sequence, hidden] with image-token positions
        replaced by the corresponding variable-resolution vision features.
    """
    if inputs_embeds is None:
        inputs_embeds = model.get_input_embeddings()(input_ids)

    per_image_features: list[torch.Tensor] = []
    for image in pixel_values:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError(
                "Nemotron Omni variable-resolution images must have shape "
                f"[channels, height, width] or [images, channels, height, width], got {tuple(image.shape)}."
            )

        projected = model.vision_projector(image, model.vision_model)
        if projected.ndim != 3:
            raise ValueError(
                "Nemotron Omni vision projector must return [images, image_tokens, hidden], "
                f"got {tuple(projected.shape)}."
            )
        per_image_features.extend(projected.unbind(0))

    if image_flags is not None:
        flat_flags = image_flags.reshape(-1)
        if flat_flags.numel() != len(per_image_features):
            raise ValueError(
                "Nemotron Omni image_flags must contain one value per image, "
                f"got {flat_flags.numel()} flags for {len(per_image_features)} images."
            )
        per_image_features = [features for features, keep in zip(per_image_features, flat_flags.tolist()) if keep == 1]

    if per_image_features:
        image_embeds = torch.cat([features.reshape(-1, features.shape[-1]) for features in per_image_features], dim=0)
    else:
        image_embeds = inputs_embeds.new_empty((0, inputs_embeds.shape[-1]))

    image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    image_positions = input_ids == model.img_context_token_id
    num_image_positions = int(image_positions.sum().item())
    if image_embeds.shape[0] != num_image_positions:
        raise ValueError(
            "Nemotron Omni variable-resolution feature count must match the image-token count, "
            f"got {image_embeds.shape[0]} features for {num_image_positions} positions."
        )

    image_mask = image_positions.unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(image_mask.to(inputs_embeds.device), image_embeds)


def apply_hf_variable_resolution_patch(model: nn.Module, *, mesh: DeviceMesh | None) -> nn.Module:
    """Patch the HF remote-code model to support ragged image batches.

    The checkpoint's projector recursively encodes list elements and then
    stacks them along the image axis. That fails first for three-dimensional
    list elements and, after adding a batch axis, for images with different
    token counts. AutoModel's native implementation already keeps the encoded
    images ragged and flattens them only for the language-model scatter, so it
    is left unchanged.

    Args:
        model: Constructed Nemotron Omni model instance.
        mesh: AutoModel device mesh. The compatibility patch is mesh-agnostic.

    Returns:
        The input model with its instance-local HF forward compatibility patch.
    """
    del mesh
    if isinstance(model, NemotronOmniForConditionalGeneration):
        return model

    hf_model = cast(_HFNemotronOmni, model)
    original_forward = model.forward

    @wraps(original_forward)
    def _forward(
        self: _HFNemotronOmni,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        sound_clips: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
        past_key_values: list[torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        """Run HF Nemotron Omni with a ragged-image compatibility path.

        Args:
            input_ids: Optional tensor of shape [batch, sequence].
            pixel_values: Optional tensor of shape [images, channels, height,
                width], or a list of per-image tensors with variable height and width.
            pixel_values_videos: Optional tensor of shape [frames, channels, height, width].
            sound_clips: Optional tensor following the checkpoint sound projector contract.
            attention_mask: Optional tensor of shape [batch, sequence].
            position_ids: Optional tensor of shape [batch, sequence].
            image_flags: Optional tensor of shape [images, 1].
            past_key_values: Optional per-layer cache tensors following the checkpoint contract.
            labels: Optional tensor of shape [batch, sequence].
            inputs_embeds: Optional tensor of shape [batch, sequence, hidden].
            use_cache: Whether to return key/value caches.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden-state tensors.
            return_dict: Whether to return a model-output object.
            **kwargs: Additional arguments accepted by the checkpoint forward method.

        Returns:
            A causal language-model output whose logits have shape [batch, sequence, vocab].
        """
        if isinstance(pixel_values, (list, tuple)) and input_ids is not None:
            inputs_embeds = _merge_variable_resolution_images(
                self,
                input_ids,
                pixel_values,
                image_flags,
                inputs_embeds,
            )
            pixel_values = None
            image_flags = None

        return original_forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            sound_clips=sound_clips,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    model.forward = MethodType(_forward, hf_model)
    return model
