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

import inspect
from functools import wraps
from types import MethodType
from typing import Protocol

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.modeling_outputs import CausalLMOutputWithPast


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
    n_images_per_sample: torch.Tensor | None,
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
        n_images_per_sample: Optional tensor of shape [batch] mapping the flat
            image list back to each text row. When omitted, the concatenated
            features are scattered across the whole batch, as in the checkpoint
            forward.
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

    keep_flags = [True] * len(per_image_features)
    if image_flags is not None:
        flat_flags = image_flags.reshape(-1)
        if flat_flags.numel() != len(per_image_features):
            raise ValueError(
                "Nemotron Omni image_flags must contain one value per image, "
                f"got {flat_flags.numel()} flags for {len(per_image_features)} images."
            )
        keep_flags = [flag == 1 for flag in flat_flags.tolist()]

    if inputs_embeds.ndim != 3:
        raise ValueError(
            "Nemotron Omni variable-resolution inputs_embeds must have shape [batch, sequence, hidden], "
            f"got {tuple(inputs_embeds.shape)}."
        )

    image_positions = input_ids == model.img_context_token_id
    if n_images_per_sample is None:
        # Without per-row counts, follow the checkpoint forward and scatter the
        # concatenated features across the whole batch. Counts are only needed
        # to drop a truncated row's unused features before the next row.
        kept_features = [
            features.reshape(-1, features.shape[-1]) for features, keep in zip(per_image_features, keep_flags) if keep
        ]
        return _scatter_image_features(inputs_embeds, image_positions, kept_features, "the batch")

    batch_size = input_ids.shape[0]
    image_counts = [int(count) for count in n_images_per_sample.reshape(-1).tolist()]
    if len(image_counts) != batch_size:
        raise ValueError(
            "Nemotron Omni n_images_per_sample must contain one value per text row, "
            f"got {len(image_counts)} values for batch size {batch_size}."
        )
    if sum(image_counts) != len(per_image_features):
        raise ValueError(
            "Nemotron Omni n_images_per_sample must cover the flat image list, "
            f"got a total of {sum(image_counts)} for {len(per_image_features)} images."
        )

    merged_rows: list[torch.Tensor] = []
    image_offset = 0
    for row_index, image_count in enumerate(image_counts):
        row_end = image_offset + image_count
        row_features = [
            features.reshape(-1, features.shape[-1])
            for features, keep in zip(per_image_features[image_offset:row_end], keep_flags[image_offset:row_end])
            if keep
        ]
        image_offset = row_end
        merged_rows.append(
            _scatter_image_features(
                inputs_embeds[row_index], image_positions[row_index], row_features, f"row {row_index}"
            )
        )

    return torch.stack(merged_rows, dim=0)


def _scatter_image_features(
    inputs_embeds: torch.Tensor,
    image_positions: torch.Tensor,
    image_features: list[torch.Tensor],
    scope: str,
) -> torch.Tensor:
    """Scatter concatenated image features into the image-token positions.

    Args:
        inputs_embeds: Tensor of shape [..., sequence, hidden].
        image_positions: Boolean tensor of shape [..., sequence] marking image tokens.
        image_features: Per-image tensors of shape [image_tokens, hidden].
        scope: Description of the scattered region used in error messages.

    Returns:
        Tensor with the same shape as ``inputs_embeds`` whose image-token
        positions hold the corresponding image features.
    """
    if image_features:
        image_embeds = torch.cat(image_features, dim=0)
    else:
        image_embeds = inputs_embeds.new_empty((0, inputs_embeds.shape[-1]))
    image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    num_positions = int(image_positions.sum().item())
    if image_embeds.shape[0] < num_positions:
        raise ValueError(
            "Nemotron Omni image features must cover the retained image-token positions, "
            f"got {image_embeds.shape[0]} features for {num_positions} positions in {scope}."
        )

    # Right truncation can retain only a prefix of the image tokens; drop the unused suffix.
    image_embeds = image_embeds[:num_positions]
    mask = image_positions.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    return inputs_embeds.masked_scatter(mask, image_embeds)


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
    if not type(model).__module__.startswith("transformers_modules."):
        return model

    original_forward = model.forward
    forward_signature = inspect.signature(original_forward)

    @wraps(original_forward)
    def _forward(
        self: _HFNemotronOmni,
        *args: object,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        """Run remote-code Nemotron Omni with a ragged-image compatibility path.

        Args:
            *args: Positional arguments accepted by the checkpoint forward method.
            **kwargs: Additional arguments accepted by the checkpoint forward method.

        Returns:
            A causal language-model output whose logits have shape [batch, sequence, vocab].
        """
        bound = forward_signature.bind_partial(*args, **kwargs)
        forwarded_kwargs = bound.arguments.get("kwargs")
        n_images_per_sample = None
        if isinstance(forwarded_kwargs, dict):
            n_images_per_sample = forwarded_kwargs.pop("n_images_per_sample", None)

        input_ids = bound.arguments.get("input_ids")
        pixel_values = bound.arguments.get("pixel_values")
        image_flags = bound.arguments.get("image_flags")
        inputs_embeds = bound.arguments.get("inputs_embeds")
        if isinstance(pixel_values, (list, tuple)) and isinstance(input_ids, torch.Tensor):
            if image_flags is not None and not isinstance(image_flags, torch.Tensor):
                raise TypeError("Nemotron Omni image_flags must be a tensor or None.")
            if n_images_per_sample is not None and not isinstance(n_images_per_sample, torch.Tensor):
                raise TypeError("Nemotron Omni n_images_per_sample must be a tensor or None.")
            if inputs_embeds is not None and not isinstance(inputs_embeds, torch.Tensor):
                raise TypeError("Nemotron Omni inputs_embeds must be a tensor or None.")

            bound.arguments["inputs_embeds"] = _merge_variable_resolution_images(
                self,
                input_ids,
                pixel_values,
                image_flags,
                n_images_per_sample,
                inputs_embeds,
            )
            bound.arguments["pixel_values"] = None
            bound.arguments["image_flags"] = None

        return original_forward(*bound.args, **bound.kwargs)

    model.forward = MethodType(_forward, model)
    return model
