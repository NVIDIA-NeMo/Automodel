# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# The image placeholder scatter follows the HuggingFace Inkling 2 implementation (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Source: https://github.com/huggingface/transformers/pull/50682
# Revision: 109a7d19bf667168837f32929e00b791e195d14d
# Original file: src/transformers/models/inkling2/modeling_inkling2.py

"""Inkling 2 backbone composing the text model and the vision tower.

The image placeholder scatter follows the Transformers ``Inkling2Model`` reference
(huggingface/transformers commit 109a7d19bf667168837f32929e00b791e195d14d, Apache-2.0):
text tokens are embedded and normalized, then each image placeholder embedding is replaced
by one projected image feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig

from .configuration import Inkling2Config
from .text import Inkling2TextModel
from .vision import Inkling2VisionModel


@dataclass
class Inkling2ModelOutput(BaseModelOutput):
    """Backbone output.

    Attributes:
        last_hidden_state: Tensor of shape ``[batch, sequence, hidden]``.
        image_hidden_states: Optional tensor of shape ``[image_tokens, hidden]``, the projected
            image features scattered into the sequence.
    """

    image_hidden_states: torch.Tensor | None = None


class Inkling2Model(nn.Module):
    """Inkling 2 text backbone plus an optional vision tower.

    The vision tower is built only when ``config.vision_config`` is set.
    """

    def __init__(
        self,
        config: Inkling2Config,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = Inkling2TextModel(config.text_config, backend, moe_config, dtype=dtype)
        self.vision_tower = None if config.vision_config is None else Inkling2VisionModel(config.vision_config)

    def get_input_embeddings(self) -> nn.Module:
        """Return the normalized token embedding."""
        return self.language_model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode images into one language-model embedding per image token.

        Args:
            pixel_values: Tensor of shape ``[image_tokens, patch * merge, patch * merge, channels]``,
                one patch group per image token in raster order within each image.
            image_grid_thw: Long tensor of shape ``[images, 3]``, the temporal, height and width
                extent of each image in image tokens.

        Returns:
            Tensor of shape ``[image_tokens, hidden]``.

        Raises:
            ValueError: If the model was built without a vision tower.
        """
        if self.vision_tower is None:
            raise ValueError("pixel_values were given to an Inkling 2 model built without vision_config")
        return self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw).pooler_output

    def _placeholder_mask(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Locate the image placeholders and check that they match the features.

        Args:
            input_ids: Optional long tensor of shape ``[batch, sequence]``.
            inputs_embeds: Tensor of shape ``[batch, sequence, hidden]``. Without ``input_ids``,
                placeholders are positions equal to the normalized placeholder embedding.
            image_features: Tensor of shape ``[image_tokens, hidden]``.

        Returns:
            Boolean tensor of shape ``[batch, sequence, hidden]``.

        Raises:
            ValueError: If the number of placeholders and image features differ.
        """
        if input_ids is None:
            token = torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            special_mask = (inputs_embeds == self.get_input_embeddings()(token)).all(-1)
        else:
            special_mask = input_ids == self.config.image_token_id
        special_mask = special_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[special_mask].numel() != image_features.numel():
            raise ValueError(
                "Image features and image placeholder tokens do not match: "
                f"tokens={int(special_mask[..., 0].sum())}, features={image_features.shape[0]}"
            )
        return special_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> Inkling2ModelOutput:
        """Merge image features into the token embeddings and run the text backbone.

        Args:
            input_ids: Optional long tensor of shape ``[batch, sequence]``.
            pixel_values: Optional tensor of shape ``[image_tokens, patch * merge, patch * merge, channels]``.
            image_grid_thw: Long tensor of shape ``[images, 3]``; required with ``pixel_values``.
            attention_mask: Optional padding tensor of shape ``[batch, sequence]``.
            position_ids: Optional long tensor of shape ``[batch or 1, sequence]``.
            inputs_embeds: Optional normalized embeddings of shape ``[batch, sequence, hidden]``.

        Returns:
            Output with ``last_hidden_state`` of shape ``[batch, sequence, hidden]`` and the
            scattered ``image_hidden_states`` of shape ``[image_tokens, hidden]`` when images are given.

        Raises:
            ValueError: If both or neither of ``input_ids`` and ``inputs_embeds`` are given, or
                ``pixel_values`` come without ``image_grid_thw``.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("pixel_values require image_grid_thw")
            image_features = self.get_image_features(pixel_values, image_grid_thw)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self._placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return Inkling2ModelOutput(last_hidden_state=outputs.last_hidden_state, image_hidden_states=image_features)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        """Initialize the text backbone and the vision tower for checkpoint-free construction."""
        self.language_model.init_weights(buffer_device)
        if self.vision_tower is not None:
            self.vision_tower.init_weights()


__all__ = ["Inkling2Model", "Inkling2ModelOutput"]
