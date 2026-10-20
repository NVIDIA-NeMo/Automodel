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

from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches
from nemo_automodel.components.models.nemotron_omni.model import NemotronOmniForConditionalGeneration


class _StubProjector(nn.Module):
    def forward(self, pixel_values: torch.Tensor | list[torch.Tensor], vision_model: nn.Module) -> torch.Tensor:
        """Project dense images while reproducing the remote list behavior.

        Args:
            pixel_values: Tensor of shape [images, channels, height, width], or
                a list whose elements are forwarded recursively.
            vision_model: Unused vision tower stub.

        Returns:
            Tensor of shape [images, image_tokens, hidden].
        """
        del vision_model
        if isinstance(pixel_values, list):
            return torch.cat([self.forward(image, nn.Identity()) for image in pixel_values], dim=0)
        if pixel_values.ndim != 4:
            raise ValueError(f"expected 4 dimensions, got {pixel_values.ndim}")

        batch, _, height, width = pixel_values.shape
        image_tokens = (height // 2) * (width // 2)
        marker = float(height * 10 + width)
        return torch.full((batch, image_tokens, 4), marker, dtype=pixel_values.dtype)


class _StubHFNemotronOmni(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(architectures=["NemotronH_Omni_Reasoning_V3"])
        self.img_context_token_id = 18
        self.embedding = nn.Embedding(32, 4)
        self.vision_model = nn.Identity()
        self.vision_projector = _StubProjector()

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        image_flags: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        """Reproduce the checkpoint's image merge on a tiny CPU model.

        Args:
            input_ids: Tensor of shape [batch, sequence].
            pixel_values: Optional tensor of shape [images, channels, height,
                width], or a list of variable-resolution image tensors.
            image_flags: Optional tensor of shape [images, 1].
            inputs_embeds: Optional tensor of shape [batch, sequence, hidden].
            **kwargs: Unused checkpoint-forward arguments.

        Returns:
            Causal language-model output with logits of shape [batch, sequence, hidden].
        """
        del kwargs
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        if pixel_values is not None:
            image_embeds = self.vision_projector(pixel_values, self.vision_model)
            if image_flags is not None:
                image_embeds = image_embeds[image_flags.squeeze(-1) == 1]
            image_mask = (input_ids == self.img_context_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return CausalLMOutputWithPast(logits=inputs_embeds)


def test_runtime_patch_packs_variable_resolution_hf_features() -> None:
    model = _StubHFNemotronOmni()
    model = apply_model_runtime_patches(model, mesh=None)

    input_ids = torch.tensor(
        [
            [5, 18, 18, 18, 18, 5, 5, 5],
            [18, 18, 18, 18, 18, 18, 18, 18],
        ]
    )
    image_mask = input_ids == model.img_context_token_id
    text_embeddings = model.embedding(input_ids)[~image_mask].detach().clone()
    pixel_values = [torch.zeros(3, 4, 4), torch.zeros(3, 4, 8)]

    output = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_flags=torch.ones(2, 1, dtype=torch.long),
    )

    assert output.logits is not None
    torch.testing.assert_close(output.logits[~image_mask], text_embeddings)
    expected_images = torch.cat([torch.full((4, 4), 44.0), torch.full((8, 4), 48.0)])
    torch.testing.assert_close(output.logits[image_mask], expected_images)


def test_runtime_patch_skips_native_nemotron_omni() -> None:
    model = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(architectures=["NemotronH_Omni_Reasoning_V3"])
    original_forward = model.forward.__func__

    patched = apply_model_runtime_patches(model, mesh=None)

    assert patched is model
    assert model.forward.__func__ is original_forward
