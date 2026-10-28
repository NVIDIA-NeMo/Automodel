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

"""Actual PP wrapper dispatch uses staged media ownership for every microbatch."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.mistral4.model import Mistral3ForConditionalGeneration
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
from nemo_automodel.components.models.qwen3_vl_moe.model import Qwen3VLMoeForConditionalGeneration
from nemo_automodel.components.models.step3p7.model import Step3p7ForConditionalGeneration


class _MediaBackbone(torch.nn.Module):
    def __init__(self, *, tensor_output=False):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.embed_tokens = torch.nn.Embedding(32, 3)
        self.language_model.lm_head = torch.nn.Linear(3, 7, bias=False)
        self.tensor_output = tensor_output
        self.calls = []

    def forward(self, input_ids, **kwargs):
        """Probe token IDs and processor-shaped media.

        Args:
            input_ids: Token IDs [batch, sequence].
            **kwargs: Optional image/video/patch pixels with arbitrary trailing
                axes and the media/patch axis first; scalar means affect hidden states.

        Returns:
            Hidden values [batch, sequence, hidden] affected by visible pixels.
        """
        self.calls.append(kwargs)
        hidden = self.language_model.embed_tokens(input_ids)
        for key in ("pixel_values", "pixel_values_videos", "patch_pixel_values"):
            if kwargs.get(key) is not None:
                hidden = hidden + kwargs[key].mean()
        return hidden if self.tensor_output else SimpleNamespace(last_hidden_state=hidden)


@pytest.mark.parametrize(
    "model_class",
    [
        Mistral3ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Step3p7ForConditionalGeneration,
    ],
)
def test_text_then_image_microbatch_matches_explicit_media_and_gradients(model_class):
    torch.manual_seed(9)
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(text_config=SimpleNamespace(output_hidden_states=False))
    model.model = _MediaBackbone(tensor_output=model_class is Step3p7ForConditionalGeneration)
    if model_class is not Mistral3ForConditionalGeneration:
        model.lm_head = model.model.language_model.lm_head
    model.cp_mesh = None
    model.mtp = None
    model.mtp_config = SimpleNamespace(enabled=False)
    model.backend = SimpleNamespace(attn="sdpa")
    model.eval()
    ids = torch.tensor([[1, 2, 3]])  # No hardcoded architecture image-token IDs.
    pixels = torch.randn(2, 3, 4, 4)
    grid = torch.tensor([[2, 3], [3, 2]])
    direct_kwargs = {"pixel_values": pixels}
    if model_class is Mistral3ForConditionalGeneration:
        direct_kwargs["image_sizes"] = grid
    elif model_class in (Qwen3_5MoeForConditionalGeneration, Qwen3VLMoeForConditionalGeneration):
        direct_kwargs["image_grid_thw"] = torch.nn.functional.pad(grid, (1, 0), value=1)
    else:
        direct_kwargs["patch_pixel_values"] = pixels[:1]
        direct_kwargs["num_patches"] = torch.tensor([1, 0])
        direct_kwargs["patch_newline_mask"] = torch.tensor([False])

    direct_text = model(ids)
    direct_image = model(ids, **direct_kwargs)
    direct_logits = direct_image.logits if hasattr(direct_image, "logits") else direct_image
    direct_logits.square().mean().backward()
    expected_gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    model._vlm_pixel_values_chunks = [pixels[:0], pixels]
    model._vlm_image_grid_hws_chunks = [grid[:0], grid]
    model._vlm_chunk_idx = 0
    if model_class is Step3p7ForConditionalGeneration:
        for key in ("patch_pixel_values", "num_patches", "patch_newline_mask"):
            value = direct_kwargs[key]
            setattr(model, f"_vlm_{key}_chunks", [value[:0], value])

    staged_text = model(ids)
    assert model._vlm_chunk_idx == 1
    assert model.model.calls[-1].get("pixel_values") is None
    staged_image = model(ids)
    assert model._vlm_chunk_idx == 2
    actual_kwargs = model.model.calls[-1]
    for key, expected in direct_kwargs.items():
        torch.testing.assert_close(actual_kwargs[key], expected)
    staged_logits = staged_image.logits if hasattr(staged_image, "logits") else staged_image
    torch.testing.assert_close(staged_logits, direct_logits)
    torch.testing.assert_close(
        staged_text.logits if hasattr(staged_text, "logits") else staged_text,
        direct_text.logits if hasattr(direct_text, "logits") else direct_text,
    )
    staged_logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_gradients[name])
    with pytest.raises(ValueError, match="chunks exhausted at microbatch 2"):
        model(ids)
    assert model._vlm_chunk_idx == 2
