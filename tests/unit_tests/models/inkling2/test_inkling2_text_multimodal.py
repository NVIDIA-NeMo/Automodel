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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration

from .parity_check_inkling2 import (
    IMAGE_TOKEN_ID,
    TINY_TEXT_CONFIG,
    TINY_VISION_CONFIG,
    _randomize,
    multimodal_inputs,
)

pytestmark = pytest.mark.cpu_only

_BACKEND = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")


def _config() -> Inkling2Config:
    return Inkling2Config(
        text_config=TINY_TEXT_CONFIG,
        vision_config=TINY_VISION_CONFIG,
        image_token_id=IMAGE_TOKEN_ID,
        torch_dtype="float32",
    )


def _model() -> Inkling2ForConditionalGeneration:
    model = Inkling2ForConditionalGeneration(_config(), backend=_BACKEND)
    _randomize(model, seed=1)
    return model


def test_image_text_forward_and_backward_reach_both_towers() -> None:
    model = _model()
    input_ids, attention_mask, pixel_values, image_grid_thw = multimodal_inputs()
    output = model(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    )

    assert output.logits.shape == (2, 24, TINY_TEXT_CONFIG["unpadded_vocab_size"])
    assert output.image_hidden_states.shape == (22, TINY_TEXT_CONFIG["hidden_size"])
    valid = attention_mask.bool()
    (output.logits * torch.randn_like(output.logits))[valid].sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.model.vision_tower.embeddings.patch_embedding.weight.grad.abs().sum() > 0


def test_image_features_replace_only_placeholder_embeddings() -> None:
    model = _model()
    input_ids, attention_mask, pixel_values, image_grid_thw = multimodal_inputs()
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "image_grid_thw": image_grid_thw}
    with torch.no_grad():
        base = model(pixel_values=pixel_values, **inputs).logits
        other = model(pixel_values=-pixel_values, **inputs).logits

    # Row 0 has its first placeholder at position 4; causality keeps earlier logits unchanged.
    torch.testing.assert_close(base[0, :4], other[0, :4], rtol=0, atol=1e-5)
    assert (base[0, 4:] - other[0, 4:]).abs().max() > 1e-3
    # Row 1's placeholders start at position 6.
    torch.testing.assert_close(base[1, 3:6], other[1, 3:6], rtol=0, atol=1e-5)


def test_inputs_embeds_locate_placeholders_by_their_embedding() -> None:
    model = _model()
    input_ids, attention_mask, pixel_values, image_grid_thw = multimodal_inputs()
    media = {"attention_mask": attention_mask, "pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
    with torch.no_grad():
        from_ids = model(input_ids=input_ids, **media).logits
        from_embeds = model(inputs_embeds=model.get_input_embeddings()(input_ids), **media).logits
    torch.testing.assert_close(from_embeds, from_ids, rtol=0, atol=0)


def test_placeholder_count_must_match_image_features() -> None:
    model = _model()
    input_ids, attention_mask, pixel_values, image_grid_thw = multimodal_inputs()
    input_ids[0, 4] = 3
    with pytest.raises(ValueError, match="do not match"):
        model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    with pytest.raises(ValueError, match="image_grid_thw"):
        model(input_ids=input_ids, pixel_values=pixel_values)


def test_checkpoint_free_initialization_covers_the_vision_tower() -> None:
    with torch.device("meta"):
        model = Inkling2ForConditionalGeneration(_config(), backend=_BACKEND)
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    model.initialize_weights(torch.device("cpu"))

    for name, tensor in model.state_dict().items():
        assert torch.isfinite(tensor).all(), name
    layer = model.model.vision_tower.layers[0]
    assert torch.all(layer.lambda_1 == layer.layer_scale_init_value)
    input_ids, attention_mask, pixel_values, image_grid_thw = multimodal_inputs()
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    ).logits
    logits.sum().backward()
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
