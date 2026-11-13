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

"""Unit tests for the native Inkling 2 vision tower."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest
import torch

from nemo_automodel.components.models.inkling2.vision import Inkling2VisionModel

from .parity_check_inkling2_vision import (
    GOLDEN_GRAD_PARAMETERS,
    IMAGE_GRID_THW,
    TINY_VISION_CONFIG,
    fill_parameters_,
    make_inputs,
)

pytestmark = pytest.mark.cpu_only

# Statistics of Transformers' Inkling2VisionModel (huggingface/transformers@109a7d19bf, sdpa, float32)
# for TINY_VISION_CONFIG, fill_parameters_ and make_inputs; regenerate with
# `python tests/unit_tests/models/inkling2/parity_check_inkling2_vision.py --golden`.
REFERENCE_POOLER_ROW_SUMS = [
    3.6563730239868164,
    1.718820333480835,
    -2.664095878601074,
    -7.893999099731445,
    -5.503066062927246,
    7.44840145111084,
    -0.1236562728881836,
    0.30241847038269043,
    -7.9813337326049805,
    -6.184201240539551,
    -3.9660961627960205,
    2.200927257537842,
    -5.655529499053955,
    0.32954585552215576,
    5.054133892059326,
    2.540538787841797,
    15.964532852172852,
    8.520109176635742,
    2.5494513511657715,
    -7.85372257232666,
    6.964329719543457,
    -0.7613542079925537,
]
REFERENCE_POOLER_FIRST_ROW = [
    0.16729603707790375,
    0.3097411096096039,
    0.3915030360221863,
    -0.25939705967903137,
    -0.08497536182403564,
    -1.4899015426635742,
    -1.1066043376922607,
    0.08870469778776169,
]
REFERENCE_GRAD_NORMS = [
    357.3167724609375,
    68.18910217285156,
    0.9797735810279846,
    4.182027816772461,
    21.37309455871582,
    365.1337890625,
]


@dataclass
class _VisionConfig:
    """Stand-in for ``Inkling2VisionConfig`` with the Transformers attribute names and defaults."""

    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 20
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    mlp_ratio: float = 4.0
    layer_scale_init_value: float = 0.1
    rope_parameters: dict = field(default_factory=lambda: {"rope_type": "axial", "rope_theta": 10000.0})
    spatial_merge_size: int = 2
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    text_hidden_size: int = 4608
    initializer_range: float = 0.02
    intermediate_size: int | None = None

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * self.mlp_ratio)


def _tiny_config() -> _VisionConfig:
    return _VisionConfig(**TINY_VISION_CONFIG)


def _tiny_model() -> tuple[_VisionConfig, Inkling2VisionModel]:
    config = _tiny_config()
    model = Inkling2VisionModel(config)
    fill_parameters_(model)
    return config, model


def test_pooler_output_matches_reference_values():
    config, model = _tiny_model()
    pixel_values, grid, upstream = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )

    output = model(pixel_values, grid)
    (output.pooler_output * upstream).sum().backward()

    parameters = dict(model.named_parameters())
    grad_norms = torch.stack([parameters[name].grad.norm() for name in GOLDEN_GRAD_PARAMETERS])
    torch.testing.assert_close(output.pooler_output.sum(-1), torch.tensor(REFERENCE_POOLER_ROW_SUMS))
    torch.testing.assert_close(output.pooler_output[0, :8], torch.tensor(REFERENCE_POOLER_FIRST_ROW))
    torch.testing.assert_close(grad_norms, torch.tensor(REFERENCE_GRAD_NORMS))


def test_output_shapes_follow_image_grid():
    config, model = _tiny_model()
    pixel_values, grid, _ = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )

    output = model(pixel_values, grid)

    lm_tokens = sum(t * h * w for t, h, w in IMAGE_GRID_THW)
    patches = lm_tokens * config.spatial_merge_size**2
    assert output.pooler_output.shape == (lm_tokens, config.text_hidden_size)
    assert output.last_hidden_state.shape == (patches + len(IMAGE_GRID_THW), config.hidden_size)


def test_images_in_one_batch_do_not_attend_to_each_other():
    config, model = _tiny_model()
    pixel_values, grid, _ = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )
    first_tokens = int(grid[0].prod())

    batched = model(pixel_values, grid).pooler_output
    first = model(pixel_values[:first_tokens], grid[:1]).pooler_output
    second = model(pixel_values[first_tokens:], grid[1:]).pooler_output

    torch.testing.assert_close(batched, torch.cat([first, second]), rtol=1e-5, atol=1e-6)


def test_state_dict_uses_reference_names_and_checkpoint_shapes():
    with torch.device("meta"):
        model = Inkling2VisionModel(_VisionConfig())

    shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}

    # Shapes of the released checkpoint's vision tensors (shard 24 of 24) after the reference renames.
    assert shapes["embeddings.patch_embedding.weight"] == (1024, 1200)
    assert shapes["embeddings.position_embedding.weight"] == (1024, 1024)
    assert shapes["embeddings.class_embedding"] == (1024,)
    assert shapes["embeddings.cls_position_embedding"] == (1024,)
    assert shapes["layers.22.mlp.fc1.weight"] == (4096, 1024)
    assert shapes["layers.22.mlp.fc2.weight"] == (1024, 4096)
    assert shapes["layers.0.lambda_1"] == (1024,)
    assert shapes["merger.linear_1.weight"] == (4096, 4096)
    assert shapes["merger.linear_2.weight"] == (4608, 4096)
    layer_keys = {name.split(".", 2)[2] for name in shapes if name.startswith("layers.0.")}
    assert layer_keys == {
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
        "layernorm_before.weight",
        "layernorm_before.bias",
        "layernorm_after.weight",
        "layernorm_after.bias",
        "lambda_1",
        "lambda_2",
    }
    other_keys = {name for name in shapes if not name.startswith("layers.")}
    assert other_keys == {
        "embeddings.patch_embedding.weight",
        "embeddings.position_embedding.weight",
        "embeddings.class_embedding",
        "embeddings.cls_position_embedding",
        "pre_layernorm.weight",
        "pre_layernorm.bias",
        "merger.linear_1.weight",
        "merger.linear_2.weight",
    }
    assert sum(name.endswith(".lambda_1") for name in shapes) == 23


def test_checkpoint_free_init_gives_finite_forward_and_backward():
    config = _tiny_config()
    with torch.device("meta"):
        model = Inkling2VisionModel(config)
    model.to_empty(device="cpu")
    model.init_weights()

    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
    torch.testing.assert_close(model.layers[1].lambda_2, torch.full((config.hidden_size,), 0.1))
    torch.testing.assert_close(model.pre_layernorm.weight, torch.ones(config.hidden_size))
    assert model.embeddings.class_embedding.abs().sum() > 0

    pixel_values, grid, upstream = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )
    output = model(pixel_values, grid).pooler_output
    (output * upstream).sum().backward()
    assert torch.isfinite(output).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_bfloat16_tower_stays_close_to_float32():
    config, model = _tiny_model()
    pixel_values, grid, _ = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )
    expected = model(pixel_values, grid).pooler_output

    actual = model.to(torch.bfloat16)(pixel_values, grid).pooler_output

    assert actual.dtype == torch.bfloat16
    relative = (actual.float() - expected).norm() / expected.norm()
    assert relative < 2e-2


def test_rejects_pixel_values_that_do_not_match_the_grid():
    config, model = _tiny_model()
    pixel_values, grid, _ = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )

    with pytest.raises(ValueError, match="do not match image_grid_thw"):
        model(pixel_values[:-1], grid)
    with pytest.raises(ValueError, match="do not match image_grid_thw"):
        model(pixel_values[:, :-1], grid)


def test_rejects_non_axial_rope():
    config = replace(_tiny_config(), rope_parameters={"rope_type": "default", "rope_theta": 10000.0})

    with pytest.raises(ValueError, match="axial"):
        Inkling2VisionModel(config)
