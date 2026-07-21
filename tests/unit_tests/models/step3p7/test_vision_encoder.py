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

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.models.step3p7.configuration_step3p7 import StepRoboticsVisionEncoderConfig
from nemo_automodel.components.models.step3p7.vision_encoder import (
    EncoderLayerScale,
    EncoderMLP,
    EncoderRope2D,
    EncoderVisionAttention,
    EncoderVisionBlock,
    EncoderVisionTransformer,
    StepRoboticsVisionEncoder,
    apply_rotary_emb,
    rotate_half,
)


def test_rotate_half_and_apply_rotary_emb_preserve_non_rotary_slices():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))

    t = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
    freqs = torch.zeros(1, 1, 2)
    out = apply_rotary_emb(freqs, t, start_index=2)
    torch.testing.assert_close(out, t)


def test_apply_rotary_emb_uses_tail_freqs_for_3d_input():
    t = torch.ones(1, 2, 4)
    freqs = torch.zeros(4, 4)
    out = apply_rotary_emb(freqs, t)
    assert out.shape == t.shape


def test_encoder_rope_2d_cached_dynamic_and_cls_paths():
    rope = EncoderRope2D(dim=4, max_grid_height=2, max_grid_width=2, use_cls_token=True)
    q = torch.randn(1, 1, 5, 4)
    k = torch.randn(1, 1, 5, 4)
    q_cached, k_cached = rope(q, k, grid_hw=(2, 2))
    assert q_cached.shape == q.shape
    assert k_cached.shape == k.shape

    q_dyn = torch.randn(1, 1, 3, 4)
    k_dyn = torch.randn(1, 1, 3, 4)
    q_dynamic, k_dynamic = rope(q_dyn, k_dyn, grid_hw=(1, 2))
    assert q_dynamic.shape == q_dyn.shape
    assert k_dynamic.shape == k_dyn.shape


def test_layer_scale_mlp_attention_block_and_transformer_forward():
    hidden = torch.randn(2, 4, 8)
    scale = EncoderLayerScale(8, 0.5)
    torch.testing.assert_close(scale(hidden), hidden * scale.gamma)

    mlp = EncoderMLP(hidden_size=8, intermediate_size=16, hidden_act="gelu")
    assert mlp(hidden).shape == hidden.shape

    attention = EncoderVisionAttention(
        hidden_size=8,
        num_heads=2,
        max_grid_height=2,
        max_grid_width=2,
        use_rope2d=False,
    )
    assert attention(hidden, grid_hw=(2, 2)).shape == hidden.shape

    rope_attention = EncoderVisionAttention(
        hidden_size=8,
        num_heads=2,
        max_grid_height=2,
        max_grid_width=2,
        use_rope2d=True,
    )
    assert rope_attention(hidden, grid_hw=(2, 2)).shape == hidden.shape

    with pytest.raises(ValueError, match="must be divisible"):
        EncoderVisionAttention(hidden_size=7, num_heads=2, max_grid_height=2, max_grid_width=2)

    block = EncoderVisionBlock(
        hidden_size=8,
        num_heads=2,
        mlp_ratio=2.0,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        ls_init_value=0.1,
        max_grid_height=2,
        max_grid_width=2,
        use_rope2d=False,
    )
    assert block(hidden, grid_hw=(2, 2)).shape == hidden.shape

    transformer = EncoderVisionTransformer(
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        ls_init_value=0.1,
        max_grid_height=2,
        max_grid_width=2,
        use_rope2d=False,
    )
    assert transformer(hidden, grid_hw=(2, 2)).shape == hidden.shape


def _small_vision_config(**kwargs):
    values = dict(
        width=8,
        layers=1,
        heads=2,
        num_channels=3,
        image_size=8,
        patch_size=2,
        mlp_ratio=2.0,
        hidden_act="gelu",
        use_ln_pre=True,
        use_ln_post=True,
        use_abs_posemb=True,
        use_rope2d=False,
        ls_init_value=0.1,
    )
    values.update(kwargs)
    return StepRoboticsVisionEncoderConfig(**values)


def test_step_robotics_vision_encoder_forward_with_cls_and_abs_posemb_interpolation():
    encoder = StepRoboticsVisionEncoder(_small_vision_config(use_cls_token=True, image_size=8))

    same_grid = encoder.sample_abs_posemb(4, 4)
    assert same_grid.shape == (1, 17, 8)

    interpolated = encoder.sample_abs_posemb(2, 2)
    assert interpolated.shape == (1, 5, 8)

    pixels = torch.randn(2, 3, 8, 8)
    out = encoder(pixels)
    assert out.shape == (2, 16, 8)


def test_step_robotics_vision_encoder_forward_without_optional_features():
    encoder = StepRoboticsVisionEncoder(
        _small_vision_config(
            layers=0,
            use_cls_token=False,
            use_abs_posemb=False,
            use_ln_pre=False,
            use_ln_post=False,
        )
    )
    pixels = torch.randn(1, 3, 8, 8)
    out = encoder(pixels)
    assert out.shape == (1, 16, 8)
