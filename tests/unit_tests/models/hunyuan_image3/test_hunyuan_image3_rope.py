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

"""Unit tests for the HunyuanImage-3.0 2D rotary position embedding."""

import pytest
import torch

from nemo_automodel.components.models.hunyuan_image3.rope import (
    apply_rope,
    build_2d_positions,
    build_2d_rope_cos_sin,
    image_block_positions,
    rotate_half,
)


class TestPositions:
    def test_text_only_is_diagonal(self):
        pos = build_2d_positions(5, [])
        torch.testing.assert_close(pos, torch.arange(5)[:, None].expand(5, 2))
        assert pos.dtype == torch.long

    def test_image_block_is_centred_on_its_span(self):
        # A 2x4 block starting at 3 spans indices 3..10: rows start at 3 + (8 - 2) / 2 = 6, cols at 3 + (8 - 4) / 2 = 5.
        rows, cols = image_block_positions(3, 2, 4)
        torch.testing.assert_close(rows, torch.tensor([6.0, 6, 6, 6, 7, 7, 7, 7]))
        torch.testing.assert_close(cols, torch.tensor([5.0, 6, 7, 8, 5, 6, 7, 8]))

    def test_fractional_offsets_are_truncated(self):
        # 3x2 block at 0: rows start at (6 - 3) / 2 = 1.5 -> 1, 2, 3; cols at (6 - 2) / 2 = 2 -> 2, 3.
        pos = build_2d_positions(6, [(0, 3, 2)])
        torch.testing.assert_close(pos[:, 0], torch.tensor([1, 1, 2, 2, 3, 3]))
        torch.testing.assert_close(pos[:, 1], torch.tensor([2, 3, 2, 3, 2, 3]))

    def test_text_resumes_after_image_block(self):
        pos = build_2d_positions(14, [(3, 2, 4)])
        torch.testing.assert_close(pos[:3], torch.arange(3)[:, None].expand(3, 2))
        torch.testing.assert_close(pos[11:], torch.arange(11, 14)[:, None].expand(3, 2))

    def test_multiple_blocks(self):
        pos = build_2d_positions(12, [(1, 2, 2), (6, 1, 3)])
        torch.testing.assert_close(pos[5], torch.tensor([5, 5]))
        # Second block: rows start at 6 + (3 - 1) / 2 = 7, cols at 6 + 0 = 6.
        torch.testing.assert_close(pos[6:9, 0], torch.tensor([7, 7, 7]))
        torch.testing.assert_close(pos[6:9, 1], torch.tensor([6, 7, 8]))


class TestCosSin:
    def test_frequencies_alternate_between_axes(self):
        head_dim, theta = 8, 10000.0
        pos = torch.tensor([[3, 7]])
        cos, sin = build_2d_rope_cos_sin(pos, head_dim, theta)
        assert cos.shape == (1, head_dim) and cos.dtype == torch.float32
        expected = []
        for i in range(head_dim // 2):
            inv_freq = theta ** (-2 * i / head_dim)
            expected.append(pos[0, i % 2].item() * inv_freq)
        angles = torch.tensor(expected * 2)
        torch.testing.assert_close(cos[0], angles.cos())
        torch.testing.assert_close(sin[0], angles.sin())

    def test_rejects_head_dim_not_divisible_by_four(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            build_2d_rope_cos_sin(torch.zeros(1, 2), 6, 10000.0)

    def test_batched_positions(self):
        pos = build_2d_positions(10, [(2, 2, 2)])
        cos, _ = build_2d_rope_cos_sin(torch.stack([pos, pos]), 16, 10000.0)
        assert cos.shape == (2, 10, 16)


class TestApplyRope:
    def test_rotate_half(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    def test_position_zero_is_identity_and_norm_is_preserved(self):
        x = torch.randn(1, 2, 6, 16)
        pos = build_2d_positions(6, [(2, 2, 2)])[None]
        cos, sin = build_2d_rope_cos_sin(pos, 16, 10000.0)
        out = apply_rope(x, cos, sin)
        torch.testing.assert_close(out[:, :, 0], x[:, :, 0])
        torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))
