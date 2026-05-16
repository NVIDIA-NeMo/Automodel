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

import math

import pytest
import torch

from nemo_automodel.components.models.mimo_v2_flash.model import (
    MiMoV2FlashRotaryEmbedding,
    MiMoV2RMSNorm,
    _apply_rotary_pos_emb,
    _convert_bool_4d_mask_to_additive,
    _derive_padding_mask,
    _fallback_additive_mask,
    _repeat_kv,
    _rotate_half,
)


class TestRotateHalf:
    def test_swap_halves_with_negation(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = _rotate_half(x)
        # rotate_half splits along last dim into halves: x1=[1,2], x2=[3,4] -> [-3,-4,1,2]
        torch.testing.assert_close(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    def test_double_rotation_is_negation(self):
        x = torch.randn(2, 6)
        torch.testing.assert_close(_rotate_half(_rotate_half(x)), -x)


class TestApplyRotaryPosEmb:
    def test_identity_when_cos_one_sin_zero(self):
        batch, heads, seq, dim = 1, 2, 4, 8
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        cos = torch.ones(batch, seq, dim)
        sin = torch.zeros(batch, seq, dim)
        q_out, k_out = _apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(q_out, q)
        torch.testing.assert_close(k_out, k)

    def test_matches_manual_formula(self):
        batch, heads, seq, dim = 1, 1, 3, 4
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        cos = torch.randn(batch, seq, dim)
        sin = torch.randn(batch, seq, dim)
        q_out, k_out = _apply_rotary_pos_emb(q, k, cos, sin)
        cu = cos.unsqueeze(1)
        su = sin.unsqueeze(1)
        torch.testing.assert_close(q_out, q * cu + _rotate_half(q) * su)
        torch.testing.assert_close(k_out, k * cu + _rotate_half(k) * su)


class TestRepeatKv:
    def test_no_op_when_n_rep_is_one(self):
        x = torch.randn(1, 2, 3, 4)
        out = _repeat_kv(x, 1)
        assert out is x or torch.equal(out, x)

    def test_repeats_along_head_dim(self):
        x = torch.randn(2, 3, 4, 5)
        out = _repeat_kv(x, 4)
        assert out.shape == (2, 12, 4, 5)
        # Adjacent groups of 4 along dim=1 should be identical (broadcasted copies).
        torch.testing.assert_close(out[:, 0], out[:, 1])
        torch.testing.assert_close(out[:, 0], out[:, 3])


class TestConvertBool4dMaskToAdditive:
    def test_passes_through_non_bool_or_non_4d(self):
        x = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        assert _convert_bool_4d_mask_to_additive(x, torch.float32) is x
        boolean_2d = torch.ones(2, 3, dtype=torch.bool)
        assert _convert_bool_4d_mask_to_additive(boolean_2d, torch.float32) is boolean_2d

    def test_converts_bool_4d_to_additive(self):
        mask = torch.tensor([[[[True, False], [True, True]]]])  # shape (1,1,2,2)
        out = _convert_bool_4d_mask_to_additive(mask, torch.float32)
        assert out.dtype == torch.float32
        min_value = torch.finfo(torch.float32).min
        expected = torch.tensor([[[[0.0, min_value], [0.0, 0.0]]]])
        torch.testing.assert_close(out, expected)


class TestDerivePaddingMask:
    def test_2d_zero_is_padding(self):
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        out = _derive_padding_mask(attention_mask)
        expected = torch.tensor([[False, False, True], [False, True, True]])
        torch.testing.assert_close(out, expected)

    def test_4d_bool_uses_diagonal(self):
        # Build a 4D bool mask where diagonals indicate "valid" tokens.
        mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
        mask[0, 0].fill_diagonal_(True)
        # Sequence 1 has 2nd token padded
        mask[1, 0].fill_diagonal_(True)
        mask[1, 0, 1, 1] = False
        out = _derive_padding_mask(mask)
        # Padding mask flips: True where token is padding.
        expected = torch.tensor([[False, False, False], [False, True, False]])
        torch.testing.assert_close(out, expected)

    def test_4d_float_nonzero_diagonal_is_padding(self):
        """In additive attention masks, valid positions have 0 on the diagonal and
        padding positions have a large negative value; _derive_padding_mask treats
        nonzero diagonals as padding (True)."""
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        min_val = torch.finfo(torch.float32).min
        # Mark index 1 as padding (nonzero diagonal); 0 and 2 are valid (zero).
        mask[0, 0, 1, 1] = min_val
        out = _derive_padding_mask(mask)
        expected = torch.tensor([[False, True, False]])
        torch.testing.assert_close(out, expected)


class TestFallbackAdditiveMask:
    def test_lower_triangular_with_no_sliding(self):
        out = _fallback_additive_mask(1, 4, torch.float32, torch.device("cpu"))
        min_value = torch.finfo(torch.float32).min
        # Upper triangle entries above diagonal should be -inf
        assert torch.all(out[0, 0, 0, 1:] == min_value)
        assert out[0, 0, 0, 0].item() == 0.0
        # Diagonal and below are 0
        for i in range(4):
            for j in range(i + 1):
                assert out[0, 0, i, j].item() == 0.0

    def test_sliding_window_drops_far_keys(self):
        out = _fallback_additive_mask(1, 4, torch.float32, torch.device("cpu"), sliding_window=2)
        min_value = torch.finfo(torch.float32).min
        # Query=3, sliding_window=2 → keys 0 and 1 are masked (3-0>=2, 3-1>=2),
        # only 2 and 3 are allowed.
        assert out[0, 0, 3, 0].item() == min_value
        assert out[0, 0, 3, 1].item() == min_value
        assert out[0, 0, 3, 2].item() == 0.0
        assert out[0, 0, 3, 3].item() == 0.0

    def test_batch_expansion_with_2d_attention_mask(self):
        attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
        out = _fallback_additive_mask(1, 3, torch.float32, torch.device("cpu"), attention_mask=attention_mask)
        # Last key (index 2) is padded — masked for every query. The fallback
        # adds finfo.min for causal masking and adds finfo.min again for the
        # padding column, so cells with both masks overflow to -inf in fp32.
        # Either way, the key is hidden by softmax, so accept "very negative".
        threshold = torch.finfo(torch.float32).min
        assert out[0, 0, 0, 2].item() <= threshold
        assert out[0, 0, 1, 2].item() <= threshold
        assert out[0, 0, 2, 2].item() <= threshold


class TestMiMoV2FlashRotaryEmbedding:
    def test_rotary_dim_matches_partial_factor(self):
        rope = MiMoV2FlashRotaryEmbedding(rope_theta=10000.0, head_dim=16, partial_rotary_factor=0.5)
        # rotary_dim = 16 * 0.5 = 8 → inv_freq has 4 entries.
        assert rope.inv_freq.shape == (4,)

    def test_rotary_dim_rounded_down_to_even(self):
        # head_dim*factor = 9 → rounds to 8 → 4 inv_freq entries
        rope = MiMoV2FlashRotaryEmbedding(rope_theta=10000.0, head_dim=18, partial_rotary_factor=0.5)
        assert rope.inv_freq.shape == (4,)

    def test_zero_rotary_dim_raises(self):
        with pytest.raises(ValueError, match="Invalid rotary_dim"):
            MiMoV2FlashRotaryEmbedding(rope_theta=10000.0, head_dim=2, partial_rotary_factor=0.1)

    def test_forward_returns_concat_double_dim(self):
        rope = MiMoV2FlashRotaryEmbedding(
            rope_theta=10000.0, head_dim=8, partial_rotary_factor=1.0, dtype=torch.float32
        )
        batch, seq = 2, 5
        x = torch.zeros(batch, seq, 16, dtype=torch.float32)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(x, position_ids)
        # emb = concat(freqs, freqs) along -1 → final dim = 2*(rotary_dim/2)*2 = rotary_dim = 8.
        assert cos.shape == (batch, seq, 8)
        assert sin.shape == (batch, seq, 8)

    def test_position_zero_is_identity(self):
        rope = MiMoV2FlashRotaryEmbedding(
            rope_theta=10000.0, head_dim=8, partial_rotary_factor=1.0, dtype=torch.float32
        )
        x = torch.zeros(1, 1, 8, dtype=torch.float32)
        cos, sin = rope(x, torch.tensor([[0]]))
        torch.testing.assert_close(cos, torch.ones_like(cos))
        torch.testing.assert_close(sin, torch.zeros_like(sin))

    def test_inv_freq_first_entry_is_one(self):
        rope = MiMoV2FlashRotaryEmbedding(
            rope_theta=10000.0, head_dim=8, partial_rotary_factor=1.0, dtype=torch.float32
        )
        # inv_freq[0] = 1 / (rope_theta ** (0/rotary_dim)) = 1.0
        assert math.isclose(rope.inv_freq[0].item(), 1.0)


class TestMiMoV2RMSNorm:
    def test_init_weight_ones(self):
        norm = MiMoV2RMSNorm(hidden_size=8, dtype=torch.float32)
        torch.testing.assert_close(norm.weight, torch.ones(8))

    def test_reset_parameters_restores_ones(self):
        norm = MiMoV2RMSNorm(hidden_size=4, dtype=torch.float32)
        with torch.no_grad():
            norm.weight.fill_(2.5)
        norm.reset_parameters()
        torch.testing.assert_close(norm.weight, torch.ones(4))

    def test_forward_preserves_dtype(self):
        norm = MiMoV2RMSNorm(hidden_size=8, dtype=torch.float32)
        x = torch.randn(2, 4, 8, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32
        assert out.shape == x.shape

    def test_forward_unit_input_unit_output(self):
        """Input rms=1 → output equals weight (broadcast)."""
        norm = MiMoV2RMSNorm(hidden_size=4, eps=1e-6, dtype=torch.float32)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        x = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        out = norm(x)
        # variance = 1, rsqrt(1) = 1, out = weight
        torch.testing.assert_close(out[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-4, rtol=1e-4)
