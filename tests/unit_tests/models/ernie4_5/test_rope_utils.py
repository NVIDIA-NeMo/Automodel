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

from types import SimpleNamespace

import torch

from nemo_automodel.components.models.ernie4_5.rope_utils import (
    Ernie4_5RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_every_two,
)


class TestRotateEveryTwo:
    def test_preserves_shape(self):
        x = torch.randn(2, 3, 4, 8)
        out = rotate_every_two(x)
        assert out.shape == x.shape

    def test_pair_swap_with_negation(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_every_two(x)
        expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
        torch.testing.assert_close(out, expected)

    def test_double_rotation_is_negation(self):
        """Rotating twice should flip the sign: (a,b)->(-b,a)->(-a,-b)."""
        x = torch.randn(2, 6)
        twice = rotate_every_two(rotate_every_two(x))
        torch.testing.assert_close(twice, -x)

    def test_handles_higher_rank(self):
        x = torch.randn(1, 5, 7, 4, 10)
        out = rotate_every_two(x)
        assert out.shape == x.shape


class TestApplyRotaryPosEmb:
    def test_identity_when_cos_one_sin_zero(self):
        """cos=1, sin=0 should leave q/k unchanged."""
        batch, seq, heads, dim = 1, 3, 2, 8
        q = torch.randn(batch, seq, heads, dim)
        k = torch.randn(batch, seq, heads, dim)
        cos = torch.ones(batch, seq, dim)
        sin = torch.zeros(batch, seq, dim)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(q_out, q)
        torch.testing.assert_close(k_out, k)

    def test_preserves_input_dtype(self):
        batch, seq, heads, dim = 1, 2, 1, 4
        q = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16)
        k = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16)
        cos = torch.ones(batch, seq, dim)
        sin = torch.zeros(batch, seq, dim)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.dtype == torch.bfloat16
        assert k_out.dtype == torch.bfloat16

    def test_matches_manual_formula(self):
        """Validates q_embed = q*cos + rotate_every_two(q)*sin."""
        batch, seq, heads, dim = 1, 2, 1, 4
        q = torch.randn(batch, seq, heads, dim, dtype=torch.float32)
        k = torch.randn(batch, seq, heads, dim, dtype=torch.float32)
        cos = torch.randn(batch, seq, dim)
        sin = torch.randn(batch, seq, dim)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

        cos_u = cos.unsqueeze(-2)
        sin_u = sin.unsqueeze(-2)
        expected_q = q * cos_u + rotate_every_two(q) * sin_u
        expected_k = k * cos_u + rotate_every_two(k) * sin_u
        torch.testing.assert_close(q_out, expected_q)
        torch.testing.assert_close(k_out, expected_k)


def _make_config(
    *,
    head_dim: int | None = 8,
    num_attention_heads: int = 4,
    hidden_size: int = 32,
    rope_theta: float = 500000.0,
    rope_parameters: dict | None = None,
):
    cfg = SimpleNamespace(
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        rope_theta=rope_theta,
        rope_parameters=rope_parameters,
    )
    return cfg


class TestErnie4_5RotaryEmbedding:
    def test_inv_freq_uses_head_dim(self):
        cfg = _make_config(head_dim=16)
        rope = Ernie4_5RotaryEmbedding(cfg)
        assert rope.inv_freq.shape == (8,)
        assert rope.inv_freq.dtype == torch.float32

    def test_inv_freq_falls_back_to_hidden_over_heads(self):
        cfg = _make_config(head_dim=None, hidden_size=64, num_attention_heads=8)
        rope = Ernie4_5RotaryEmbedding(cfg)
        # head_dim derived = 64 / 8 = 8, so inv_freq has 4 entries
        assert rope.inv_freq.shape == (4,)

    def test_rope_theta_default(self):
        cfg = _make_config(rope_theta=10000.0)
        rope = Ernie4_5RotaryEmbedding(cfg)
        # inv_freq[0] = 1 / (10000^(0/head_dim)) = 1.0
        assert torch.isclose(rope.inv_freq[0], torch.tensor(1.0))

    def test_rope_theta_from_rope_parameters(self):
        """rope_parameters takes precedence when present."""
        cfg = _make_config(rope_theta=10000.0, rope_parameters={"rope_theta": 1000.0})
        rope = Ernie4_5RotaryEmbedding(cfg)
        # inv_freq[1] = 1 / (1000^(2/head_dim)) — different from rope_theta=10000 case
        expected = 1.0 / (1000.0 ** (2.0 / 8))
        assert torch.isclose(rope.inv_freq[1], torch.tensor(expected, dtype=torch.float32))

    def test_attention_scaling_default(self):
        cfg = _make_config()
        rope = Ernie4_5RotaryEmbedding(cfg)
        assert rope.attention_scaling == 1.0

    def test_forward_bshd_shape(self):
        cfg = _make_config(head_dim=8)
        rope = Ernie4_5RotaryEmbedding(cfg)
        batch, seq = 2, 5
        x = torch.randn(batch, seq, 32)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(x, position_ids)
        assert cos.shape == (batch, seq, 8)
        assert sin.shape == (batch, seq, 8)

    def test_forward_thd_squeezes_batch_dim(self):
        cfg = _make_config(head_dim=8)
        rope = Ernie4_5RotaryEmbedding(cfg)
        seq = 5
        x = torch.randn(seq, 32)
        position_ids = torch.arange(seq)
        cos, sin = rope(x, position_ids, qkv_format="thd")
        assert cos.shape == (seq, 8)
        assert sin.shape == (seq, 8)

    def test_forward_handles_1d_position_ids(self):
        cfg = _make_config(head_dim=8)
        rope = Ernie4_5RotaryEmbedding(cfg)
        seq = 4
        x = torch.randn(1, seq, 32)
        position_ids = torch.arange(seq)  # 1-D, will be unsqueezed
        cos, sin = rope(x, position_ids)
        assert cos.shape == (1, seq, 8)
        assert sin.shape == (1, seq, 8)

    def test_forward_interleaves_cos_sin(self):
        """ERNIE 4.5 uses repeat_interleave so adjacent pairs of cos[..., i] match."""
        cfg = _make_config(head_dim=8)
        rope = Ernie4_5RotaryEmbedding(cfg)
        x = torch.zeros(1, 1, 1)
        position_ids = torch.tensor([[3]])
        cos, sin = rope(x, position_ids)
        # cos[..., 0] should equal cos[..., 1] (interleaved pairs)
        torch.testing.assert_close(cos[..., 0::2], cos[..., 1::2])
        torch.testing.assert_close(sin[..., 0::2], sin[..., 1::2])

    def test_forward_position_zero_produces_identity(self):
        """At position 0 angles are 0, so cos=1 and sin=0."""
        cfg = _make_config(head_dim=8)
        rope = Ernie4_5RotaryEmbedding(cfg)
        x = torch.zeros(1, 1, 1)
        position_ids = torch.tensor([[0]])
        cos, sin = rope(x, position_ids)
        torch.testing.assert_close(cos, torch.ones_like(cos))
        torch.testing.assert_close(sin, torch.zeros_like(sin))
