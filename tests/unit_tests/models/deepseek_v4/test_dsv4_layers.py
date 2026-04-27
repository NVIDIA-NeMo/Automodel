# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the standalone helpers in
``nemo_automodel.components.models.deepseek_v4.layers``.

Pieces here are easy to construct in isolation (grouped output projection,
the Hyper-Connections weight builder, and the partial-RoPE helper).
Full-model behaviour is covered by ``test_dsv4_model_smoke.py``.
"""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4GroupedLinear,
    DeepseekV4HyperConnection,
    _apply_partial_rope_interleaved,
)


class TestDeepseekV4GroupedLinear:
    """Block-diagonal grouped linear backing the V4 attention output
    projection ``wo_a`` (n_groups=8 in DSV4-Flash).
    """

    def test_weight_shape(self):
        # n_heads=64, head_dim=512, n_groups=8 -> in_per_group=4096, out_total=8192
        proj = DeepseekV4GroupedLinear(in_features_per_group=4096, out_features=8192, n_groups=8)
        assert proj.weight.shape == (8192, 4096)
        assert proj.bias is None  # bias=False default

    def test_n_groups_attribute(self):
        proj = DeepseekV4GroupedLinear(64, 256, n_groups=4)
        assert proj.n_groups == 4

    def test_forward_shape(self):
        bsz, seq, n_groups, in_per = 2, 4, 8, 64
        out_total = 256
        proj = DeepseekV4GroupedLinear(in_per, out_total, n_groups=n_groups)
        x = torch.randn(bsz, seq, n_groups, in_per)
        out = proj(x)
        assert out.shape == (bsz, seq, n_groups, out_total // n_groups)

    def test_forward_matches_per_group_matmul(self):
        """Output should equal a per-group matmul: y[g] = x[g] @ W[g].T."""
        n_groups, in_per = 4, 8
        out_total = 16
        proj = DeepseekV4GroupedLinear(in_per, out_total, n_groups=n_groups)
        with torch.no_grad():
            proj.weight.normal_()
        x = torch.randn(3, n_groups, in_per)
        out = proj(x)
        w_ref = proj.weight.view(n_groups, out_total // n_groups, in_per)
        out_ref = torch.stack([x[:, g, :] @ w_ref[g].t() for g in range(n_groups)], dim=1)
        torch.testing.assert_close(out, out_ref)


class TestDeepseekV4HyperConnection:
    """``compute_weights`` returns the (pre, post, comb) tensors used at
    each HC site.  Shapes are deterministic given ``hc_mult`` and
    ``hidden_size``; values change with the learned parameters.
    """

    @pytest.fixture
    def hc(self):
        return DeepseekV4HyperConnection(
            hc_mult=4,
            hidden_size=16,
            hc_sinkhorn_iters=4,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
        )

    def test_parameter_dtypes_are_fp32(self, hc):
        # HC params must stay fp32 even when the surrounding model is bf16.
        assert hc.fn.dtype == torch.float32
        assert hc.base.dtype == torch.float32
        assert hc.scale.dtype == torch.float32

    def test_compute_weights_output_shapes(self, hc):
        bsz, seq, hc_mult, hidden = 2, 5, 4, 16
        x = torch.randn(bsz, seq, hc_mult, hidden)
        pre, post, comb = hc.compute_weights(x)
        assert pre.shape == (bsz, seq, hc_mult)
        assert post.shape == (bsz, seq, hc_mult)
        assert comb.shape == (bsz, seq, hc_mult, hc_mult)

    def test_post_uses_2x_sigmoid(self, hc):
        """``post`` is ``2 * sigmoid(...)``, so it can exceed 1."""
        x = torch.randn(2, 3, 4, 16)
        with torch.no_grad():
            hc.scale.data[1].fill_(50.0)
            hc.base.data[hc.hc_mult : 2 * hc.hc_mult].fill_(50.0)
        _, post, _ = hc.compute_weights(x)
        assert post.max().item() > 1.5

    def test_comb_rows_sum_close_to_one(self, hc):
        """After softmax+sinkhorn, ``comb`` is doubly-(near-)stochastic."""
        x = torch.randn(1, 2, 4, 16)
        _, _, comb = hc.compute_weights(x)
        row_sums = comb.sum(dim=-1)
        col_sums = comb.sum(dim=-2)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-2)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), rtol=0, atol=1e-2)


class TestApplyPartialRopeInterleaved:
    """Released DSV4-Flash uses INTERLEAVED RoPE pairs ``(2k, 2k+1)``."""

    def _make_cos_sin(self, batch, seq, rd):
        """Build the Llama-style ``cat([f, f], -1)`` cos/sin tensors that
        ``_apply_partial_rope_interleaved`` consumes (it uses only the
        first half).
        """
        half = rd // 2
        freqs = torch.arange(half, dtype=torch.float32) * 0.1
        pos = torch.arange(seq, dtype=torch.float32).unsqueeze(-1) * freqs
        cos_h = pos.cos()
        sin_h = pos.sin()
        cos = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        sin = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        return cos, sin

    def test_preserves_nope_prefix(self):
        bsz, heads, seq, rd, nope = 1, 2, 4, 8, 16
        x = torch.randn(bsz, heads, seq, nope + rd)
        cos, sin = self._make_cos_sin(bsz, seq, rd)
        y = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        torch.testing.assert_close(y[..., :nope], x[..., :nope])

    def test_inverse_with_negated_sin_round_trips(self):
        """Rotating with ``sin`` then ``-sin`` recovers the input."""
        bsz, heads, seq, rd = 2, 4, 3, 8
        x = torch.randn(bsz, heads, seq, rd + 4)
        cos, sin = self._make_cos_sin(bsz, seq, rd)
        rotated = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        unrotated = _apply_partial_rope_interleaved(rotated, cos, -sin, rope_head_dim=rd)
        torch.testing.assert_close(unrotated, x, rtol=1e-4, atol=1e-5)

    def test_zero_angles_is_identity(self):
        bsz, heads, seq, rd = 1, 1, 2, 4
        x = torch.randn(bsz, heads, seq, rd)
        cos = torch.ones(bsz, seq, rd)
        sin = torch.zeros(bsz, seq, rd)
        y = _apply_partial_rope_interleaved(x, cos, sin, rope_head_dim=rd)
        torch.testing.assert_close(y, x)
