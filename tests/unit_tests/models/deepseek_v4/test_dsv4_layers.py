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

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.layers import (
    GroupedOutputProjection,
    hc_post_approx,
    hc_pre_approx,
)


class TestGroupedOutputProjection:
    def test_weight_shape(self):
        n_heads, head_dim, o_lora_rank, n_groups = 64, 512, 1024, 8
        proj = GroupedOutputProjection(n_heads, head_dim, o_lora_rank, n_groups)
        # [n_groups * o_lora_rank, n_heads_per_group * head_dim]
        assert proj.weight.shape == (n_groups * o_lora_rank, (n_heads // n_groups) * head_dim)
        assert proj.weight.shape == (8192, 4096)

    def test_forward_bshd_shape(self):
        n_heads, head_dim, o_lora_rank, n_groups = 64, 512, 1024, 8
        proj = GroupedOutputProjection(n_heads, head_dim, o_lora_rank, n_groups)
        bsz, seq = 2, 4
        o = torch.randn(bsz, seq, n_heads, head_dim)
        out = proj(o)
        assert out.shape == (bsz, seq, n_groups * o_lora_rank)

    def test_forward_thd_shape(self):
        n_heads, head_dim, o_lora_rank, n_groups = 64, 512, 1024, 8
        proj = GroupedOutputProjection(n_heads, head_dim, o_lora_rank, n_groups)
        ntok = 7
        o = torch.randn(ntok, n_heads, head_dim)
        out = proj(o)
        assert out.shape == (ntok, n_groups * o_lora_rank)

    def test_requires_divisible_heads(self):
        with pytest.raises(AssertionError):
            GroupedOutputProjection(n_heads=7, head_dim=64, o_lora_rank=128, n_groups=4)

    def test_init_weights(self):
        proj = GroupedOutputProjection(8, 64, 128, 4)
        # After init_weights the weight should not be all zeros
        proj.init_weights(init_std=0.02)
        assert proj.weight.abs().max() > 0


class TestHCApprox:
    def test_hc_pre_output_shape(self):
        bsz, seq, hc_mult, dim = 2, 8, 4, 16
        x = torch.randn(bsz, seq, hc_mult, dim)
        reduced, post, comb = hc_pre_approx(x)
        assert reduced.shape == (bsz, seq, dim)
        assert post is None
        assert comb is None

    def test_hc_pre_is_mean(self):
        bsz, seq, hc_mult, dim = 1, 1, 4, 8
        x = torch.randn(bsz, seq, hc_mult, dim)
        reduced, _, _ = hc_pre_approx(x)
        assert torch.allclose(reduced, x.mean(dim=2))

    def test_hc_post_output_shape(self):
        bsz, seq, hc_mult, dim = 2, 8, 4, 16
        x = torch.randn(bsz, seq, dim)
        residual = torch.randn(bsz, seq, hc_mult, dim)
        out = hc_post_approx(x, residual, None, None)
        assert out.shape == (bsz, seq, hc_mult, dim)

    def test_hc_post_is_broadcast_add(self):
        bsz, seq, hc_mult, dim = 1, 3, 4, 5
        x = torch.ones(bsz, seq, dim)
        residual = torch.zeros(bsz, seq, hc_mult, dim)
        out = hc_post_approx(x, residual, None, None)
        # Every hc copy should equal x
        assert torch.allclose(out, x.unsqueeze(2).expand_as(out))

    def test_hc_roundtrip_shape_consistent(self):
        """hc_pre followed by hc_post should return the original shape."""
        bsz, seq, hc_mult, dim = 2, 6, 4, 32
        x = torch.randn(bsz, seq, hc_mult, dim)
        reduced, post, comb = hc_pre_approx(x)
        out = hc_post_approx(reduced, x, post, comb)
        assert out.shape == x.shape
