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

from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.layers import (
    HY4FlashHyperConnection,
    HY4FlashRMSNorm,
    apply_interleaved_rope,
    inverse_frequencies,
    mrope_cos_sin,
)


def _oracle_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Complex-number RoPE on interleaved pairs, written independently of the production helper."""
    rope_dim = x.shape[-1]
    out = torch.empty_like(x, dtype=torch.float64)
    for b in range(x.shape[0]):
        for s in range(x.shape[1]):
            for i in range(rope_dim // 2):
                angle = float(positions[b, s]) * theta ** (-2.0 * i / rope_dim)
                z = complex(float(x[b, s, 0, 2 * i]), float(x[b, s, 0, 2 * i + 1])) * complex(
                    math.cos(angle), math.sin(angle)
                )
                out[b, s, 0, 2 * i], out[b, s, 0, 2 * i + 1] = z.real, z.imag
    return out


@pytest.mark.parametrize("theta", [1.0e4, 1.0e7])
def test_text_rope_matches_complex_oracle_at_nonzero_positions(theta):
    torch.manual_seed(0)
    rope_dim, section = 64, [12, 10, 10]
    x = torch.randn(2, 5, 1, rope_dim)
    pos = torch.tensor([[7, 8, 9, 1000, 1001], [0, 3, 4, 5, 3000]])
    cos, sin = mrope_cos_sin(pos[None].expand(3, -1, -1), inverse_frequencies(rope_dim, theta), section, torch.float32)
    got = apply_interleaved_rope(x, cos, sin).double()
    torch.testing.assert_close(got, _oracle_rope(x, pos, theta).to(got.dtype), rtol=1e-4, atol=1e-4)


def test_rope_theta_negative_control_is_detected():
    rope_dim = 64
    x = torch.randn(1, 4, 1, rope_dim)
    pos = torch.tensor([[100, 200, 300, 400]])
    cos, sin = mrope_cos_sin(pos[None].expand(3, -1, -1), inverse_frequencies(rope_dim, 1.0e4), [12, 10, 10], torch.float32)
    wrong = _oracle_rope(x, pos, 1.0e7).float()
    assert not torch.allclose(apply_interleaved_rope(x, cos, sin), wrong, atol=1e-2)


def test_mrope_interleaves_height_and_width_planes():
    section, rope_dim = [12, 10, 10], 64
    inv = torch.ones(rope_dim // 2)
    pos = torch.stack([torch.full((1, 1), 1), torch.full((1, 1), 2), torch.full((1, 1), 3)])  # t=1,h=2,w=3
    cos, _ = mrope_cos_sin(pos, inv, section, torch.float32)
    angle = torch.acos(cos[0, 0]).round().long()  # angle == position of the plane that owns the slot
    expected = torch.ones(32, dtype=torch.long)
    expected[1:30:3] = 2
    expected[2:30:3] = 3
    # acos(cos(3)) = 3 - 2*pi*0 for angles in [0, pi]; 1, 2 and 3 are all below pi.
    assert torch.equal(angle, expected)


def test_rms_norm_casts_before_weight_multiply():
    norm = HY4FlashRMSNorm(8, eps=1e-5)
    norm.weight.data.fill_(1.0000001)
    x = torch.randn(3, 8).bfloat16()
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(norm(x), (norm.weight * ref.to(torch.bfloat16)), rtol=0, atol=0)


def test_hyper_connection_collapse_and_expand():
    cfg = HY4FlashTextConfig(hidden_size=16, num_hidden_layers=2, layer_types=["sliding_attention"] * 2)
    hc = HY4FlashHyperConnection(cfg)
    hc.init_weights(cfg.initializer_range)
    streams = torch.randn(2, 3, cfg.hc_mult, 16)
    post, collapsed = hc(streams)
    flat = streams.flatten(2)
    mixes = flat @ hc.fn.T * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    pre = torch.sigmoid(mixes[..., :4] * hc.scale[0] + hc.base[:4]) + cfg.hc_eps
    torch.testing.assert_close(collapsed, (pre.unsqueeze(-1) * streams).sum(2), rtol=1e-5, atol=1e-6)
    assert post.shape == (2, 3, 4) and post.dtype == torch.float32
    out = torch.randn(2, 3, 16)
    torch.testing.assert_close(hc.expand(post, out, streams), post.unsqueeze(-1) * out.unsqueeze(2) + streams)
