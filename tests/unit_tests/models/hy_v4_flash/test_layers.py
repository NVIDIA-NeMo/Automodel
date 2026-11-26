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

from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.layers import (
    HYV4FlashHyperConnection,
    HYV4FlashRMSNorm,
    apply_interleaved_rope,
    inverse_frequencies,
    mrope_cos_sin,
)
from nemo_automodel.components.moe.layers import MoE
from tests.unit_tests.models.hy_v4_flash.conftest import build_model, complex_rotation


def _oracle_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """The shared complex-rotation oracle on [batch, sequence, 1, rope_dim] inputs with [batch, sequence] positions."""
    return torch.stack([complex_rotation(x[row, :, 0], positions[row], theta) for row in range(x.shape[0])])[:, :, None]


@pytest.mark.parametrize("theta", [1.0e4, 1.0e7])
def test_text_rope_matches_complex_oracle_at_nonzero_positions(theta: float) -> None:
    torch.manual_seed(0)
    rope_dim, section = 64, [12, 10, 10]
    x = torch.randn(2, 5, 1, rope_dim)
    pos = torch.tensor([[7, 8, 9, 1000, 1001], [0, 3, 4, 5, 3000]])
    cos, sin = mrope_cos_sin(pos[None].expand(3, -1, -1), inverse_frequencies(rope_dim, theta), section, torch.float32)
    got = apply_interleaved_rope(x, cos.unsqueeze(2), sin.unsqueeze(2)).double()
    torch.testing.assert_close(got, _oracle_rope(x, pos, theta).to(got.dtype), rtol=1e-4, atol=1e-4)


def test_rope_theta_negative_control_is_detected() -> None:
    rope_dim = 64
    x = torch.randn(1, 4, 1, rope_dim)
    pos = torch.tensor([[100, 200, 300, 400]])
    cos, sin = mrope_cos_sin(
        pos[None].expand(3, -1, -1), inverse_frequencies(rope_dim, 1.0e4), [12, 10, 10], torch.float32
    )
    wrong = _oracle_rope(x, pos, 1.0e7).float()
    assert not torch.allclose(apply_interleaved_rope(x, cos.unsqueeze(2), sin.unsqueeze(2)), wrong, atol=1e-2)


def test_mrope_interleaves_height_and_width_planes() -> None:
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


def test_rms_norm_casts_before_weight_multiply() -> None:
    norm = HYV4FlashRMSNorm(8, eps=1e-5)
    norm.weight.data.fill_(1.0000001)
    x = torch.randn(3, 8).bfloat16()
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(norm(x), (norm.weight * ref.to(torch.bfloat16)), rtol=0, atol=0)


def test_hyper_connection_collapse_and_expand() -> None:
    cfg = HYV4FlashTextConfig(hidden_size=16, num_hidden_layers=2, layer_types=["sliding_attention"] * 2)
    hc = HYV4FlashHyperConnection(cfg)
    hc.init_weights(cfg.initializer_range)
    with torch.no_grad():  # distinct pre / post scales and biases, so swapping the two groups is visible
        hc.scale.copy_(torch.tensor([0.7, 1.3]))
        hc.base.copy_(torch.randn(2 * cfg.hc_mult))
    streams = torch.randn(2, 3, cfg.hc_mult, 16)
    post, collapsed = hc(streams)
    flat = streams.flatten(2)
    mixes = flat @ hc.fn.T * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    pre = torch.sigmoid(mixes[..., :4] * hc.scale[0] + hc.base[:4]) + cfg.hc_eps
    torch.testing.assert_close(collapsed, (pre.unsqueeze(-1) * streams).sum(2), rtol=1e-5, atol=1e-6)
    assert post.shape == (2, 3, 4) and post.dtype == torch.float32
    expected_post = cfg.hc_magnitude * torch.sigmoid(mixes[..., 4:] * hc.scale[1] + hc.base[4:]) + cfg.hc_eps
    torch.testing.assert_close(post, expected_post, rtol=1e-5, atol=1e-6)
    out = torch.randn(2, 3, 16)
    torch.testing.assert_close(hc.expand(post, out, streams), post.unsqueeze(-1) * out.unsqueeze(2) + streams)


def test_fp32_parameter_forward_returns_a_copy_that_survives_freed_parameter_storage() -> None:
    """FSDP frees the unsharded parameter storage when a unit reshards after forward; the result must not alias it."""
    from nemo_automodel.components.models.hy_v4_flash.attention import DeepseekV4FP32Parameter

    holder = DeepseekV4FP32Parameter(torch.arange(6.0).reshape(2, 3))  # the holder of the sinks and the indexer APE
    out = holder(torch.zeros(1))
    assert out.dtype == torch.float32 and out.requires_grad
    assert out.data_ptr() != holder.weight.data_ptr()
    expected = holder.weight.detach().clone()
    with torch.no_grad():
        holder.weight.zero_()  # stands in for the resharded (freed and reused) storage
    torch.testing.assert_close(out.detach(), expected)
    out.sum().backward()
    torch.testing.assert_close(holder.weight.grad, torch.ones(2, 3))


def _moe_oracle(
    mlp: MoE, x: torch.Tensor, limit: float, *, clamp_routed: bool = True, clamp_shared: bool = False
) -> torch.Tensor:
    """Latent MoE as the reference writes it, with the routed and the shared clamp switchable.

    Returns the output [tokens, hidden] for ``x`` [tokens, hidden] and the largest routed and shared pre-activation
    magnitudes.
    """
    import torch.nn.functional as F

    def swiglu(gate: torch.Tensor, up: torch.Tensor, clamp: bool) -> torch.Tensor:
        if clamp:
            gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        return F.silu(gate) * up

    scores = torch.sigmoid(x @ mlp.gate.weight.T)
    top = (scores + mlp.gate.e_score_correction_bias).topk(mlp.gate.topk, dim=-1).indices
    weights = scores.gather(1, top)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * mlp.gate.route_scale
    latent = x @ mlp.fc1_latent_proj.weight.T
    routed = torch.zeros_like(latent)
    routed_peak = 0.0
    for slot in range(top.shape[1]):
        gate_up = torch.einsum("ti,tio->to", latent, mlp.experts.gate_and_up_projs[top[:, slot]])
        gate, up = gate_up.chunk(2, dim=-1)
        routed_peak = max(routed_peak, float(gate.max()), float(up.abs().max()))
        hidden = swiglu(gate, up, clamp_routed)
        routed += torch.einsum("ti,tio->to", hidden, mlp.experts.down_projs[top[:, slot]]) * weights[:, slot, None]
    shared = mlp.shared_experts
    gate, up = shared.gate_proj(x), shared.up_proj(x)
    out = routed @ mlp.fc2_latent_proj.weight.T + shared.down_proj(swiglu(gate, up, clamp_shared))
    return out, routed_peak, max(float(gate.max()), float(up.abs().max()))


def test_swiglu_clamp_applies_to_routed_experts_and_not_to_the_shared_expert() -> None:
    """Upstream clamps only the routed experts, unlike the shared MoE.

    The input is scaled until both kinds of pre-activation exceed the limit; the two wrong variants must not match.
    """
    model = build_model()
    mlp = next(layer.mlp for layer in model.model.layers if type(layer.mlp).__name__ == "MoE")
    limit = model.config.text_config.swiglu_limit
    torch.manual_seed(3)
    x = torch.randn(40, model.config.text_config.hidden_size) * 400.0
    with torch.no_grad():
        mlp.experts.gate_and_up_projs.mul_(30.0)  # the latent projection shrinks the routed input; scale it back up
        ours = mlp(x[None])[0]
        expected, routed_peak, shared_peak = _moe_oracle(mlp, x, limit)
        shared_clamped, _, _ = _moe_oracle(mlp, x, limit, clamp_shared=True)
        routed_unclamped, _, _ = _moe_oracle(mlp, x, limit, clamp_routed=False)
    assert routed_peak > limit and shared_peak > limit  # the clamp is actually exercised on both paths
    torch.testing.assert_close(ours, expected, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(ours, shared_clamped, rtol=1e-2, atol=1e-2)
    assert not torch.allclose(ours, routed_unclamped, rtol=1e-2, atol=1e-2)
