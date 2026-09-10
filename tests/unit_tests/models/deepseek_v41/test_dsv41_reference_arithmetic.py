# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Independent released equations for the V4.1 precision-sensitive boundaries.

These CPU tests use literal mathematical oracles and separate reference leaves;
no downloaded inference module or production function supplies expected outputs.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v41.layers import (
    DeepseekV41Compressor,
    DeepseekV41HyperConnection,
    DeepseekV41Indexer,
    DeepseekV41RotaryEmbedding,
    DeepseekV41SharedState,
    _apply_partial_rope,
    fake_quant_fp4,
    fake_quant_fp8,
    hc_collapse,
    hc_expand,
)
from tests.unit_tests.models.deepseek_v41.conftest import tiny_backend, tiny_config


def _reference_frequencies(positions, dim, theta, scaling):
    frequencies = 1 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if scaling:
        original = scaling["original_max_position_embeddings"]
        low = max(
            math.floor(dim * math.log(original / (scaling["beta_fast"] * 2 * math.pi)) / (2 * math.log(theta))), 0
        )
        high = min(
            math.ceil(dim * math.log(original / (scaling["beta_slow"] * 2 * math.pi)) / (2 * math.log(theta))), dim - 1
        )
        ramp = ((torch.arange(dim // 2).float() - low) / max(high - low, 1e-3)).clamp(0, 1)
        frequencies = frequencies / scaling["factor"] * ramp + frequencies * (1 - ramp)
    angles = positions.float().unsqueeze(-1) * frequencies
    return torch.polar(torch.ones_like(angles), angles)


@pytest.mark.parametrize("use_yarn", [False, True])
@pytest.mark.parametrize("heads", [None, 3])
def test_rotary_matches_complex_reference_after_bf16_cast_and_backward(use_yarn, heads):
    torch.manual_seed(94)
    config = tiny_config()
    scaling = config.rope_scaling if use_yarn else None
    module = DeepseekV41RotaryEmbedding(config.rope_theta, 64, 0.25, rope_scaling=scaling).bfloat16()
    positions = torch.tensor([[0, 1, 63, 64, 257, 4096], [0, 3, 0, 1, 9, 65536]])
    shape = (2, 6, 64) if heads is None else (2, heads, 6, 64)
    x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    phases = _reference_frequencies(positions, 16, config.rope_theta, scaling)
    cos, sin = module(x, positions)
    assert cos.dtype == sin.dtype == torch.float32
    torch.testing.assert_close(cos[..., :8], phases.real, rtol=0, atol=0)
    torch.testing.assert_close(sin[..., :8], phases.imag, rtol=0, atol=0)
    if heads is not None:
        phases = phases.unsqueeze(1)
    for inverse in (False, True):
        sign = -1 if inverse else 1
        actual = _apply_partial_rope(x, cos, sign * sin, 16)
        pairs = torch.view_as_complex(reference_x[..., -16:].float().reshape(*shape[:-1], 8, 2))
        rotated = torch.view_as_real(pairs * (phases.conj() if inverse else phases)).flatten(-2).bfloat16()
        expected = torch.cat([reference_x[..., :-16], rotated], -1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        upstream = torch.randn_like(actual)
        actual.backward(upstream, retain_graph=True)
        expected.backward(upstream, retain_graph=True)
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
        x.grad = reference_x.grad = None


def _mhc_reference(x, parameters, streams, norm_eps, hc_eps, repeat):
    flat = x.flatten(2).float()
    projected = F.linear(flat, parameters["fn"]) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    scales = parameters["scale"].repeat_interleave(torch.tensor([streams, streams, streams * streams]))
    logits = torch.addcmul(parameters["base"], projected, scales)
    pre = logits[..., :streams].sigmoid() + hc_eps
    post = 2 * logits[..., streams : 2 * streams].sigmoid()
    comb = logits[..., 2 * streams :].reshape(*x.shape[:2], streams, streams).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mhc_coefficients_collapse_expand_and_gradients_match_released_equations(dtype):
    torch.manual_seed(43)
    module = DeepseekV41HyperConnection(4, 16, 4, 1e-6, 1e-6, sinkhorn_backend="torch")
    with torch.no_grad():
        module.fn.normal_(std=0.1)
        module.base.normal_(std=0.3)
        module.scale.copy_(torch.tensor([0.9, 1.1, 0.7]))
    reference_parameters = {name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()}
    x = torch.randn(2, 3, 4, 16, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual_mix = module(x)
    expected_mix = _mhc_reference(reference_x, reference_parameters, 4, 1e-6, 1e-6, 4)
    for actual, expected in zip(actual_mix, expected_mix):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    previous = torch.rand(2, 3, 4, requires_grad=True)
    reference_previous = previous.detach().clone().requires_grad_()
    collapsed = hc_collapse(x, previous)
    reference_collapsed = (reference_previous[..., None] * reference_x.float()).sum(2).to(dtype)
    torch.testing.assert_close(collapsed, reference_collapsed, rtol=0, atol=0)
    actual = hc_expand(collapsed, x, actual_mix[1], actual_mix[2])
    # Share each FP32 cast across output streams. Repeating the cast inside
    # the loop would round separate BF16 gradient contributions prematurely.
    reference_collapsed_fp32 = reference_collapsed.float()
    reference_x_fp32 = reference_x.float()
    expected = torch.stack(
        [
            reference_collapsed_fp32 * expected_mix[1][..., out, None]
            + (reference_x_fp32 * expected_mix[2][..., :, out, None]).sum(2)
            for out in range(4)
        ],
        2,
    ).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    loss = (actual * upstream).sum() + actual_mix[0].square().sum()
    reference_loss = (expected * upstream).sum() + expected_mix[0].square().sum()
    loss.backward()
    reference_loss.backward()
    tolerance = dict(atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
    torch.testing.assert_close(previous.grad, reference_previous.grad, **tolerance)
    for name, parameter in module.named_parameters():
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, **tolerance)


def _independent_mx_scale(amax, max_value):
    # frexp observes the exact FP32 reciprocal-product value, without the
    # production IEEE754 bit reinterpretation or the old inaccurate log2 path.
    scaled = float(torch.tensor(amax, dtype=torch.float32) * (1 / max_value))
    mantissa, exponent = math.frexp(scaled)
    return math.ldexp(1.0, exponent - (mantissa == 0.5))


@pytest.mark.parametrize("format", ["fp8", "e8m0", "e4m3"])
@pytest.mark.parametrize("exponent", [-8, 0, 8])
def test_cache_quantization_boundary_values_signed_zero_and_ste(format, exponent):
    max_value = 448.0 if format == "fp8" else 6.0
    unit = 2.0**exponent
    boundary = torch.tensor(max_value * unit)
    maxima = [
        torch.nextafter(boundary, torch.tensor(-torch.inf)),
        boundary,
        torch.nextafter(boundary, torch.tensor(torch.inf)),
        torch.tensor(0.0),
    ]
    block_size = 16 if format == "e4m3" else 32
    rows = []
    for maximum in maxima:
        row = torch.zeros(block_size)
        row[:10] = torch.tensor([-0.0, 0.0, -0.25, 0.5, 0.75, 1.75, 3.5, -5.0, 1 / 512, 1.5]) * unit
        if maximum == 0:
            row.zero_()
            row[0] = -0.0
        else:
            row[-1] = maximum
        rows.append(row)
    x = torch.stack(rows).requires_grad_()
    actual = fake_quant_fp8(x, block_size) if format == "fp8" else fake_quant_fp4(x, block_size, format)
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    expected_rows = []
    for row in x.detach():
        amax = float(row.abs().max())
        if format == "e4m3":
            scale = float(torch.tensor(max(amax, 6 * 2.0**-9) / 6).to(torch.float8_e4m3fn).float())
        else:
            floor = 1e-4 if format == "fp8" else 6 * 2.0**-126
            scale = _independent_mx_scale(max(amax, floor), max_value)
        if format == "fp8":
            expected_rows.append((row / scale).to(torch.float8_e4m3fn).float() * scale)
        else:
            values = []
            for value in (row / scale).tolist():
                index = min(range(len(grid)), key=lambda i: (abs(abs(value) - grid[i]), i % 2))
                values.append(math.copysign(grid[index] * scale, value))
            expected_rows.append(torch.tensor(values))
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    if format in ("fp8", "e8m0") and exponent == 8:
        # Negative control: at large exponents, float32 log2 can round the
        # adjacent-above power back down, producing the old wrong MX scale.
        old_scale = torch.exp2(torch.ceil(torch.log2(maxima[2] / max_value)))
        assert _independent_mx_scale(float(maxima[2]), max_value) != old_scale.item()
    upstream = torch.arange(actual.numel()).reshape_as(actual).float() / 7
    actual.backward(upstream)
    torch.testing.assert_close(x.grad, upstream, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_cast_boundary_and_all_gradients_match_literal_projection(ratio):
    torch.manual_seed(73)
    config = tiny_config(torch_dtype="bfloat16")
    module = DeepseekV41Compressor(config, ratio)
    # Strict FSDP storage can promote the ratio-1 projection while its compute
    # must follow the BF16 activation. Ordinary norm weight stays BF16.
    module.wkv.float()
    reference = {name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()}
    x = torch.randn(2, 5, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    if ratio == 1:
        latent = F.linear(reference_x, reference["wkv.weight"].bfloat16())
    else:
        chunks = reference_x[:, :4].float().reshape(2, 2, 2, config.hidden_size)
        kv = F.linear(chunks, reference["wkv.weight"])
        logits = F.linear(chunks, reference["wgate.weight"])
        latent = (kv * logits.softmax(2)).sum(2).bfloat16()
    expected = F.rms_norm(
        latent.float(), (config.head_dim,), reference["norm.weight"].float(), config.rms_norm_eps
    ).bfloat16()
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0.015625, rtol=0.015625)
    if ratio == 2:
        assert not x.grad[:, -1].any()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference[name].grad, atol=0.0625, rtol=0.02)


def test_indexer_uses_bf16_scores_and_returns_sorted_positions_with_empty_queries():
    torch.manual_seed(83)
    config = tiny_config(torch_dtype="bfloat16", index_topk=3, kv_cache_fake_quant=False)
    module = DeepseekV41Indexer(config, 5, tiny_backend())
    x = torch.randn(2, 4, config.hidden_size, dtype=torch.bfloat16)
    qr = torch.randn(2, 4, config.q_lora_rank, dtype=torch.bfloat16)
    keys = torch.randn(2, 7, config.index_head_dim, dtype=torch.bfloat16)
    # Identity RoPE isolates the indexer's projection, BF16 score boundaries,
    # candidate visibility, position ordering, and frozen ownership.
    cos, sin = torch.ones(2, 4, config.qk_rope_head_dim), torch.zeros(2, 4, config.qk_rope_head_dim)
    allowed = torch.ones(2, 4, 7, dtype=torch.bool)
    allowed[:, 0] = False
    candidates = torch.zeros_like(allowed)
    candidates[:, :, [0, 2, 4, 6]] = True
    candidates[:, 1, [4, 6]] = False
    state = DeepseekV41SharedState(compress_ratio=1, index_k=keys, allowed=allowed, candidates=candidates)
    actual, returned_candidates = module(x, qr, cos, sin, state)
    q = F.linear(qr, module.wq_b.weight).reshape(2, 4, config.index_n_heads, config.index_head_dim)
    weights = F.linear(x, module.weights_proj.weight) * (config.index_head_dim**-0.5 * config.index_n_heads**-0.5)
    scores = torch.einsum("bshd,btd->bsht", q, keys).relu()
    scores = (scores * weights[..., None]).sum(2)
    scores = scores.masked_fill(~(allowed & candidates), -torch.inf)
    selected = scores.topk(3, sorted=False).indices.sort(-1).values
    expected = torch.where(torch.isfinite(scores.gather(-1, selected)), selected, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert returned_candidates is candidates
    assert not actual[:, 0].ge(0).any()
    assert all(not parameter.requires_grad for parameter in module.parameters())
