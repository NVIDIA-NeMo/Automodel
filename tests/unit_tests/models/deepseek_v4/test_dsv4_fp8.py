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

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.fp8 import quantize_dsv4_indexer, quantize_dsv4_kv
from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import HAS_TILELANG
from nemo_automodel.components.models.deepseek_v4.kernels.tilelang_hyperconnection import (
    HAS_DEEP_GEMM,
    _torch_prepare,
    exact_mhc_post,
    exact_mhc_prepare,
)
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Attention, _apply_q_head_norm_rope
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    dsv4_fp8_sparse_attention,
    dsv4_indexer_scores,
    dsv4_sparse_attention,
    indexer_scores_torch,
)


def test_fp8_ds_mla_torch_oracle_uses_vllm_group64_pow2_scales():
    kv = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
    kv[..., 64:128] = 448.0
    kv[..., 128:192] = 224.0
    kv[..., 192:256] = 500.0
    kv[..., 448:] = torch.arange(64, dtype=torch.bfloat16)

    packed = quantize_dsv4_kv(kv, backend="torch")

    torch.testing.assert_close(
        packed.scales[0, 0],
        torch.tensor([105, 127, 126, 128, 105, 105, 105, 0], dtype=torch.uint8),
        atol=0,
        rtol=0,
    )
    assert packed.storage_nbytes == 584
    assert packed.vllm_token_data().shape == (1, 1, 576)
    torch.testing.assert_close(packed.rope, kv[..., 448:], atol=0, rtol=0)


def test_fp8_ds_mla_torch_oracle_preserves_empty_sequence_shape():
    packed = quantize_dsv4_kv(torch.empty(2, 0, 512, dtype=torch.bfloat16), backend="torch")

    assert packed.nope.shape == (2, 0, 448)
    assert packed.rope.shape == (2, 0, 64)
    assert packed.scales.shape == (2, 0, 8)
    assert packed.storage_nbytes == 0


def test_fp8_indexer_torch_oracle_uses_vllm_per_row_pow2_scales():
    activations = torch.zeros(2, 128, dtype=torch.bfloat16)
    activations[0, 0] = 448.0
    activations[1, 0] = 224.0

    packed = quantize_dsv4_indexer(activations, backend="torch")

    torch.testing.assert_close(packed.scales[:, 0], torch.tensor([1.0, 0.5]), atol=0, rtol=0)
    torch.testing.assert_close(packed.values[:, 0].float(), torch.tensor([448.0, 448.0]), atol=0, rtol=0)
    torch.testing.assert_close(packed.dequantize(), activations, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("backend", "match"),
    (
        (BackendConfig(attn="sdpa", linear="torch", experts="torch_mm"), "backend.attn='tilelang'"),
        (
            BackendConfig(attn="tilelang", linear="torch", experts="torch_mm"),
            "requires linear='te'",
        ),
    ),
)
def test_fp8_ds_mla_rejects_partial_precision_configuration(backend, match):
    config = DeepseekV4Config(
        num_hidden_layers=1,
        num_attention_heads=16,
        head_dim=512,
        qk_rope_head_dim=64,
        compress_ratios=[0],
        kv_cache_dtype="fp8_ds_mla",
    )
    with pytest.raises(ValueError, match=match):
        DeepseekV4Attention(config, layer_idx=0, backend=backend)


_FP8_TILELANG = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TILELANG),
    reason="DSV4 FP8 KV kernels require CUDA TileLang",
)
_EXACT_HC = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TILELANG and HAS_DEEP_GEMM),
    reason="Exact DSV4 HyperConnection requires CUDA TileLang and DeepGEMM",
)


@_EXACT_HC
def test_exact_hyperconnection_forward_and_backward_match_reference():
    torch.manual_seed(59)
    tokens, hc_mult, hidden = 17, 4, 256
    residual = torch.randn(
        tokens,
        hc_mult,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    fn = (
        torch.randn(
            24,
            hc_mult * hidden,
            device="cuda",
            dtype=torch.float32,
        )
        * 1.0e-3
    ).requires_grad_(True)
    scale = torch.tensor([0.9, 1.1, 0.8], device="cuda", requires_grad=True)
    base = (torch.randn(24, device="cuda", dtype=torch.float32) * 0.01).requires_grad_(True)

    exact = exact_mhc_prepare(
        residual,
        fn,
        scale,
        base,
        norm_eps=1.0e-6,
        pre_eps=1.0e-6,
        sinkhorn_eps=1.0e-6,
        post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
    reference = _torch_prepare(
        residual,
        fn,
        scale,
        base,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        2.0,
        20,
    )
    torch.testing.assert_close(exact[0], reference[0], atol=1.0e-2, rtol=1.0e-2)
    torch.testing.assert_close(exact[1], reference[1], atol=1.0e-4, rtol=1.0e-4)
    torch.testing.assert_close(exact[2], reference[2], atol=1.0e-4, rtol=1.0e-4)

    grad_outputs = tuple(torch.randn_like(output) for output in exact)
    exact_grads = torch.autograd.grad(
        exact,
        (residual, fn, scale, base),
        grad_outputs,
        retain_graph=True,
    )
    reference_grads = torch.autograd.grad(
        reference,
        (residual, fn, scale, base),
        grad_outputs,
    )
    for actual, expected in zip(exact_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    layer_output = torch.randn(
        tokens,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    post_output = exact_mhc_post(residual, layer_output, exact[1], exact[2])
    post_output.float().square().mean().backward()
    assert post_output.shape == residual.shape
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert layer_output.grad is not None and torch.isfinite(layer_output.grad).all()


@_FP8_TILELANG
def test_tilelang_q_head_norm_rope_matches_fp32_reference_and_backward():
    torch.manual_seed(53)
    batch, heads, seq_len, head_dim, rope_dim = 1, 5, 11, 512, 64
    q = (torch.randn(batch, heads, seq_len, head_dim, device="cuda") * 0.7).to(torch.bfloat16)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32) / rope_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).unsqueeze(0)
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).unsqueeze(0)

    q_actual = q.detach().clone().requires_grad_(True)
    q_expected = q.detach().clone().requires_grad_(True)
    actual = _apply_q_head_norm_rope(
        q_actual,
        cos,
        sin,
        rope_dim,
        1.0e-6,
        use_tilelang=True,
    )
    expected = _apply_q_head_norm_rope(q_expected, cos, sin, rope_dim, 1.0e-6)

    # The torch reference uses a different 512-way reduction tree, so values
    # on a BF16 rounding boundary can differ by one ULP.  The dedicated vLLM
    # parity test validates bitwise equality to its fused CUDA kernel.
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0)
    assert torch.count_nonzero(actual != expected) < actual.numel() // 1000

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(q_actual.grad, q_expected.grad, atol=0.015625, rtol=0)


@_FP8_TILELANG
def test_tilelang_fp8_ds_mla_quantizer_is_bitwise_equal_to_torch_oracle():
    torch.manual_seed(11)
    kv = (torch.randn(2, 5, 512, device="cuda", dtype=torch.bfloat16) * 0.7).contiguous()
    expected = quantize_dsv4_kv(kv, backend="torch")
    actual = quantize_dsv4_kv(kv, backend="tilelang")

    assert torch.equal(actual.nope, expected.nope)
    assert torch.equal(actual.rope, expected.rope)
    assert torch.equal(actual.scales, expected.scales)


@_FP8_TILELANG
def test_tilelang_fp8_indexer_matches_vllm_oracle_and_dequantized_scores():
    torch.manual_seed(29)
    rows = (torch.randn(17, 128, device="cuda", dtype=torch.bfloat16) * 0.7).contiguous()
    expected_rows = quantize_dsv4_indexer(rows, backend="torch")
    actual_rows = quantize_dsv4_indexer(rows, backend="tilelang")
    assert torch.equal(actual_rows.values, expected_rows.values)
    assert torch.equal(actual_rows.scales, expected_rows.scales)

    batch, seq_len, heads, kv_len = 1, 12, 64, 3
    q = (torch.randn(batch, seq_len, heads, 128, device="cuda", dtype=torch.bfloat16) * 0.4).contiguous()
    k = (torch.randn(batch, kv_len, 128, device="cuda", dtype=torch.bfloat16) * 0.4).contiguous()
    weights = torch.randn(batch, seq_len, heads, device="cuda", dtype=torch.float32) / heads**0.5
    q_fp8 = quantize_dsv4_indexer(q, backend="tilelang")
    k_fp8 = quantize_dsv4_indexer(k, backend="tilelang")

    actual = dsv4_indexer_scores(
        q_fp8,
        k_fp8,
        weights,
        compress_ratio=4,
        softmax_scale=128**-0.5,
        backend="tilelang",
    )
    expected = indexer_scores_torch(q_fp8.dequantize(), k_fp8.dequantize(), weights, 128**-0.5)
    valid_end = ((torch.arange(seq_len, device="cuda") + 1) // 4).clamp(max=kv_len)
    valid = torch.arange(kv_len, device="cuda").view(1, 1, kv_len) < valid_end.view(1, seq_len, 1)
    expected = torch.where(valid, expected, torch.full_like(expected, float("-inf")))

    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=2e-3, rtol=2e-3)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    for row in range(3, seq_len):
        end = int(valid_end[row])
        assert torch.equal(
            actual[0, row, :end].topk(min(2, end)).indices,
            expected[0, row, :end].topk(min(2, end)).indices,
        )


@_FP8_TILELANG
def test_tilelang_fp8_empty_kv_skips_zero_grid_and_has_zero_attention_gradients():
    q = torch.randn(2, 3, 16, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.empty(2, 0, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.randn(16, device="cuda", requires_grad=True)
    topk = torch.full((2, 3, 8), -1, device="cuda", dtype=torch.int64)

    packed = quantize_dsv4_kv(kv, backend="tilelang")
    output = dsv4_fp8_sparse_attention(q, packed, sinks, topk, 512**-0.5)
    output.sum().backward()

    assert output.shape == q.shape
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(q.grad) == 0
    assert kv.grad.shape == kv.shape
    assert torch.count_nonzero(sinks.grad) == 0


@_FP8_TILELANG
def test_tilelang_fp8_sparse_attention_matches_fake_quant_ste_forward_backward():
    torch.manual_seed(21)
    base_q = torch.randn(1, 2, 16, 512, device="cuda", dtype=torch.bfloat16)
    base_kv = torch.randn(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    base_sinks = torch.randn(16, device="cuda")
    topk = torch.tensor(
        [[[0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6, -1, -1, -1, -1]]],
        device="cuda",
    )
    grad = torch.randn_like(base_q)

    q = base_q.clone().requires_grad_()
    kv = base_kv.clone().requires_grad_()
    sinks = base_sinks.clone().requires_grad_()
    packed = quantize_dsv4_kv(kv, backend="tilelang")
    actual = dsv4_fp8_sparse_attention(q, packed, sinks, topk, 512**-0.5)
    (actual * grad).sum().backward()

    q_ref = base_q.clone().requires_grad_()
    kv_ref = base_kv.clone().requires_grad_()
    sinks_ref = base_sinks.clone().requires_grad_()
    packed_ref = quantize_dsv4_kv(kv_ref, backend="torch")
    fake_quant_kv = kv_ref + (packed_ref.dequantize() - kv_ref).detach()
    expected = dsv4_sparse_attention(
        q_ref,
        fake_quant_kv,
        sinks_ref,
        topk,
        512**-0.5,
        backend="tilelang",
    )
    (expected * grad).sum().backward()

    for actual_tensor, expected_tensor in (
        (actual, expected),
        (q.grad, q_ref.grad),
        (kv.grad, kv_ref.grad),
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
    # The FP8 forward follows FlashMLA's reciprocal/multiply epilogue exactly,
    # while the vendored fake-quant reference uses per-element division.  That
    # can move the FP32 sink gradient by one rounding step even when output,
    # dQ, and dKV are bitwise equal.
    torch.testing.assert_close(sinks.grad, sinks_ref.grad, atol=5.0e-7, rtol=2.0e-6)
