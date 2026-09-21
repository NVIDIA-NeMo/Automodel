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
import torch.nn as nn

import nemo_automodel.components.models.kimi_k3.situ as situ_mod
from nemo_automodel.components.models.kimi_k3.attn_res_triton import (
    HAVE_TRITON,
    MAX_ENTRIES,
    attn_res_bwd_triton,
    attn_res_fwd_triton,
)
from nemo_automodel.components.models.kimi_k3.model import KimiRMSNorm, _apply_attn_res
from nemo_automodel.components.models.kimi_k3.situ import (
    _attn_res_core,
    _attn_res_triton_applies,
    _AttnResTritonFunction,
    _enable_attn_res_triton,
)

needs_gpu_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="Triton attn-res kernels need a GPU and Triton"
)


def _reference(prefix_sum, block_residual, norm_weight, proj_weight, eps):
    """The eager fp32 chain (situ._attn_res_core) on the concatenated entries."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    return _attn_res_core(values, norm_weight, proj_weight, eps, values.dtype)


def _inputs(tokens, blocks, hidden, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ps = torch.randn(tokens, hidden, generator=g).to(device=device, dtype=dtype).requires_grad_(True)
    br = torch.randn(tokens, blocks, hidden, generator=g).to(device=device, dtype=dtype).requires_grad_(True)
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=g)).to(device=device, dtype=dtype).requires_grad_(True)
    pw = (0.05 * torch.randn(hidden, generator=g)).to(device=device, dtype=dtype).requires_grad_(True)
    return ps, br, nw, pw


@pytest.fixture
def triton_enabled(monkeypatch):
    monkeypatch.setattr(situ_mod, "_ATTN_RES_TRITON_ENABLED", True)


def test_applies_requires_enabled_cuda_and_shapes():
    ps = torch.randn(4, 8)
    br = torch.randn(4, 2, 8)
    assert not _attn_res_triton_applies(ps, br)  # disabled by default
    saved = situ_mod._ATTN_RES_TRITON_ENABLED
    try:
        situ_mod._ATTN_RES_TRITON_ENABLED = True
        assert not _attn_res_triton_applies(ps, br)  # CPU tensors never route to Triton
        if torch.cuda.is_available():
            ps_c, br_c = ps.cuda(), br.cuda()
            assert _attn_res_triton_applies(ps_c, br_c)
            assert not _attn_res_triton_applies(ps_c, br_c[:, :, :4])  # hidden mismatch
            assert not _attn_res_triton_applies(ps_c[:2], br_c)  # token mismatch
            assert not _attn_res_triton_applies(ps_c, torch.randn(4, MAX_ENTRIES + 1, 8, device="cuda"))
            assert not _attn_res_triton_applies(ps_c[:0], br_c[:0])
    finally:
        situ_mod._ATTN_RES_TRITON_ENABLED = saved


def test_enable_is_noop_without_cuda_or_triton(monkeypatch):
    monkeypatch.setattr(situ_mod, "_ATTN_RES_TRITON_ENABLED", False)
    monkeypatch.setattr(situ_mod, "_HAVE_SITU_TRITON", False)
    _enable_attn_res_triton()
    assert situ_mod._ATTN_RES_TRITON_ENABLED is False


def test_enable_sets_flag_once(monkeypatch):
    monkeypatch.setattr(situ_mod, "_ATTN_RES_TRITON_ENABLED", False)
    monkeypatch.setattr(situ_mod, "_HAVE_SITU_TRITON", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _enable_attn_res_triton()
    assert situ_mod._ATTN_RES_TRITON_ENABLED is True
    _enable_attn_res_triton()  # idempotent: the second call returns early and keeps the flag
    assert situ_mod._ATTN_RES_TRITON_ENABLED is True


@needs_gpu_triton
@pytest.mark.parametrize(
    "tokens,blocks,hidden",
    [(64, 3, 128), (37, 8, 200), (5, 0, 32), (130, 1, 7168), (16, 16, 96)],
)
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton attention-residual kernels on a cold cache (up to 9.5 s measured on GB200).",
)
def test_fp32_forward_and_grads_match_reference(tokens, blocks, hidden):
    """fp32 in/out: the kernels reproduce the eager chain to fp32 accumulation-order noise."""
    ps, br, nw, pw = _inputs(tokens, blocks, hidden, torch.float32, "cuda")
    ref_in = [t.detach().clone().requires_grad_(True) for t in (ps, br, nw, pw)]
    eps = 1e-6

    out = _AttnResTritonFunction.apply(ps, br, nw, pw, eps)
    ref = _reference(*ref_in, eps)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    for got, want in zip((ps, br), ref_in[:2]):
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-4, atol=1e-4)
    # The weight gradients reduce over tokens in a different fp32 order (per-program partials
    # summed in torch vs autograd's single reduction): tolerance scaled to their magnitude.
    for got, want in zip((nw, pw), ref_in[2:]):
        scale = want.grad.abs().max().clamp_min(1e-6).item()
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-3, atol=1e-4 * scale)


@needs_gpu_triton
@pytest.mark.parametrize("tokens,blocks,hidden", [(64, 3, 128), (37, 8, 7168), (9, 1, 64)])
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton attention-residual kernels on a cold cache (up to 9.5 s measured on GB200).",
)
def test_bf16_forward_within_one_ulp_and_grads_close(tokens, blocks, hidden):
    """bf16 in/out (the training dtype): forward within one bf16 ulp of the eager chain, grads close."""
    ps, br, nw, pw = _inputs(tokens, blocks, hidden, torch.bfloat16, "cuda")
    ref_in = [t.detach().clone().requires_grad_(True) for t in (ps, br, nw, pw)]
    eps = 1e-6

    out = _AttnResTritonFunction.apply(ps, br, nw, pw, eps)
    ref = _reference(*ref_in, eps)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=8e-3, atol=1e-5)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    torch.testing.assert_close(ps.grad.float(), ref_in[0].grad.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(br.grad.float(), ref_in[1].grad.float(), rtol=2e-2, atol=2e-2)
    # The weight gradients are reductions over tokens: compare in fp32 with a relative tolerance
    # on their scale, as the bf16 chain rounds them once at the end too.
    for got, want in ((nw.grad, ref_in[2].grad), (pw.grad, ref_in[3].grad)):
        scale = want.float().abs().max().clamp_min(1e-6)
        assert ((got.float() - want.float()).abs().max() / scale).item() < 2e-2


@needs_gpu_triton
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton attention-residual kernels on a cold cache (up to 9.5 s measured on GB200).",
)
def test_raw_kernels_accept_strided_entries_and_reject_bad_shapes():
    ps, br, nw, pw = _inputs(12, 4, 64, torch.float32, "cuda")
    ps_d, br_d, nw_d, pw_d = (t.detach() for t in (ps, br, nw, pw))
    # Entries selected out of a wider buffer: strided token / entry dims, contiguous last dim.
    wide = torch.randn(12, 6, 64, device="cuda")
    wide[:, 1:5] = br_d
    out_strided, stats = attn_res_fwd_triton(ps_d, wide[:, 1:5], nw_d, pw_d, 1e-6)
    out_dense, _ = attn_res_fwd_triton(ps_d, br_d, nw_d, pw_d, 1e-6)
    torch.testing.assert_close(out_strided, out_dense)
    assert stats.shape == (3, 12, 5)
    torch.testing.assert_close(stats[0].sum(-1), torch.ones(12, device="cuda"), rtol=1e-5, atol=1e-5)

    g = torch.randn_like(out_dense)
    d_ps, d_br, d_sw = attn_res_bwd_triton(ps_d, br_d, nw_d, pw_d, g, stats)
    assert d_ps.shape == ps_d.shape and d_br.shape == br_d.shape and d_sw.shape == (64,)

    with pytest.raises(ValueError):
        attn_res_fwd_triton(ps_d, br_d[:6], nw_d, pw_d, 1e-6)
    with pytest.raises(ValueError):
        attn_res_fwd_triton(ps_d, torch.randn(12, MAX_ENTRIES + 1, 64, device="cuda"), nw_d, pw_d, 1e-6)
    with pytest.raises(ValueError):
        attn_res_fwd_triton(ps_d.t().contiguous().t(), br_d, nw_d, pw_d, 1e-6)


@needs_gpu_triton
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton attention-residual kernels on a cold cache (up to 9.5 s measured on GB200).",
)
def test_apply_attn_res_routes_to_triton_when_enabled(triton_enabled, monkeypatch):
    """_apply_attn_res takes the Triton path for CUDA inputs and matches the eager chain."""
    hidden = 96
    ps, br, _, _ = _inputs(40, 2, hidden, torch.bfloat16, "cuda")
    proj = nn.Linear(hidden, 1, bias=False).to(device="cuda", dtype=torch.bfloat16)
    norm = KimiRMSNorm(hidden, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(hidden))

    calls = []
    real_apply = _AttnResTritonFunction.apply
    monkeypatch.setattr(_AttnResTritonFunction, "apply", staticmethod(lambda *a: calls.append(1) or real_apply(*a)))
    out = _apply_attn_res(ps, br, proj, norm)
    assert calls == [1]

    monkeypatch.setattr(situ_mod, "_ATTN_RES_TRITON_ENABLED", False)
    ref = _apply_attn_res(ps.detach(), br.detach(), proj, norm)
    torch.testing.assert_close(out, ref, rtol=8e-3, atol=1e-5)

    out.float().square().sum().backward()
    assert ps.grad is not None and br.grad is not None
    assert proj.weight.grad is not None and norm.weight.grad is not None
