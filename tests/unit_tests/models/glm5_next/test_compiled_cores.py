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

"""CPU contracts, dispatch and precision boundaries for GLM compiled cores."""

import inspect
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v41.layers import _hc_expand_core as dsv4_expand
from nemo_automodel.components.models.deepseek_v41.layers import _rms_norm_core as dsv4_norm
from nemo_automodel.components.models.glm5_next import optimized_kernels
from nemo_automodel.components.models.glm5_next.layers import (
    Glm5NextHyperConnection,
    Glm5NextRMSNorm,
)
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
from nemo_automodel.components.models.glm5_next.optimized_kernels import (
    collapse_core,
    expand_core,
    norm_core,
    project_core,
)
from nemo_automodel.components.moe import experts as moe_experts
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("core", ["project", "collapse", "expand", "norm"])
def test_eager_core_matches_glm_arithmetic_and_vjp(dtype: torch.dtype, core: str) -> None:
    torch.manual_seed(613)
    config = tiny_glm5_next_config().text_config
    config.torch_dtype = dtype
    hc = Glm5NextHyperConnection(config)
    hc.init_weights(torch.device("cpu"), 0.1)
    hidden = torch.randn(1, 13, 2, 16, dtype=dtype).requires_grad_()
    if core == "project":
        args = (hidden, hc.fn)
        expected = F.linear(hc.input_norm(hidden.flatten(start_dim=2).float()), hc.fn.float())
        actual = project_core(*args, config.rms_norm_eps)
    elif core == "collapse":
        pre = torch.rand(1, 13, 2, requires_grad=True)
        args = (hidden, pre)
        expected = (pre.unsqueeze(-1) * hidden).sum(dim=2).to(dtype)
        actual = collapse_core(*args)
    elif core == "expand":
        update = torch.randn(1, 13, 16, dtype=dtype, requires_grad=True)
        post = torch.rand(1, 13, 2, requires_grad=True)
        comb = torch.rand(1, 13, 2, 2, requires_grad=True)
        args = (update, hidden, post, comb)
        expected = post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden
        )
        actual = expand_core(*args)
    else:
        norm = Glm5NextRMSNorm(16, config.rms_norm_eps, dtype)
        with torch.no_grad():
            norm.weight.uniform_(0.7, 1.3)
        args = (hidden, norm.weight)
        values = hidden.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + config.rms_norm_eps)
        expected = norm.weight * values.to(dtype)
        actual = norm_core(*args, config.rms_norm_eps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn(*actual.shape[:-2], actual.shape[-1], actual.shape[-2], dtype=actual.dtype).transpose(-1, -2)
    assert not upstream.is_contiguous()
    expected_grad = torch.autograd.grad(expected, args, upstream)
    actual_grad = torch.autograd.grad(actual, args, upstream)
    for observed, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def test_dsv4_expand_is_not_the_glm_bf16_contract() -> None:
    torch.manual_seed(617)
    residual = torch.randn(1, 31, 4, 16, dtype=torch.bfloat16)
    update = torch.randn(1, 31, 16, dtype=torch.bfloat16)
    post = torch.rand(1, 31, 4)
    comb = torch.rand(1, 31, 4, 4)
    assert not torch.equal(expand_core(update, residual, post, comb), dsv4_expand(update, residual, post, comb))


def test_dsv4_norm_is_not_the_glm_bf16_contract() -> None:
    torch.manual_seed(619)
    hidden = torch.randn(1, 31, 16, dtype=torch.bfloat16)
    weight = torch.rand(16, dtype=torch.bfloat16)
    assert not torch.equal(norm_core(hidden, weight, 1e-6), dsv4_norm(hidden, weight, 1e-6))


@pytest.mark.parametrize("compile_hc,compile_norm", [(True, False), (False, True), (True, True)])
def test_compile_dispatch_is_instance_local_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch, compile_hc: bool, compile_norm: bool
) -> None:
    """Mock compilation only: verify routing, state layout and the complete CPU path.

    Actual CUDA compilation/parity is validated separately by the module
    artifact; returning the eager callable here is not CUDA-kernel evidence.
    """
    optimized_kernels.hc_cores.cache_clear()
    optimized_kernels.rms_norm_core.cache_clear()
    optimized_kernels.rms_norm_gated_core.cache_clear()
    calls = {
        name: 0
        for name in ("project_core", "collapse_core", "_expand_core_four_streams", "norm_core", "gated_norm_core")
    }
    compiled_names = []

    def fake_compile(function, **kwargs):
        assert kwargs["dynamic"] is False
        assert kwargs["options"]["emulate_precision_casts"] is True
        assert kwargs["options"]["eager_numerics.division_rounding"] is True
        compiled_names.append(function.__name__)

        def wrapped(*args):
            calls[function.__name__] += 1
            return function(*args)

        return wrapped

    # This MoE activation was compiled at import time, before the fake factory.
    # Preserve its complete Python arithmetic without invoking an unrelated
    # compiler in this GLM dispatch test, including on a cold compiler cache.
    monkeypatch.setattr(moe_experts, "swiglu_clamped_deepep", inspect.unwrap(moe_experts.swiglu_clamped_deepep))
    monkeypatch.setattr(torch, "compile", fake_compile)
    try:
        torch.manual_seed(631)
        eager = Glm5NextForConditionalGeneration(tiny_glm5_next_config(), backend=tiny_backend())
        eager.initialize_weights(torch.device("cpu"), dtype=torch.float32)
        optimized = Glm5NextForConditionalGeneration(
            tiny_glm5_next_config(),
            backend=replace(tiny_backend(), compile_hc=compile_hc, compile_norm=compile_norm),
        )
        assert eager.state_dict().keys() == optimized.state_dict().keys()
        optimized.load_state_dict(eager.state_dict())
        for layer in optimized.language_model.layers.values():
            assert layer.attn_hc._fp32_params.sinkhorn_backend == "torch"
            assert layer.attn_hc.base.dtype == torch.float32
            assert layer.attn_hc.scale.dtype == torch.float32
        expected_compiled = set()
        if compile_hc:
            expected_compiled.update(("project_core", "collapse_core", "_expand_core_four_streams"))
        if compile_norm:
            expected_compiled.update(("norm_core", "gated_norm_core"))
        assert set(compiled_names) == expected_compiled
        assert len(compiled_names) == len(expected_compiled)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        expected = eager(input_ids=input_ids).logits
        expected.square().mean().backward()
        # Constructing another instance with opt-in compilation must not affect
        # existing eager modules or flip any module-level dispatch globally.
        assert not any(calls.values())
        actual = optimized(input_ids=input_ids).logits
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (_, observed), (_, reference) in zip(optimized.named_parameters(), eager.named_parameters()):
            if reference.grad is None:
                assert observed.grad is None
            else:
                torch.testing.assert_close(observed.grad, reference.grad, rtol=0, atol=0)
        assert calls["project_core"] == (8 if compile_hc else 0)
        assert calls["collapse_core"] == (8 if compile_hc else 0)
        assert calls["_expand_core_four_streams"] == (8 if compile_hc else 0)
        assert calls["norm_core"] == (9 if compile_norm else 0)
        assert calls["gated_norm_core"] == (3 if compile_norm else 0)
    finally:
        optimized_kernels.hc_cores.cache_clear()
        optimized_kernels.rms_norm_core.cache_clear()
        optimized_kernels.rms_norm_gated_core.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 17, 4, 128), (1, 4096, 64, 128)])
def test_compiled_gated_norm_cuda_forward_and_backward(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    """Exercise the actual compiled core against independent eager arithmetic."""
    torch.manual_seed(641)
    hidden = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    gate = (torch.randn(shape, device="cuda", dtype=dtype) * 4).requires_grad_()
    weight = (torch.rand(shape[-1], device="cuda", dtype=dtype) + 0.5).requires_grad_()
    states = hidden.float()
    normalized = (states * torch.rsqrt(states.square().mean(-1, keepdim=True) + 1e-5)).to(dtype)
    expected = (weight * normalized) * torch.sigmoid(gate.float()).to(dtype)
    actual = optimized_kernels.rms_norm_gated_core(True)(hidden, gate, weight, 1e-5)
    upstream = torch.randn_like(hidden)
    expected_grads = torch.autograd.grad(expected, (hidden, gate, weight), upstream)
    actual_grads = torch.autograd.grad(actual, (hidden, gate, weight), upstream)
    pairs = [(actual, expected), *zip(actual_grads, expected_grads)]
    for observed, reference in pairs:
        assert torch.isfinite(observed).all()
        if dtype == torch.float32:
            torch.testing.assert_close(observed, reference, rtol=1e-4, atol=2e-3)
        else:
            relative_rms = (
                observed.float() - reference.float()
            ).square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-7)
            assert relative_rms < 0.005
            torch.testing.assert_close(observed, reference, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("streams", [2, 4])
def test_four_stream_expand_preserves_output_and_all_input_gradients(dtype: torch.dtype, streams: int) -> None:
    """Check the asymmetric stream mixing and all VJPs, including fallback cases."""
    torch.manual_seed(647)
    residual = torch.randn(2, 7, streams, 32, dtype=dtype).requires_grad_()
    update = torch.randn(2, 7, 32, dtype=dtype).requires_grad_()
    post = torch.rand(2, 7, streams, requires_grad=True)
    comb = torch.rand(2, 7, streams, streams, requires_grad=True)
    args = (update, residual, post, comb)
    # Independent unchanged GLM arithmetic; asymmetric coefficients expose a
    # mistaken source/destination transpose. The oracle is not the new helper.
    mixed = torch.matmul(comb.to(dtype).transpose(-1, -2), residual)
    expected = post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + mixed
    actual = optimized_kernels._expand_core_four_streams(*args)
    upstream = torch.randn(2, 7, 32, streams, dtype=dtype).transpose(-1, -2)
    assert not upstream.is_contiguous()
    pairs = [
        (actual, expected),
        *zip(
            torch.autograd.grad(actual, args, upstream),
            torch.autograd.grad(expected, args, upstream),
        ),
    ]
    for observed, reference in pairs:
        assert torch.isfinite(observed).all()
        if dtype == torch.float32 or streams != 4:
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        else:
            scale = reference.double().square().mean().sqrt().clamp_min(1e-12)
            error = observed.double() - reference.double()
            assert error.square().mean().sqrt() / scale <= 0.002
            assert error.abs().max() / scale <= 0.05
    wrong_orientation = post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(comb.to(dtype), residual)
    assert not torch.allclose(wrong_orientation, expected, rtol=0.002, atol=0.002)


def _assert_matches(observed: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    """Compare a compiled result with its eager oracle by RMS and worst-case error relative to the oracle scale.

    Args:
        observed: Compiled output or gradient of any shape.
        reference: Eager oracle with the same shape as ``observed``.
        name: Label used in assertion messages.
    """
    assert observed.shape == reference.shape and observed.dtype == reference.dtype, name
    assert torch.isfinite(observed).all(), name
    scale = reference.double().square().mean().sqrt().clamp_min(1e-12)
    error = observed.double() - reference.double()
    if reference.dtype == torch.float32:
        assert error.abs().max() / scale <= 1e-4, name
    else:
        assert error.square().mean().sqrt() / scale <= 0.005, name
        assert error.abs().max() / scale <= 0.05, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compilation requires a GPU")
def test_compiled_mhc_cores_cuda_match_eager_arithmetic_and_vjps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real compiled projection, collapse and four-stream expansion on CUDA."""
    # The FP32 tolerances assume true FP32 GEMMs; some containers enable TF32 by default.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch._dynamo.reset()
    torch.manual_seed(653)
    streams, hidden = 4, 256
    project, collapse, expand = optimized_kernels.hc_cores(True)
    shape = (2, 33, streams, hidden)
    hidden_streams = torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
    fn = (torch.randn(streams * (2 + streams), streams * hidden, device="cuda") * 0.02).requires_grad_()
    pre = torch.rand(2, 33, streams, device="cuda").requires_grad_()
    update = torch.randn(2, 33, hidden, device="cuda", dtype=torch.bfloat16).requires_grad_()
    post = torch.rand(2, 33, streams, device="cuda").requires_grad_()
    # Asymmetric source -> destination mixing, so a transposed implementation fails.
    comb = (torch.rand(2, 33, streams, streams, device="cuda") * torch.arange(1, streams + 1, device="cuda")).detach()
    comb.requires_grad_()

    def project_oracle(states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """FP32 RMS-normalized flattened streams [2, 33, streams * hidden] times weight^T."""
        flat = states.flatten(start_dim=2).float()
        return F.linear(flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5), weight.float())

    def collapse_oracle(states: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """Pre-weighted stream sum [2, 33, hidden], rounded once to the stream dtype."""
        return (coefficients.unsqueeze(-1) * states).sum(dim=2).to(states.dtype)

    def expand_oracle(
        sublayer: torch.Tensor, residual: torch.Tensor, post_w: torch.Tensor, comb_w: torch.Tensor
    ) -> torch.Tensor:
        """Streams [2, 33, streams, hidden]: post-weighted update plus comb^T-mixed residual streams."""
        dtype = residual.dtype
        return post_w.to(dtype).unsqueeze(-1) * sublayer.unsqueeze(-2) + torch.matmul(
            comb_w.to(dtype).transpose(-1, -2), residual
        )

    cases = [
        ("project", project, project_oracle, (hidden_streams, fn), (1e-5,)),
        ("collapse", collapse, collapse_oracle, (hidden_streams, pre), ()),
        ("expand", expand, expand_oracle, (update, hidden_streams, post, comb), ()),
    ]
    for name, compiled, oracle, inputs, extra in cases:
        actual = compiled(*inputs, *extra)
        expected = oracle(*inputs)
        # A fixed noncontiguous upstream gradient, as produced by transposed consumers.
        upstream = torch.randn(*expected.shape[:-2], expected.shape[-1], expected.shape[-2], device="cuda")
        upstream = upstream.to(expected.dtype).transpose(-1, -2)
        assert not upstream.is_contiguous()
        _assert_matches(actual, expected, name)
        for index, (observed, reference) in enumerate(
            zip(torch.autograd.grad(actual, inputs, upstream), torch.autograd.grad(expected, inputs, upstream))
        ):
            _assert_matches(observed, reference, f"{name} d_input{index}")
