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
    calls = {name: 0 for name in ("project_core", "collapse_core", "expand_core", "norm_core")}
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
            expected_compiled.update(("project_core", "collapse_core", "expand_core"))
        if compile_norm:
            expected_compiled.add("norm_core")
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
        assert calls["expand_core"] == (8 if compile_hc else 0)
        assert calls["norm_core"] == (9 if compile_norm else 0)
    finally:
        optimized_kernels.hc_cores.cache_clear()
        optimized_kernels.rms_norm_core.cache_clear()
