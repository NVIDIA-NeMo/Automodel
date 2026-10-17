# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Pin the shared compiled SwiGLU boundary and compare default Beam experts with HF."""

import math

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.beam.model import beam_moe_config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.moe.megatron.moe_utils import weighted_bias_swiglu_impl

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def fp32_weighted_swiglu_reference(y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Oracle: HF's fp32 activation boundary with the routing weight folded in (``weights == 1`` is exactly HF's
    ``BeamExperts._apply_gate``). ``y`` = ``[..., 2 * inter]`` gate-first, ``weights`` = fp32 ``[..., 1]``."""
    gate, up = torch.chunk(y, 2, dim=-1)
    return (F.silu(gate.float()) * up.float() * weights).to(y.dtype)


def _inputs(n, inter, device, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    y = (torch.randn(n, 2 * inter, generator=g) * 2).to(device, dtype)
    w = torch.rand(n, 1, generator=g).to(device)
    grad = torch.randn(n, inter, generator=g).to(device, dtype)
    return y, w, grad


def _run(fn, y, w, grad):
    y, w = y.clone().requires_grad_(), w.clone().requires_grad_()
    out = fn(y, w)
    out.backward(grad)
    return out.detach(), y.grad, w.grad


def _bf16_ulp(ref: torch.Tensor) -> torch.Tensor:
    """One bf16 ulp at each reference value (8 significand bits)."""
    mag = ref.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


def _within_one_ulp_fraction(a, ref, max_fraction):
    diff = (a.float() - ref.float()).abs()
    assert (diff <= _bf16_ulp(ref) + 1e-30).all(), f"differences above one bf16 ulp: {(diff > _bf16_ulp(ref)).sum()}"
    frac = (a != ref).float().mean().item()
    assert frac <= max_fraction, f"{frac:.5%} of elements differ (declared bound {max_fraction:.5%})"
    return frac


class TestSharedExpertActivationBoundary:
    @pytest.fixture(autouse=True)
    def _eager_moe_swiglu(self):
        """Override the package-wide eager fixture: this class pins the compiled kernel itself."""
        yield

    @pytest.fixture(autouse=True)
    def _fresh_dynamo_state(self):
        """Keep this compiled-kernel test independent of earlier tests' Dynamo recompile counts."""
        torch.compiler.reset()
        yield

    # CUDA only: training runs the compiled kernel on GPU, and the CPU variant's cold Inductor C++ build alone took
    # ~25 s on H200 hosts, close to the 30 s cap for new tests.
    @pytest.mark.runtime_budget(
        30,
        reason="Compiles the shared weighted SwiGLU forward/backward kernels from a reset Dynamo state "
        "(cold Triton build).",
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="The compiled kernel runs on CUDA in training")
    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("n,inter", [(4096, 64), (1000, 24)])
    def test_compiled_kernel_meets_fp32_boundary(self, device, n, inter):
        """Declared bounds (measured 0.0004% / 0.001% on 16k x 512 and 16k x 2560): output and input-gradient
        elements differ from the fp32 oracle in <= 0.01% of positions and never by more than one bf16 ulp; the
        routing-weight gradient (an fp32 reduction) is within 1e-4 relative."""
        y, w, grad = _inputs(n, inter, device, torch.bfloat16, 0)
        ref = _run(fp32_weighted_swiglu_reference, y, w, grad)
        out = _run(weighted_bias_swiglu_impl, y, w, grad)
        _within_one_ulp_fraction(out[0], ref[0], 1e-4)
        _within_one_ulp_fraction(out[1], ref[1], 1e-4)
        torch.testing.assert_close(out[2], ref[2], rtol=1e-4, atol=1e-4 * ref[2].abs().max().item())


def _backend(experts):
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts=experts,
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=False,
    )


def _hf_experts_and_router(cfg, device, dtype, seed):
    from transformers.models.beam.modeling_beam import BeamExperts, BeamTopKRouter

    g = torch.Generator().manual_seed(seed)
    hf = BeamExperts(cfg)
    router = BeamTopKRouter(cfg)
    with torch.no_grad():
        hf.gate_up_proj.copy_(torch.randn(hf.gate_up_proj.shape, generator=g) * 0.05)
        hf.down_proj.copy_(torch.randn(hf.down_proj.shape, generator=g) * 0.05)
        router.weight.copy_(torch.randn(router.weight.shape, generator=g))
        router.e_score_correction_bias.copy_(torch.randn(cfg.num_experts, generator=g) * 0.1)
    return hf.to(device, dtype), router.to(device)


def _native_experts(cfg, hf, experts, dtype, device):
    moe_cfg = beam_moe_config(cfg, dtype)
    ge = GroupedExperts(moe_cfg, backend=_backend(experts)).to(device)
    with torch.no_grad():
        ge.gate_and_up_projs.copy_(hf.gate_up_proj.transpose(1, 2))
        ge.down_projs.copy_(hf.down_proj.transpose(1, 2))
    return ge


def _compare_experts(cfg, experts, dtype, device, n_tokens=192):
    hf, router = _hf_experts_and_router(cfg, device, dtype, 5)
    ge = _native_experts(cfg, hf, experts, dtype, device)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(n_tokens, cfg.hidden_size, generator=g).to(device, dtype)
    grad = torch.randn(n_tokens, cfg.hidden_size, generator=g).to(device, dtype)
    with torch.no_grad():
        _, w, idx = router(x.float())
    scale = float(cfg.routed_scaling_factor)
    mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

    x_hf, w_hf = x.clone().requires_grad_(), w.clone().requires_grad_()
    out_hf = hf(x_hf, idx, w_hf) * scale
    out_hf.backward(grad)
    x_nat, w_nat = x.clone().requires_grad_(), (w * scale).clone().requires_grad_()
    out_nat = ge(x_nat, mask, w_nat, idx)
    out_nat.backward(grad)

    def maxdiff(a, b):
        return math.nan if a is None or b is None else (a.float() - b.float()).abs().max().item()

    metrics = {
        "out": maxdiff(out_nat, out_hf),
        "cos": F.cosine_similarity(out_nat.float().flatten(), out_hf.float().flatten(), dim=0).item(),
        "grad_x": maxdiff(x_nat.grad, x_hf.grad),
        "grad_gate_up": maxdiff(ge.gate_and_up_projs.grad.transpose(1, 2), hf.gate_up_proj.grad),
        "grad_down": maxdiff(ge.down_projs.grad.transpose(1, 2), hf.down_proj.grad),
        # native routing weights carry the scale: dL/dw_nat = dL/dw_hf / scale
        "grad_w": maxdiff(None if w_nat.grad is None else w_nat.grad * scale, w_hf.grad),
    }
    return metrics, out_nat, ge


class TestCompleteExpertsVsHF:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("experts", ["torch", pytest.param("torch_mm", marks=CUDA)])
    def test_fp32_wiring_is_tight(self, beam_config, device, experts):
        """fp32 activations: the default placement must match HF closely (validates the weight/layout mapping, the scale
        placement and the gradient bookkeeping independent of bf16 rounding)."""
        if experts == "torch_mm" and device == "cpu":
            pytest.skip("torch._grouped_mm needs CUDA")
        m, _, _ = _compare_experts(beam_config, experts, torch.float32, device)
        assert m["out"] < 1e-5 and m["grad_x"] < 1e-5 and m["grad_gate_up"] < 1e-4 and m["grad_down"] < 1e-4, m
        assert m["grad_w"] < 1e-4, m

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("experts", ["torch", pytest.param("torch_mm", marks=CUDA)])
    def test_bf16_execution_and_bounds(self, beam_config, device, experts):
        """bf16 forward + backward execute on the real expert implementation with the default routing-weight placement;
        outputs/gradients finite and close to HF (cosine > 0.999)."""
        if experts == "torch_mm" and device == "cpu":
            pytest.skip("torch._grouped_mm needs CUDA")
        m, out, ge = _compare_experts(beam_config, experts, torch.bfloat16, device)
        assert torch.isfinite(out).all() and m["cos"] > 0.999, m
        assert all(torch.isfinite(p.grad).all() for p in ge.parameters())
