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

"""Closed-form checks of the shared fp32 decay gates.

``kda_decay_gate`` (Kimi K3, Kimi Linear, GLM-5.3) and ``gdn_decay_gate`` (Qwen3-Next, Qwen3.5,
Qwen3.8-Flash-Next) are compared bitwise with the closed-form fp32 formulas on every layout the
families use (flat / per-head gates, 1-D / 4-D ``A_log``); the fused FLA path is checked against
the torch path; and the Qwen3-Next override's forward is compared bitwise with HF's.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
    torch_chunk_gated_delta_rule,
)

from nemo_automodel.components.models.common.fp32_gates import (
    HAVE_FUSED_KDA_GATE,
    gdn_decay_gate,
    kda_decay_gate,
)
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextFp32GatedDeltaNet

HEADS, HEAD_DIM = 4, 8
LOWER_BOUNDS = (None, -5.0)
GATE_LAYOUTS = ("flat", "per_head")
A_LOG_LAYOUTS = ("1d", "4d")


# --------------------------------------------------------------------------- closed forms
def _softplus_decay(gate: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Kimi Linear / unbounded KDA: ``-exp(A_log) * softplus(g + dt_bias)`` on a ``[..., H, D]`` gate."""
    return -a_log.float().view(HEADS, 1).exp() * F.softplus(gate.float() + dt_bias.float().view(HEADS, HEAD_DIM))


def _bounded_decay(gate: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float) -> torch.Tensor:
    """Kimi K3 / GLM-5.3 bounded KDA: ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``."""
    decay = a_log.float().view(HEADS, 1).exp()
    return lower_bound * torch.sigmoid(decay * (gate.float() + dt_bias.float().view(HEADS, HEAD_DIM)))


def _kda_reference(g: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float | None):
    gate = g if g.shape[-1] == HEAD_DIM else g.reshape(*g.shape[:-1], HEADS, HEAD_DIM)
    if lower_bound is None:
        return _softplus_decay(gate, a_log, dt_bias)
    return _bounded_decay(gate, a_log, dt_bias, lower_bound)


def _gdn_reference(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """GatedDeltaNet: ``-exp(A_log) * softplus(a + dt_bias)``, HF's inline formula with fp32 ``dt_bias``."""
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias)


# --------------------------------------------------------------------------- inputs
def _kda_inputs(device: str, gate_dtype: torch.dtype, gate_layout: str, a_log_layout: str):
    generator = torch.Generator(device="cpu").manual_seed(7)
    g = torch.randn(2, 5, HEADS * HEAD_DIM, generator=generator).to(device=device, dtype=gate_dtype)
    a_log = torch.empty(HEADS).uniform_(1, 16, generator=generator).log().to(device)
    dt_bias = torch.randn(HEADS * HEAD_DIM, generator=generator).to(device)
    if gate_layout == "per_head":
        g = g.reshape(2, 5, HEADS, HEAD_DIM)
    if a_log_layout == "4d":
        a_log = a_log.view(1, 1, HEADS, 1)
    return g, a_log, dt_bias


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


# --------------------------------------------------------------------------- torch path
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", LOWER_BOUNDS)
@pytest.mark.parametrize("gate_layout", GATE_LAYOUTS)
@pytest.mark.parametrize("a_log_layout", A_LOG_LAYOUTS)
def test_torch_kda_gate_matches_closed_form_bitwise(device, gate_dtype, lower_bound, gate_layout, a_log_layout):
    g, a_log, dt_bias = _kda_inputs(device, gate_dtype, gate_layout, a_log_layout)

    out = kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=lower_bound, use_fused=False)

    assert out.dtype is torch.float32 and out.shape == (2, 5, HEADS, HEAD_DIM)
    assert torch.equal(out, _kda_reference(g, a_log, dt_bias, lower_bound))


def test_torch_kda_gate_backpropagates_to_fp32_parameters():
    g, a_log, dt_bias = _kda_inputs("cpu", torch.bfloat16, "flat", "1d")
    a_log, dt_bias = a_log.requires_grad_(), dt_bias.requires_grad_()

    kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=-5.0, use_fused=False).sum().backward()

    assert a_log.grad is not None and a_log.grad.dtype is torch.float32 and torch.isfinite(a_log.grad).all()
    assert dt_bias.grad is not None and dt_bias.grad.dtype is torch.float32 and torch.isfinite(dt_bias.grad).all()


# --------------------------------------------------------------------------- fused path
fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_FUSED_KDA_GATE), reason="the fused KDA gate needs CUDA and the fla extra"
)


@fused
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", LOWER_BOUNDS)
@pytest.mark.parametrize("gate_layout", GATE_LAYOUTS)
@pytest.mark.parametrize("a_log_layout", A_LOG_LAYOUTS)
def test_fused_kda_gate_matches_torch_path(gate_dtype, lower_bound, gate_layout, a_log_layout):
    """The fused Triton gate agrees with the torch path on every installed FLA API generation."""
    g, a_log, dt_bias = _kda_inputs("cuda", gate_dtype, gate_layout, a_log_layout)

    out = kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=lower_bound, use_fused=True)

    expected = kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=lower_bound, use_fused=False)
    assert out.dtype is torch.float32 and out.shape == (2, 5, HEADS, HEAD_DIM)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- GatedDeltaNet gate
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("a_dtype", [torch.bfloat16, torch.float32])
def test_gdn_decay_gate_matches_hf_formula_bitwise(device, a_dtype):
    generator = torch.Generator(device="cpu").manual_seed(3)
    a = torch.randn(2, 5, HEADS, generator=generator).to(device=device, dtype=a_dtype)
    A_log = torch.empty(HEADS).uniform_(0, 16, generator=generator).log().to(device)
    dt_bias = torch.randn(HEADS, generator=generator).to(device)

    out = gdn_decay_gate(a, A_log, dt_bias)

    assert out.dtype is torch.float32
    assert torch.equal(out, _gdn_reference(a, A_log, dt_bias))


def test_gdn_decay_gate_backpropagates_to_fp32_parameters():
    a = torch.randn(2, 5, HEADS).to(torch.bfloat16)
    A_log = torch.zeros(HEADS, requires_grad=True)
    dt_bias = torch.ones(HEADS, requires_grad=True)

    gdn_decay_gate(a, A_log, dt_bias).sum().backward()

    for param in (A_log, dt_bias):
        assert param.grad is not None and param.grad.dtype is torch.float32 and torch.isfinite(param.grad).all()


# --------------------------------------------------------------------------- Qwen3-Next forward (M3)
def _qwen3_next_config() -> Qwen3NextConfig:
    return Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )


def _use_torch_kernels(gdn: Qwen3NextGatedDeltaNet) -> None:
    """Route the CPU forward through HF's pure-torch conv and chunk kernels.

    transformers <= 5.12 keeps the kernel callables as instance attributes (as does the
    Automodel override); 5.15 dispatches module-level fallbacks, which are already torch.
    """
    for name, value in (("causal_conv1d_fn", None), ("chunk_gated_delta_rule", torch_chunk_gated_delta_rule)):
        if name in vars(gdn):
            setattr(gdn, name, value)


def test_qwen3_next_fp32_gdn_forward_matches_hf_forward_bitwise():
    """With fp32 ``A_log``/``dt_bias`` the override's gate equals HF's inline gate bitwise.

    The override is kept for the 4D-mask guard and the kernel binding (see the class
    docstring), not for the gate: the full forward matches HF's on the torch kernels.
    """
    # With FLA installed, HF builds the gated norm on the current CUDA device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)
    config = _qwen3_next_config()
    ours = Qwen3NextFp32GatedDeltaNet(config, layer_idx=0).to(device)
    theirs = Qwen3NextGatedDeltaNet(config, layer_idx=0).to(device)
    theirs.load_state_dict(ours.state_dict())
    _use_torch_kernels(ours)
    _use_torch_kernels(theirs)
    hidden_states = torch.randn(2, 8, config.hidden_size, device=device)

    assert ours.A_log.dtype is torch.float32 and ours.dt_bias.dtype is torch.float32
    assert torch.equal(ours(hidden_states), theirs(hidden_states))
