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

"""Bitwise equivalence of the shared fp32 decay gates with the per-family code they replaced.

The reference functions below are verbatim copies of the removed Kimi K3, Kimi Linear,
GLM-5.3 and Qwen GatedDeltaNet gate bodies, so a change to ``fp32_gates`` that alters a
single bit of any family's numerics fails here.
"""

from __future__ import annotations

import inspect

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
from nemo_automodel.shared.import_utils import safe_import_from

HEADS, HEAD_DIM = 4, 8
LOWER_BOUNDS = (None, -5.0)
_, _fla_fused_kda_gate = safe_import_from("fla.ops.kda.gate", "fused_kda_gate")
_FUSED_PARAMS = inspect.signature(_fla_fused_kda_gate).parameters if HAVE_FUSED_KDA_GATE else {}
_HAS_G_BIAS = "g_bias" in _FUSED_PARAMS
_HAS_LOWER_BOUND = "lower_bound" in _FUSED_PARAMS


# --------------------------------------------------------------------------- removed family code
def _kimi_k3_torch_kda_gate(g, a_log, head_dim, dt_bias, lower_bound):
    gate = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    num_heads = gate.shape[-2]
    gate = gate.float() + dt_bias.float().view(num_heads, head_dim)
    decay = a_log.float().view(num_heads, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(decay * gate)
    return -decay * F.softplus(gate)


def _kimi_k3_fused_kda_gate(g, a_log, head_dim, dt_bias, lower_bound):
    if _HAS_G_BIAS:
        if lower_bound is not None:
            return _kimi_k3_torch_kda_gate(g, a_log, head_dim, dt_bias, lower_bound)
        return _fla_fused_kda_gate(g, a_log.view(1, 1, -1, 1), head_dim, g_bias=dt_bias)
    gate_input = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    kwargs = {"dt_bias": dt_bias}
    if _HAS_LOWER_BOUND:
        kwargs["lower_bound"] = lower_bound
    elif lower_bound is not None:
        return _kimi_k3_torch_kda_gate(gate_input, a_log, head_dim, dt_bias, lower_bound)
    return _fla_fused_kda_gate(gate_input, a_log, **kwargs)


def _kimi_linear_torch_kda_gate(g, a_log, head_dim, dt_bias):
    gate = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    num_heads = gate.shape[-2]
    gate = gate.float() + dt_bias.float().view(num_heads, head_dim)
    return -a_log.float().view(num_heads, 1).exp() * F.softplus(gate)


def _kimi_linear_fused_kda_gate(g, a_log, head_dim, dt_bias):
    if _HAS_G_BIAS:
        return _fla_fused_kda_gate(g, a_log, head_dim, g_bias=dt_bias)
    gate_input = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    return _fla_fused_kda_gate(gate_input, a_log, dt_bias=dt_bias)


def _glm5_torch_decay_gate(gate, A_log, dt_bias, head_dim, lower_bound):
    gate = gate.reshape(*gate.shape[:-1], -1, head_dim)
    gate = gate.float() + dt_bias.view(1, 1, -1, head_dim)
    decay = A_log.view(1, 1, -1, 1).exp()
    return lower_bound * torch.sigmoid(decay * gate) if lower_bound is not None else -decay * F.softplus(gate)


def _glm5_fused_decay_gate(gate, A_log, dt_bias, head_dim, lower_bound):
    gate = gate.reshape(*gate.shape[:-1], -1, head_dim)
    return _fla_fused_kda_gate(gate, A_log.contiguous(), dt_bias=dt_bias.contiguous(), lower_bound=lower_bound)


def _gdn_compute_gate(a, A_log, dt_bias):
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


def _hf_qwen3_next_gate(a, A_log, dt_bias):
    # transformers Qwen3NextGatedDeltaNet.forward (5.8 through 5.15.1), dt_bias already fp32.
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias)


# --------------------------------------------------------------------------- inputs
def _kda_inputs(device: str, gate_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7)
    g = torch.randn(2, 5, HEADS * HEAD_DIM, generator=generator).to(device=device, dtype=gate_dtype)
    a_log = torch.empty(HEADS).uniform_(1, 16, generator=generator).log().to(device)
    dt_bias = torch.randn(HEADS * HEAD_DIM, generator=generator).to(device)
    return g, a_log, dt_bias


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


# --------------------------------------------------------------------------- torch path
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", LOWER_BOUNDS)
@pytest.mark.parametrize("per_head", [False, True])
def test_torch_kda_gate_matches_kimi_k3(device, gate_dtype, lower_bound, per_head):
    g, a_log, dt_bias = _kda_inputs(device, gate_dtype)
    g = g.reshape(2, 5, HEADS, HEAD_DIM) if per_head else g

    out = kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=lower_bound, use_fused=False)

    expected = _kimi_k3_torch_kda_gate(g, a_log, HEAD_DIM, dt_bias, lower_bound)
    assert out.dtype is torch.float32 and out.shape == (2, 5, HEADS, HEAD_DIM)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
def test_torch_kda_gate_matches_kimi_linear_with_4d_a_log(device, gate_dtype):
    g, a_log, dt_bias = _kda_inputs(device, gate_dtype)
    a_log = a_log.view(1, 1, HEADS, 1)

    out = kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=None, use_fused=False)

    assert torch.equal(out, _kimi_linear_torch_kda_gate(g, a_log, HEAD_DIM, dt_bias))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", LOWER_BOUNDS)
def test_torch_kda_gate_matches_glm5_next(device, gate_dtype, lower_bound):
    g, a_log, dt_bias = _kda_inputs(device, gate_dtype)
    gate = g.reshape(*g.shape[:-1], -1, HEAD_DIM)

    out = kda_decay_gate(gate, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=lower_bound, use_fused=False)

    assert torch.equal(out, _glm5_torch_decay_gate(g, a_log, dt_bias, HEAD_DIM, lower_bound))


def test_torch_kda_gate_backpropagates_to_fp32_parameters():
    g, a_log, dt_bias = _kda_inputs("cpu", torch.bfloat16)
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
def test_fused_kda_gate_matches_kimi_k3(gate_dtype, lower_bound):
    g, a_log, dt_bias = _kda_inputs("cuda", gate_dtype)

    out = kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=lower_bound, use_fused=True)

    assert torch.equal(out, _kimi_k3_fused_kda_gate(g, a_log.contiguous(), HEAD_DIM, dt_bias.contiguous(), lower_bound))


@fused
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
def test_fused_kda_gate_matches_kimi_linear_with_4d_a_log(gate_dtype):
    g, a_log, dt_bias = _kda_inputs("cuda", gate_dtype)
    a_log = a_log.view(1, 1, HEADS, 1)

    out = kda_decay_gate(g, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=None, use_fused=True)

    assert torch.equal(out, _kimi_linear_fused_kda_gate(g, a_log.contiguous(), HEAD_DIM, dt_bias.contiguous()))


@fused
@pytest.mark.skipif(not _HAS_LOWER_BOUND, reason="GLM-5.3 only ran the fused gate on the lower_bound FLA API")
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", LOWER_BOUNDS)
def test_fused_kda_gate_matches_glm5_next(gate_dtype, lower_bound):
    g, a_log, dt_bias = _kda_inputs("cuda", gate_dtype)
    gate = g.reshape(*g.shape[:-1], -1, HEAD_DIM)

    out = kda_decay_gate(gate, a_log, dt_bias, head_dim=HEAD_DIM, lower_bound=lower_bound, use_fused=True)

    assert torch.equal(out, _glm5_fused_decay_gate(g, a_log, dt_bias, HEAD_DIM, lower_bound))


# --------------------------------------------------------------------------- GatedDeltaNet gate
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("a_dtype", [torch.bfloat16, torch.float32])
def test_gdn_decay_gate_matches_the_family_and_hf_formulas(device, a_dtype):
    generator = torch.Generator(device="cpu").manual_seed(3)
    a = torch.randn(2, 5, HEADS, generator=generator).to(device=device, dtype=a_dtype)
    A_log = torch.empty(HEADS).uniform_(0, 16, generator=generator).log().to(device)
    dt_bias = torch.randn(HEADS, generator=generator).to(device)

    out = gdn_decay_gate(a, A_log, dt_bias)

    assert out.dtype is torch.float32
    assert torch.equal(out, _gdn_compute_gate(a, A_log, dt_bias))
    assert torch.equal(out, _hf_qwen3_next_gate(a, A_log, dt_bias))


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
