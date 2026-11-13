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

"""Mamba output-projection dispatch and real LoRA-gradient regression.

CPU tests replace only the unavailable CUDA scan with a differentiable stand-in.
They exercise the native mixer and actual adapter implementations. The expected
projection is independent matrix algebra; this is not a Mamba-kernel parity test.
"""

import sys
from types import ModuleType

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components._peft.lora import LinearLoRA, TritonLinearLoRA, patch_linear_module
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

pytestmark = pytest.mark.cpu_only


def _pre_projection(projected: torch.Tensor, width: int, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """A deterministic normalized scan stand-in; not an SSM reference."""
    gate = projected[..., :width]
    value = projected[..., width : 2 * width] * torch.sigmoid(gate)
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight


def _install_cpu_scan(monkeypatch, mixer: NemotronV3Mamba2Mixer):
    calls = []

    def scan(projected, conv_weight, conv_bias, dt_bias, a, **kwargs):
        assert kwargs["rmsnorm_weight"] is mixer.norm.weight
        calls.append(kwargs)
        result = _pre_projection(projected, mixer.intermediate_size, kwargs["rmsnorm_weight"], kwargs["rmsnorm_eps"])
        if kwargs["outproj_weight"] is not None:
            result = F.linear(result, kwargs["outproj_weight"], kwargs["outproj_bias"])
        return result

    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    module = ModuleType("mamba_ssm.ops.triton.ssd_combined")
    module.mamba_split_conv1d_scan_combined = scan
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def _mixer() -> NemotronV3Mamba2Mixer:
    torch.manual_seed(20261108)
    return NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0).float()


@pytest.mark.parametrize("adapter", ["patched", "triton"])
@pytest.mark.parametrize("nonzero_b", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_mamba_projection_preserves_adapter_forward_and_all_gradients(monkeypatch, adapter, nonzero_b, packed):
    mixer = _mixer()
    mixer.requires_grad_(False)
    if adapter == "patched":
        mixer.out_proj = patch_linear_module(
            mixer.out_proj, dim=4, alpha=8, use_triton=False, use_memory_efficient_lora=False
        )
        assert isinstance(mixer.out_proj, LinearLoRA)
        assert type(mixer.out_proj).__name__ == "PatchedLinearLoRA"
    else:
        mixer.out_proj = TritonLinearLoRA(mixer.out_proj, dim=4, alpha=8, use_memory_efficient_lora=False)
        mixer.out_proj.weight.requires_grad_(False)
        if mixer.out_proj.bias is not None:
            mixer.out_proj.bias.requires_grad_(False)
    if nonzero_b:
        with torch.no_grad():
            mixer.out_proj.lora_B.weight.normal_(std=0.1)
    assert mixer.out_proj.lora_A.weight.requires_grad and mixer.out_proj.lora_B.weight.requires_grad
    calls = _install_cpu_scan(monkeypatch, mixer)
    forward_calls = []
    mixer.out_proj.register_forward_pre_hook(lambda module, args: forward_calls.append(tuple(args[0].shape)))

    shape = (14, mixer.hidden_size) if packed else (2, 7, mixer.hidden_size)
    x = torch.randn(shape, requires_grad=True)
    seq_idx = torch.tensor([[0] * 7 + [1] * 7], dtype=torch.int32) if packed else None
    actual = mixer(x, seq_idx=seq_idx)
    projection = mixer.out_proj
    x_reference = x.detach().clone().requires_grad_()
    projected = mixer.in_proj(x_reference.unsqueeze(0) if packed else x_reference)
    normalized = _pre_projection(projected, mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon)
    base = F.linear(normalized, projection.weight, projection.bias)
    update = F.linear(F.linear(normalized, projection.lora_A.weight), projection.lora_B.weight) * projection.scale
    expected = base + update
    if packed:
        expected = expected.squeeze(0)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    upstream = torch.cos(torch.arange(actual.numel(), dtype=actual.dtype)).reshape(actual.shape)
    (actual * upstream).sum().backward()
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), (x_reference, projection.lora_A.weight, projection.lora_B.weight)
    )
    # At B=0 the A gradient is mathematically zero, but it must exist in the graph.
    assert projection.lora_A.weight.grad is not None
    assert projection.lora_B.weight.grad is not None
    torch.testing.assert_close(x.grad, expected_gradients[0], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(projection.lora_A.weight.grad, expected_gradients[1], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(projection.lora_B.weight.grad, expected_gradients[2], rtol=2e-6, atol=2e-6)
    assert torch.count_nonzero(projection.lora_B.weight.grad) > 0
    assert bool(torch.count_nonzero(projection.lora_A.weight.grad)) == nonzero_b
    assert len(forward_calls) == 1
    assert calls[0]["outproj_weight"] is None and calls[0]["outproj_bias"] is None
    assert calls[0]["rmsnorm_weight"] is mixer.norm.weight
    assert calls[0]["seq_idx"] is seq_idx


def test_plain_linear_keeps_output_projection_inside_fused_kernel(monkeypatch):
    mixer = _mixer()
    assert type(mixer.out_proj) is nn.Linear
    calls = _install_cpu_scan(monkeypatch, mixer)
    forward_calls = []
    mixer.out_proj.register_forward_pre_hook(lambda module, args: forward_calls.append(True))
    x = torch.randn(2, 7, mixer.hidden_size, requires_grad=True)
    actual = mixer(x)
    normalized = _pre_projection(
        mixer.in_proj(x), mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon
    )
    expected = F.linear(normalized, mixer.out_proj.weight, mixer.out_proj.bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert mixer.out_proj.weight.grad is not None
    assert calls[0]["outproj_weight"] is mixer.out_proj.weight
    assert calls[0]["outproj_bias"] is mixer.out_proj.bias
    assert forward_calls == []


class _CustomLinear(nn.Linear):
    """A non-plain projection whose module forward is part of its semantics."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs) * 1.25


def test_custom_linear_subclass_forward_and_module_hooks_are_preserved(monkeypatch):
    mixer = _mixer()
    mixer.out_proj.__class__ = _CustomLinear
    calls = _install_cpu_scan(monkeypatch, mixer)
    forward_calls = []

    def pre_hook(module, args):
        forward_calls.append(True)
        return (args[0] * 2,)

    mixer.out_proj.register_forward_pre_hook(pre_hook)
    x = torch.randn(2, 7, mixer.hidden_size)
    actual = mixer(x)
    normalized = _pre_projection(
        mixer.in_proj(x), mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon
    )
    expected = F.linear(normalized * 2, mixer.out_proj.weight, mixer.out_proj.bias) * 1.25
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert forward_calls == [True]
    assert calls[0]["outproj_weight"] is None
