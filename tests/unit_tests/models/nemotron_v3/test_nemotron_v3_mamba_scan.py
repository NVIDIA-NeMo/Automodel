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

"""``BackendConfig.mamba_norm_backward_grid_scale`` and the mixer's ``scan_ac_exempt`` on a CPU stand-in.

Which scan the mixer runs, how often an output projection that saves its input after computing (as a
Transformer Engine linear does) runs under a non-reentrant activation checkpoint, and what context
parallelism refuses. The comparisons with mamba_ssm's own scan on the real kernels are in
``tests/functional_tests/kernel_parity/test_nemotron_v3_mamba_scan.py``.
"""

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3 import layers as nemotron_layers
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer
from nemo_automodel.components.models.nemotron_v3.mamba_scan import split_conv1d_scan
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mamba_out_projection import (
    _install_cpu_scan,
    _pre_projection,
)
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config


def _backend(**mamba: int | float | bool | str) -> BackendConfig:
    return BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch", **mamba)


@pytest.mark.cpu_only
def test_defaults_keep_the_library_scan_and_checkpointed_scan_state():
    backend = _backend()
    assert backend.mamba_norm_backward_grid_scale == 1
    for mixer in (
        NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0),
        NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0, backend=backend),
    ):
        assert mixer.norm_backward_grid_scale == 1
        assert mixer.scan_ac_exempt is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [2.5, 8.0, True, "8"])
def test_grid_scale_that_is_not_an_integer_is_rejected(scale):
    """YAML can deliver a float, a boolean or a string; the value multiplies a kernel's launch grid."""
    with pytest.raises(ValueError, match="mamba_norm_backward_grid_scale must be an integer"):
        _backend(mamba_norm_backward_grid_scale=scale)


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [0, -3])
def test_grid_scale_below_one_is_rejected(scale):
    with pytest.raises(ValueError, match="mamba_norm_backward_grid_scale"):
        _backend(mamba_norm_backward_grid_scale=scale)
    with pytest.raises(ValueError, match="norm_backward_grid_scale"):
        split_conv1d_scan(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1),
            None,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            chunk_size=1,
            seq_idx=None,
            dt_limit=(0.0, float("inf")),
            activation="silu",
            rmsnorm_weight=torch.zeros(1),
            rmsnorm_eps=1e-5,
            outproj_weight=None,
            outproj_bias=None,
            headdim=1,
            ngroups=1,
            norm_backward_grid_scale=scale,
            checkpoint_exempt=False,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("scale", "scan_exempt"), [(1, False), (8, False), (1, True), (8, True)])
@pytest.mark.parametrize("fused", [True, False], ids=["torch_linear", "module_after_kernel"])
def test_mixer_selects_the_scan_by_grid_scale_and_exemption(monkeypatch, scale, scan_exempt, fused):
    torch.manual_seed(3)
    backend = _backend(mamba_norm_backward_grid_scale=scale)
    mixer = NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0, backend=backend).float()
    if not fused:
        # Only a plain ``torch.nn.Linear`` is applied inside the kernel; any other module, as a
        # Transformer Engine linear, runs after it.
        mixer.out_proj = _SaveAfterLinear(mixer.out_proj.weight)
    mixer.scan_ac_exempt = scan_exempt
    library_calls = _install_cpu_scan(monkeypatch, mixer)
    own_calls = []

    def own_scan(projected, conv_weight, conv_bias, dt_bias, a, d, **kwargs):
        own_calls.append({"conv_weight": conv_weight, "conv_bias": conv_bias, "dt_bias": dt_bias, "D": d, **kwargs})
        result = _pre_projection(projected, mixer.intermediate_size, kwargs["rmsnorm_weight"], kwargs["rmsnorm_eps"])
        if kwargs["outproj_weight"] is not None:
            result = result @ kwargs["outproj_weight"].T
        return result

    monkeypatch.setattr(nemotron_layers, "split_conv1d_scan", own_scan)
    x = torch.randn(2, 7, mixer.hidden_size)
    actual = mixer(x)

    # Independent expectation: the stand-in scan without a projection, then plain matrix algebra.
    expected = (
        _pre_projection(mixer.in_proj(x), mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon)
        @ mixer.out_proj.weight.T
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if scale == 1 and not scan_exempt:
        assert len(library_calls) == 1 and own_calls == []
        assert (library_calls[0]["outproj_weight"] is mixer.out_proj.weight) == fused
        return
    assert library_calls == [] and len(own_calls) == 1
    call = own_calls[0]
    assert call["norm_backward_grid_scale"] == scale
    assert call["checkpoint_exempt"] is scan_exempt
    assert call["rmsnorm_weight"] is mixer.norm.weight
    assert call["rmsnorm_eps"] == mixer.norm.variance_epsilon
    assert (call["outproj_weight"] is mixer.out_proj.weight) == fused
    assert call["outproj_bias"] is None
    assert call["conv_weight"].shape == (mixer.conv_dim, mixer.conv_kernel_size)
    assert call["conv_bias"] is mixer.conv1d.bias
    assert call["chunk_size"] == mixer.chunk_size
    assert call["headdim"] == mixer.head_dim and call["ngroups"] == mixer.n_groups
    assert call["dt_limit"] == (0.0, float("inf"))
    assert call["activation"] == "silu" and call["seq_idx"] is None


class _SaveAfterMatmul(torch.autograd.Function):
    """``x @ weight.T`` that saves its inputs after computing, as a Transformer Engine linear does."""

    @staticmethod
    def forward(ctx, x, weight):
        out = x @ weight.T
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        return grad @ weight, torch.einsum("bso,bsi->oi", grad, x)


class _SaveAfterLinear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.computed = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.computed += 1
        return _SaveAfterMatmul.apply(x, self.weight)


@pytest.mark.cpu_only
@pytest.mark.parametrize("scan_exempt", [False, True])
def test_scan_ac_exempt_keeps_what_a_projection_after_the_kernel_saves(monkeypatch, scan_exempt):
    """The GEMM of a projection applied after the kernel runs twice under a checkpoint unless the scan is exempt.

    The stand-in scans are plain autograd operations, so they run in the forward and in the recompute either
    way: here only the tensors saved by the projection are kept. That the real scan runs once is checked on
    the real kernels.
    """
    torch.manual_seed(17)
    mixer = NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0, backend=_backend()).float()
    mixer.scan_ac_exempt = scan_exempt
    mixer.out_proj = _SaveAfterLinear(mixer.out_proj.weight)
    library_calls = _install_cpu_scan(monkeypatch, mixer)
    own_calls = []

    def own_scan(projected, conv_weight, conv_bias, dt_bias, a, d, **kwargs):
        own_calls.append(kwargs)
        return _pre_projection(projected, mixer.intermediate_size, kwargs["rmsnorm_weight"], kwargs["rmsnorm_eps"])

    monkeypatch.setattr(nemotron_layers, "split_conv1d_scan", own_scan)

    x = torch.randn(2, 7, mixer.hidden_size)
    hidden = x.clone().requires_grad_()
    out = checkpoint(mixer, hidden, use_reentrant=False)
    assert mixer.out_proj.computed == 1
    out.square().sum().backward()

    # An exempt scan goes through ``split_conv1d_scan``, the other one through the library scan.
    scan_calls, other_calls = (own_calls, library_calls) if scan_exempt else (library_calls, own_calls)
    assert len(scan_calls) == 2, "the stand-in scan runs in the forward and in the recompute"
    assert other_calls == []
    assert all(call["outproj_weight"] is None for call in scan_calls)
    assert mixer.out_proj.computed == (1 if scan_exempt else 2)

    # Independent expectation: an uncheckpointed stand-in scan and plain matrix algebra under autograd.
    reference_hidden = x.clone().requires_grad_()
    reference = (
        _pre_projection(
            mixer.in_proj(reference_hidden), mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon
        )
        @ mixer.out_proj.weight.T
    )
    torch.testing.assert_close(out, reference, rtol=1e-6, atol=1e-6)
    names = ["hidden", "in_proj.weight", "norm.weight", "out_proj.weight"]
    actual_grads = [hidden.grad, mixer.in_proj.weight.grad, mixer.norm.weight.grad, mixer.out_proj.weight.grad]
    expected_grads = torch.autograd.grad(
        reference.square().sum(),
        (reference_hidden, mixer.in_proj.weight, mixer.norm.weight, mixer.out_proj.weight),
    )
    for name, actual, expected in zip(names, actual_grads, expected_grads):
        assert actual is not None and torch.count_nonzero(actual) > 0, name
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)


class _UnusedContextParallel:
    """Stands in for the context-parallel transport of a mixer; a refused forward never calls it."""


@pytest.mark.cpu_only
@pytest.mark.parametrize("scan_exempt", [True, False])
def test_checkpoint_exempt_scan_state_is_refused_with_context_parallelism(monkeypatch, scan_exempt):
    """The context-parallel scan has no checkpoint-exempt path: it must not silently recompute the state."""
    torch.manual_seed(23)
    mixer = NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0, backend=_backend()).float()
    mixer.scan_ac_exempt = scan_exempt
    _install_cpu_scan(monkeypatch, mixer)
    x = torch.randn(2, 7, mixer.hidden_size)
    mixer.cp = _UnusedContextParallel()
    if scan_exempt:
        with pytest.raises(ValueError, match="not supported with context parallelism"):
            mixer(x)
    else:
        # The transport is reached: the stand-in has none of its methods.
        with pytest.raises(AttributeError, match="pre_conv_ssm"):
            mixer(x)
