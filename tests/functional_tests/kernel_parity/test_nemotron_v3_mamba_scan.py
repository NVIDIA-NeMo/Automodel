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

"""``mamba_scan.split_conv1d_scan`` of Nemotron V3 against mamba_ssm's own fused scan on the real kernels.

Every gradient has to equal the library's bit for bit, except the ones the library accumulates with atomic
adds and the norm weight when the launch grid differs; the gated-RMSNorm backward launched from this
repository is also checked against float32 autograd of the norm's formula, and a mixer whose scan state is
exempt from activation checkpointing is counted to run its scan once. The first run of a dtype
compiles the Triton kernels of the scan, about a minute each, which is why these comparisons are not unit
tests.
"""

import math

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer
from nemo_automodel.components.models.nemotron_v3.mamba_scan import _gated_rmsnorm_backward, split_conv1d_scan
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mamba_scan import _backend
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="The Mamba scan kernels run on CUDA")


class _GridRecorder:
    """Stands in for a Triton kernel and records the grid of every launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(tuple(grid))
        return self.kernel[grid]


@pytest.mark.parametrize("scale", [1, 8])
@pytest.mark.parametrize("recompute_output", [False, True])
def test_gated_rmsnorm_backward_matches_float32_autograd(monkeypatch, scale, recompute_output):
    layernorm_gated = pytest.importorskip("mamba_ssm.ops.triton.layernorm_gated")
    recorder = _GridRecorder(layernorm_gated._layer_norm_bwd_kernel)
    monkeypatch.setattr(layernorm_gated, "_layer_norm_bwd_kernel", recorder)
    device = torch.device("cuda")
    rows, ngroups, group_size, eps = 1200, 4, 32, 1e-5
    width = ngroups * group_size
    generator = torch.Generator(device=device).manual_seed(5)
    x, z, dy = (torch.randn(rows, width, device=device, generator=generator) for _ in range(3))
    weight = torch.rand(width, device=device, generator=generator) + 0.5

    # Independent expectation: the formula of the gated norm under float32 autograd.
    x_ref, z_ref, weight_ref = (tensor.clone().requires_grad_() for tensor in (x, z, weight))
    gated = (x_ref * z_ref * torch.sigmoid(z_ref)).view(rows, ngroups, group_size)
    rstd = torch.rsqrt(gated.square().mean(-1) + eps)
    y = (gated * rstd.unsqueeze(-1)).view(rows, width) * weight_ref
    expected_dx, expected_dz, expected_dweight = torch.autograd.grad((y * dy).sum(), (x_ref, z_ref, weight_ref))

    dz = torch.full_like(z, float("nan"))
    recomputed = torch.full_like(x, float("nan")) if recompute_output else None
    dx, dweight = _gated_rmsnorm_backward(
        dy,
        x,
        weight,
        eps,
        rstd.detach().T.contiguous().view(-1),  # the kernel reads one row of reciprocal RMS values per group
        z,
        group_size=group_size,
        dz=dz,
        recomputed_out=recomputed,
        grid_scale=scale,
    )
    torch.testing.assert_close(dx, expected_dx, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dz, expected_dz, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dweight, expected_dweight, rtol=1e-4, atol=1e-4)
    if recompute_output:
        torch.testing.assert_close(recomputed, y.detach(), rtol=1e-5, atol=1e-6)

    # 32 columns per group run with one warp, for which mamba_ssm launches four row blocks per multiprocessor.
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert recorder.grids == [(min(math.ceil(sm_count * scale * 4 / ngroups), rows), ngroups)]
    # The library's own launch for the same shapes, to pin "scale 1 is the library's grid".
    layernorm_gated._layer_norm_bwd(
        dy, x, weight, None, eps, None, rstd.detach().T.contiguous().view(-1), z, group_size=group_size,
        norm_before_gate=False, is_rms_norm=True,
    )  # fmt: skip
    library_grid = recorder.grids[1]
    assert (recorder.grids[0] == library_grid) == (scale == 1)
    assert recorder.grids[0][0] == min(library_grid[0] * scale, rows)


def _scan_inputs(dtype: torch.dtype, fused: bool) -> tuple[MockNemotronV3Config, dict[str, torch.Tensor]]:
    """Leaf tensors for one scan call; four heads in two groups so the norm has more than one group."""
    config = MockNemotronV3Config(mamba_num_heads=4, n_groups=2)
    device = torch.device("cuda")
    dim = config.mamba_num_heads * config.mamba_head_dim
    conv_dim = dim + 2 * config.n_groups * config.ssm_state_size
    generator = torch.Generator(device=device).manual_seed(20261123)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, device=device, generator=generator) * scale).to(dtype)

    # 160 tokens = 2.5 chunks of 64, so the scan crosses chunk boundaries and pads the last chunk.
    tensors = {
        "zxbcdt": randn(2, 160, dim + conv_dim + config.mamba_num_heads),
        "conv1d_weight": randn(conv_dim, config.conv_kernel, scale=0.5),
        "conv1d_bias": randn(conv_dim, scale=0.1),
        "dt_bias": torch.rand(config.mamba_num_heads, device=device, generator=generator) + 0.1,
        "A": -torch.rand(config.mamba_num_heads, device=device, generator=generator) - 0.5,
        "D": torch.rand(config.mamba_num_heads, device=device, generator=generator) + 0.5,
        "rmsnorm_weight": (torch.rand(dim, device=device, generator=generator) + 0.5).to(dtype),
    }
    if fused:
        tensors["outproj_weight"] = randn(config.hidden_size, dim, scale=0.1)
    return config, {name: tensor.requires_grad_() for name, tensor in tensors.items()}


# mamba_ssm accumulates these with atomic adds: two runs of the library itself differ in the last bits.
_ATOMIC_ADD_GRADIENTS = ("conv1d_weight", "conv1d_bias", "dt_bias", "A")


def _assert_close_to_rounding(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    tolerance = 1e-4 if expected.dtype is torch.float32 else 2e-2
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=tolerance, atol=tolerance * float(expected.abs().max()), msg=name
    )


def _weighted_sum(output: torch.Tensor) -> torch.Tensor:
    weights = torch.cos(torch.arange(output.numel(), device=output.device, dtype=torch.float32))
    return (output.float() * weights.reshape(output.shape)).sum()


@pytest.mark.parametrize("checkpoint_exempt", [False, True], ids=["one_node", "two_nodes"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("scale", [1, 8])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_split_conv1d_scan_matches_the_library_scan_on_the_real_kernels(dtype, fused, scale, packed, checkpoint_exempt):
    ssd_combined = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    config, tensors = _scan_inputs(dtype, fused)
    seq_idx = None
    if packed:
        # Two sub-sequences per row with the boundary inside a chunk.
        seq_idx = (torch.arange(160, device="cuda") >= 90).to(torch.int32).expand(2, -1).contiguous()
    common = {
        "chunk_size": config.chunk_size,
        "seq_idx": seq_idx,
        "activation": config.mamba_hidden_act,
        "rmsnorm_weight": tensors["rmsnorm_weight"],
        "rmsnorm_eps": config.layer_norm_epsilon,
        "outproj_weight": tensors.get("outproj_weight"),
        "outproj_bias": None,
        "headdim": config.mamba_head_dim,
        "ngroups": config.n_groups,
    }
    positional = [tensors[name] for name in ("zxbcdt", "conv1d_weight", "conv1d_bias", "dt_bias", "A")]
    leaves = list(tensors.values())

    library_out = ssd_combined.mamba_split_conv1d_scan_combined(
        *positional, D=tensors["D"], norm_before_gate=False, return_final_states=False, **common
    )
    library_grads = torch.autograd.grad(_weighted_sum(library_out), leaves)
    own_out = split_conv1d_scan(
        *positional,
        tensors["D"],
        dt_limit=(0.0, float("inf")),
        norm_backward_grid_scale=scale,
        checkpoint_exempt=checkpoint_exempt,
        **common,
    )
    own_grads = torch.autograd.grad(_weighted_sum(own_out), leaves)

    assert own_out.dtype == dtype and own_out.shape == (2, 160, config.hidden_size if fused else 64)
    assert torch.equal(own_out, library_out)
    for name, own, library in zip(tensors, own_grads, library_grads):
        assert torch.isfinite(library).all() and torch.count_nonzero(library) > 0, name
        if name in _ATOMIC_ADD_GRADIENTS or (name == "rmsnorm_weight" and scale != 1):
            # Another summation order in float32, then one rounding to the gradient's dtype.
            _assert_close_to_rounding(own, library, name)
        else:
            assert torch.equal(own, library), name


@pytest.mark.parametrize("checkpoint_exempt", [False, True], ids=["one_node", "two_nodes"])
@pytest.mark.parametrize("scale", [1, 8])
def test_split_conv1d_scan_takes_an_expanded_output_gradient(scale, checkpoint_exempt):
    """The gradient of ``out.sum()`` is one element expanded to the shape of the output, with stride 0.

    Without an output projection that tensor reaches the gated-RMSNorm backward kernel, which reads every row
    as consecutive elements and would run past the single element. Here the expanded gradient is built by
    hand, so that the memory behind its element is known to hold NaN. The gradients have to be those of a
    dense gradient of ones through mamba_ssm's own scan, whose backward refuses the expanded one.
    """
    ssd_combined = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    config, tensors = _scan_inputs(torch.float32, fused=False)
    common = {
        "chunk_size": config.chunk_size,
        "seq_idx": None,
        "activation": config.mamba_hidden_act,
        "rmsnorm_weight": tensors["rmsnorm_weight"],
        "rmsnorm_eps": config.layer_norm_epsilon,
        "outproj_weight": None,
        "outproj_bias": None,
        "headdim": config.mamba_head_dim,
        "ngroups": config.n_groups,
    }
    positional = [tensors[name] for name in ("zxbcdt", "conv1d_weight", "conv1d_bias", "dt_bias", "A")]
    leaves = list(tensors.values())

    library_out = ssd_combined.mamba_split_conv1d_scan_combined(
        *positional, D=tensors["D"], norm_before_gate=False, return_final_states=False, **common
    )
    library_grads = torch.autograd.grad(library_out, leaves, grad_outputs=torch.ones_like(library_out))
    own_out = split_conv1d_scan(
        *positional,
        tensors["D"],
        dt_limit=(0.0, float("inf")),
        norm_backward_grid_scale=scale,
        checkpoint_exempt=checkpoint_exempt,
        **common,
    )
    backing = torch.full((own_out.shape[-1],), float("nan"), device=own_out.device, dtype=own_out.dtype)
    backing[0] = 1.0
    expanded = backing[:1].reshape(()).expand(own_out.shape)
    assert expanded.stride() == (0, 0, 0)
    own_grads = torch.autograd.grad(own_out, leaves, grad_outputs=expanded)

    assert torch.equal(own_out, library_out)
    for name, own, library in zip(tensors, own_grads, library_grads):
        assert torch.isfinite(library).all() and torch.count_nonzero(library) > 0, name
        if name in _ATOMIC_ADD_GRADIENTS or (name == "rmsnorm_weight" and scale != 1):
            _assert_close_to_rounding(own, library, name)
        else:
            assert torch.equal(own, library), name


@pytest.mark.parametrize("checkpoint_exempt", [False, True], ids=["one_node", "two_nodes"])
def test_split_conv1d_scan_backward_runs_in_the_autocast_dtype_of_its_forward(checkpoint_exempt):
    """Under ``torch.autocast`` the output projection's backward runs in the autocast dtype of the forward.

    FP32 tensors under a BF16 autocast: the library's backward computes the projection's input gradient in
    BF16. A backward that ran in another autocast dtype, or in none, would round it differently.
    """
    ssd_combined = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    config, tensors = _scan_inputs(torch.float32, fused=True)
    common = {
        "chunk_size": config.chunk_size,
        "seq_idx": None,
        "activation": config.mamba_hidden_act,
        "rmsnorm_weight": tensors["rmsnorm_weight"],
        "rmsnorm_eps": config.layer_norm_epsilon,
        "outproj_weight": tensors["outproj_weight"],
        "outproj_bias": None,
        "headdim": config.mamba_head_dim,
        "ngroups": config.n_groups,
    }
    positional = [tensors[name] for name in ("zxbcdt", "conv1d_weight", "conv1d_bias", "dt_bias", "A")]
    leaves = list(tensors.values())

    with torch.autocast("cuda", dtype=torch.bfloat16):
        library_out = ssd_combined.mamba_split_conv1d_scan_combined(
            *positional, D=tensors["D"], norm_before_gate=False, return_final_states=False, **common
        )
        own_out = split_conv1d_scan(
            *positional,
            tensors["D"],
            dt_limit=(0.0, float("inf")),
            norm_backward_grid_scale=1,
            checkpoint_exempt=checkpoint_exempt,
            **common,
        )
    # The backward runs outside the autocast context, as ``loss.backward()`` does in a training step.
    library_grads = torch.autograd.grad(_weighted_sum(library_out), leaves)
    own_grads = torch.autograd.grad(_weighted_sum(own_out), leaves)
    assert torch.equal(own_out, library_out)
    for name, own, library in zip(tensors, own_grads, library_grads):
        assert torch.isfinite(library).all() and torch.count_nonzero(library) > 0, name
        if name in _ATOMIC_ADD_GRADIENTS:
            _assert_close_to_rounding(own, library, name)
        else:
            assert torch.equal(own, library), name
    # The comparison can tell the autocast dtypes apart: without autocast the input gradient is another one.
    plain_out = split_conv1d_scan(
        *positional,
        tensors["D"],
        dt_limit=(0.0, float("inf")),
        norm_backward_grid_scale=1,
        checkpoint_exempt=checkpoint_exempt,
        **common,
    )
    plain_grads = torch.autograd.grad(_weighted_sum(plain_out), leaves)
    assert not torch.equal(plain_grads[0], library_grads[0])


@pytest.mark.parametrize("mamba_linear", ["torch", "te"])
def test_mixer_with_a_scaled_grid_matches_the_default_mixer_on_the_real_kernels(mamba_linear):
    """Torch projections apply the output projection inside the kernel, Transformer Engine ones after it."""
    pytest.importorskip("mamba_ssm")
    if mamba_linear == "te":
        pytest.importorskip("transformer_engine")
    config = MockNemotronV3Config(mamba_num_heads=4, n_groups=2)
    torch.manual_seed(20261123)
    default = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_backend(mamba_linear=mamba_linear))
    scaled = NemotronV3Mamba2Mixer(
        config, layer_idx=0, backend=_backend(mamba_linear=mamba_linear, mamba_norm_backward_grid_scale=8)
    )
    with torch.no_grad():
        default.norm.weight.uniform_(0.5, 1.5)
    scaled.load_state_dict(default.state_dict())
    inputs = torch.randn(2, 160, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    results = []
    for mixer in (default, scaled):
        mixer.to(device="cuda", dtype=torch.bfloat16)
        mixer._fp32_params.float()  # the SSM parameters are fp32 in every recipe
        hidden = inputs.clone().requires_grad_()
        output = mixer(hidden)
        _weighted_sum(output).backward()
        results.append((output.detach(), hidden.grad, {name: p.grad for name, p in mixer.named_parameters()}))
    (default_out, default_dx, default_grads), (scaled_out, scaled_dx, scaled_grads) = results
    assert torch.equal(scaled_out, default_out)
    assert torch.equal(scaled_dx, default_dx)
    for name, expected in default_grads.items():
        assert expected is not None and torch.count_nonzero(expected) > 0, name
        if name in ("in_proj.weight", "out_proj.weight", "_fp32_params.D"):
            assert torch.equal(scaled_grads[name], expected), name
        else:
            _assert_close_to_rounding(scaled_grads[name], expected, name)


class _ForwardCounter:
    """Counts the calls of a function it wraps."""

    def __init__(self, function):
        self.function = function
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.function(*args, **kwargs)


@pytest.mark.parametrize("scale", [1, 8])
@pytest.mark.parametrize("mamba_linear", ["torch", "te"])
@pytest.mark.parametrize("scan_exempt", [False, True])
def test_scan_exempt_mixer_recomputes_only_its_input_projection(monkeypatch, scan_exempt, mamba_linear, scale):
    """Under a checkpoint, an exempt scan runs once and the gradients are those of an uncheckpointed mixer.

    Torch projections apply the output projection inside the kernel, Transformer Engine ones after it.
    """
    ssd_combined = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    if mamba_linear == "te":
        pytest.importorskip("transformer_engine")
    config = MockNemotronV3Config(mamba_num_heads=4, n_groups=2)
    backend = _backend(mamba_linear=mamba_linear, mamba_norm_backward_grid_scale=scale)
    torch.manual_seed(20261123)
    mixer = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=backend)
    with torch.no_grad():
        mixer.norm.weight.uniform_(0.5, 1.5)
    mixer.to(device="cuda", dtype=torch.bfloat16)
    mixer._fp32_params.float()  # the SSM parameters are fp32 in every recipe
    inputs = torch.randn(2, 160, config.hidden_size, device="cuda", dtype=torch.bfloat16)

    def gradients(checkpointed: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        mixer.zero_grad(set_to_none=True)
        hidden = inputs.clone().requires_grad_()
        output = checkpoint(mixer, hidden, use_reentrant=False) if checkpointed else mixer(hidden)
        _weighted_sum(output).backward()
        return output.detach(), hidden.grad, {name: p.grad.clone() for name, p in mixer.named_parameters()}

    # Reference: the same mixer with the library scan (or the one-node scan), not checkpointed.
    reference_out, reference_dx, reference_grads = gradients(checkpointed=False)

    mixer.scan_ac_exempt = scan_exempt
    scan_forward = _ForwardCounter(ssd_combined.MambaSplitConv1dScanCombinedFn.forward)
    monkeypatch.setattr(ssd_combined.MambaSplitConv1dScanCombinedFn, "forward", staticmethod(scan_forward))
    in_proj_calls = []
    mixer.in_proj.register_forward_pre_hook(lambda module, args: in_proj_calls.append(True))
    out, dx, grads = gradients(checkpointed=True)

    assert len(in_proj_calls) == 2, "the input projection runs in the forward and in the recompute"
    assert scan_forward.calls == (1 if scan_exempt else 2)
    assert torch.equal(out, reference_out)
    assert torch.equal(dx, reference_dx)
    for name, expected in reference_grads.items():
        assert torch.count_nonzero(expected) > 0, name
        if name in ("in_proj.weight", "out_proj.weight", "_fp32_params.D"):
            assert torch.equal(grads[name], expected), name
        else:
            _assert_close_to_rounding(grads[name], expected, name)
