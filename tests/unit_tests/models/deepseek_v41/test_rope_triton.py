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

"""Fused Triton RoPE: CPU-safe dispatch tests and CUDA parity against the eager ``_apply_rope``."""

import math

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41 import attention, rope_triton
from nemo_automodel.components.models.deepseek_v41.attention import _apply_rope, use_triton_rope
from nemo_automodel.components.models.deepseek_v41.rope_triton import HAVE_TRITON, apply_rope_triton

_CUDA_AND_TRITON = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA and Triton are required for the fused RoPE kernel"
)
_COMPILE_REASON = "Triton compiles the fused RoPE kernel per dtype, block shape and direction on first use"

# (label, shape, rotary_pairs) at batch 1 and sequence 64; the attention output is the query shape with inverse=True.
_SHAPES = [
    ("query", (1, 64, 64, 512), 32),
    ("kv", (1, 64, 512), 32),
    ("indexer_q", (1, 64, 32, 128), 32),
]
_FLOAT_INT_VIEWS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def _inputs(shape, pairs, dtype, device):
    torch.manual_seed(0)
    values = (torch.randn(*shape, device=device) * 2).to(dtype)
    angles = torch.rand(shape[0], shape[1], pairs, device=device) * (2 * math.pi)
    grad_output = torch.randn(*shape, device=device).to(dtype)
    return values, angles, grad_output


def _forward_backward(fn, values, angles, inverse, grad_output):
    leaf = values.detach().requires_grad_(True)
    output = fn(leaf, angles, inverse=inverse)
    output.backward(grad_output)
    return output.detach(), leaf.grad


def _ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Per-element distance in representable steps of the shared dtype (exact for finite values)."""
    int_dtype = _FLOAT_INT_VIEWS[actual.dtype]
    bits_min = torch.iinfo(int_dtype).min

    def key(tensor):
        bits = tensor.contiguous().view(int_dtype).to(torch.int64)
        return torch.where(bits < 0, bits_min - bits, bits)

    return (key(actual) - key(expected)).abs()


def _assert_matches(actual: torch.Tensor, expected: torch.Tensor, what: str, scale: torch.Tensor) -> None:
    """Require bitwise equality; failing that, report the ulp gap and bound it by the FMA-contraction tolerance."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    except AssertionError:
        pass
    ulps = _ulp_distance(actual, expected)
    message = (
        f"{what} is not bitwise identical to _apply_rope: {int((ulps > 0).sum())} of {actual.numel()} elements "
        f"differ, max {int(ulps.max())} ulp ({actual.dtype})"
    )
    if actual.dtype == torch.float32:
        # A different FMA contraction moves each product by at most one FP32 ulp of its magnitude, and
        # |cos|, |sin| <= 1 bound the products by the input magnitude.
        atol = 2 * torch.finfo(torch.float32).eps * float(scale.abs().max())
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=atol, msg=message)
    else:
        assert int(ulps.max()) <= 1, message


def test_module_imports_without_triton_or_gpu():
    assert isinstance(HAVE_TRITON, bool)
    assert callable(apply_rope_triton)
    assert attention._ROPE_IMPL == "torch"


def test_use_triton_rope_toggles_dispatch(monkeypatch):
    monkeypatch.setattr(attention, "_ROPE_IMPL", "torch")
    calls = []

    def fake_apply_rope_triton(values, angles, *, inverse=False):
        calls.append((values, angles, inverse))
        return torch.full_like(values, 7.0)

    monkeypatch.setattr(attention, "apply_rope_triton", fake_apply_rope_triton)
    values = torch.randn(1, 3, 2, 8)
    angles = torch.rand(1, 3, 2)
    eager = _apply_rope(values, angles, inverse=True)
    assert calls == []

    use_triton_rope(True)
    assert attention._ROPE_IMPL == "triton"
    routed = _apply_rope(values, angles, inverse=True)
    assert len(calls) == 1
    assert calls[0][0] is values and calls[0][1] is angles and calls[0][2] is True
    assert torch.equal(routed, torch.full_like(values, 7.0))

    use_triton_rope(False)
    assert attention._ROPE_IMPL == "torch"
    assert torch.equal(_apply_rope(values, angles, inverse=True), eager)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
@pytest.mark.parametrize("shape", [(1, 5, 12), (1, 5, 3, 12)], ids=["3d", "4d"])
def test_eager_mirror_matches_attention_apply_rope(shape, inverse, dtype):
    values, angles, _ = _inputs(shape, 4, dtype, "cpu")
    assert torch.equal(
        rope_triton._apply_rope_torch(values, angles, inverse=inverse), _apply_rope(values, angles, inverse=inverse)
    )


def test_backend_rope_triton_routes_the_model(monkeypatch):
    """``BackendConfig(rope="triton")`` flips the process-wide dispatch when the V4.1 model is built; the default does not."""
    from dataclasses import replace

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config

    assert BackendConfig().rope == "torch"
    monkeypatch.setattr(attention, "_ROPE_IMPL", "torch")
    with torch.device("meta"):
        DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    assert attention._ROPE_IMPL == "torch"
    with torch.device("meta"):
        DeepseekV41ForCausalLM(_tiny_config(), backend=replace(_backend(), rope="triton"))
    assert attention._ROPE_IMPL == "triton"


def test_cpu_tensors_fall_back_to_eager():
    values, angles, grad_output = _inputs((1, 5, 3, 12), 4, torch.float32, "cpu")
    expected_out, expected_grad = _forward_backward(_apply_rope, values, angles, True, grad_output)
    actual_out, actual_grad = _forward_backward(apply_rope_triton, values, angles, True, grad_output)
    assert torch.equal(actual_out, expected_out)
    assert torch.equal(actual_grad, expected_grad)


def test_angles_requiring_grad_fall_back_to_eager():
    values, angles, grad_output = _inputs((1, 5, 3, 12), 4, torch.float32, "cpu")
    angles.requires_grad_(True)
    expected_out, expected_grad = _forward_backward(_apply_rope, values, angles, False, grad_output)
    expected_angle_grad, angles.grad = angles.grad, None
    actual_out, actual_grad = _forward_backward(apply_rope_triton, values, angles, False, grad_output)
    assert torch.equal(actual_out, expected_out)
    assert torch.equal(actual_grad, expected_grad)
    assert angles.grad is not None and torch.equal(angles.grad, expected_angle_grad)


@_CUDA_AND_TRITON
@pytest.mark.runtime_budget(30, reason=_COMPILE_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
@pytest.mark.parametrize("label, shape, pairs", _SHAPES, ids=[entry[0] for entry in _SHAPES])
def test_triton_matches_eager_forward_and_backward(label, shape, pairs, inverse, dtype):
    values, angles, grad_output = _inputs(shape, pairs, dtype, "cuda")
    expected_out, expected_grad = _forward_backward(_apply_rope, values, angles, inverse, grad_output)
    actual_out, actual_grad = _forward_backward(apply_rope_triton, values, angles, inverse, grad_output)
    assert actual_out.is_contiguous() and actual_out.data_ptr() != values.data_ptr()
    _assert_matches(actual_out, expected_out, f"{label} output", scale=values)
    _assert_matches(actual_grad, expected_grad, f"{label} input gradient", scale=grad_output)


@_CUDA_AND_TRITON
@pytest.mark.runtime_budget(30, reason=_COMPILE_REASON)
def test_triton_accepts_non_contiguous_values():
    torch.manual_seed(0)
    # Heads-major storage transposed to [batch, sequence, heads, channels]: unit channel stride, permuted row strides.
    transposed = (torch.randn(1, 64, 64, 512, device="cuda") * 2).to(torch.bfloat16).transpose(1, 2)
    # A strided channel dimension has to be made contiguous before the kernel reads it.
    channel_strided = (torch.randn(1, 64, 1024, device="cuda") * 2).to(torch.bfloat16)[..., ::2]
    angles = torch.rand(1, 64, 32, device="cuda") * (2 * math.pi)
    for values in (transposed, channel_strided):
        assert not values.is_contiguous()
        grad_output = torch.randn(values.shape, device="cuda").to(values.dtype)
        expected_out, expected_grad = _forward_backward(_apply_rope, values, angles, True, grad_output)
        actual_out, actual_grad = _forward_backward(apply_rope_triton, values, angles, True, grad_output)
        assert actual_out.is_contiguous()
        _assert_matches(actual_out, expected_out, f"output for strides {values.stride()}", scale=values)
        _assert_matches(actual_grad, expected_grad, f"input gradient for strides {values.stride()}", scale=grad_output)


@_CUDA_AND_TRITON
@pytest.mark.runtime_budget(30, reason=_COMPILE_REASON)
def test_cuda_angles_requiring_grad_use_eager():
    values, angles, grad_output = _inputs((1, 64, 512), 32, torch.bfloat16, "cuda")
    angles.requires_grad_(True)
    expected_out, expected_grad = _forward_backward(_apply_rope, values, angles, False, grad_output)
    expected_angle_grad, angles.grad = angles.grad, None
    actual_out, actual_grad = _forward_backward(apply_rope_triton, values, angles, False, grad_output)
    assert torch.equal(actual_out, expected_out)
    assert torch.equal(actual_grad, expected_grad)
    assert angles.grad is not None and torch.equal(angles.grad, expected_angle_grad)
