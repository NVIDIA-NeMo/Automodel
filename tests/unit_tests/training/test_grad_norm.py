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

import pytest
import torch

from nemo_automodel.components.training.triton.grad_norm import (
    _CHUNK,
    HAVE_TRITON,
    _build_chunk_ends,
    multi_tensor_div_,
    multi_tensor_mul_,
    multi_tensor_sumsq,
    sumsq_reference,
)
from nemo_automodel.components.training.utils import _clip_grad_norm_impl, _local_te_l2_norm
from nemo_automodel.shared.import_utils import safe_import_te


def test_build_chunk_ends_handles_empty_and_boundary_sized_tensors():
    tensors = [torch.empty(0), torch.empty(1), torch.empty(_CHUNK), torch.empty(_CHUNK + 1)]

    chunk_ends, total = _build_chunk_ends(tensors, torch.device("cpu"))

    torch.testing.assert_close(chunk_ends, torch.tensor([0, 1, 2, 4], dtype=torch.int64))
    assert total == 4


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
def test_multi_tensor_sumsq_matches_fp64_reference_and_is_repeatable():
    gradients = [
        torch.linspace(-3, 3, _CHUNK + 17, device="cuda").to(torch.bfloat16),
        torch.linspace(-0.25, 0.25, 2 * _CHUNK + 1, device="cuda"),
    ]

    expected = sumsq_reference(gradients)
    actual = [multi_tensor_sumsq(gradients).cpu() for _ in range(3)]

    torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-12)
    assert all(torch.equal(actual[0], repeated) for repeated in actual[1:])


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
def test_te_and_triton_l2_backends_match_fp64_reference():
    if not safe_import_te()[0]:
        pytest.skip("requires Transformer Engine")

    gradients = [
        torch.linspace(-3, 3, _CHUNK + 17, device="cuda").to(torch.bfloat16),
        torch.linspace(-0.25, 0.25, 2 * _CHUNK + 1, device="cuda"),
    ]
    expected = sumsq_reference(gradients).sqrt()

    triton_norm = multi_tensor_sumsq(gradients).sqrt().cpu()
    te_norm = _local_te_l2_norm(gradients, torch.device("cuda")).cpu()

    torch.testing.assert_close(triton_norm, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(te_norm, expected, rtol=2e-6, atol=0)


def _scale_test_tensors(device: str) -> list[torch.Tensor]:
    """Tensors of each kernel dtype with sizes below, at and above the chunk boundary."""
    generator = torch.Generator().manual_seed(1234)
    sizes = [(7,), (_CHUNK,), (_CHUNK + 17,), (3, 2 * _CHUNK + 1)]
    dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
    tensors = []
    for size, dtype in zip(sizes, dtypes, strict=True):
        values = torch.randn(size, generator=generator, dtype=torch.float64) * 3.0
        tensors.append(values.to(dtype).to(device))
    # More BF16 and FP32 tensors so one launch covers several tensors of one dtype.
    tensors.append(
        (torch.randn(4 * _CHUNK + 5, generator=generator, dtype=torch.float64) * 1e-3).to(torch.bfloat16).to(device)
    )
    tensors.append(torch.randn(1025, generator=generator, dtype=torch.float64).to(torch.float32).to(device))
    return tensors


def _with_kernel_ineligible_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Add a non-contiguous CUDA view and a CPU tensor: both take the per-tensor op."""
    strided_base = torch.linspace(-2, 2, 64, device="cuda").to(torch.bfloat16)
    return [*tensors, strided_base[::2], torch.linspace(-2, 2, 9).to(torch.bfloat16)]


@pytest.mark.cpu_only
def test_multi_tensor_scaling_on_cpu_applies_the_per_tensor_ops():
    tensors = [torch.linspace(-3, 3, 11).to(torch.bfloat16), torch.linspace(-1, 1, 5), torch.empty(0)]
    factor = torch.tensor(0.3712345678901234, dtype=torch.float64)

    multiplied = [t.clone() for t in tensors]
    multi_tensor_mul_(multiplied, factor)
    divided = [t.clone() for t in tensors]
    multi_tensor_div_(divided, 3.0)

    for original, actual in zip(tensors, multiplied, strict=True):
        assert torch.equal(actual, original.clone().mul_(factor))
    for original, actual in zip(tensors, divided, strict=True):
        assert torch.equal(actual, original.clone().div_(3.0))


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
@pytest.mark.parametrize("factor", [0.3712345678901234, 0.99999, 1.0, 3.0, 1e-30])
@pytest.mark.parametrize("factor_device", ["cuda", "cpu"])
def test_multi_tensor_mul_is_bit_identical_to_per_tensor_mul(factor_device, factor):
    tensors = _with_kernel_ineligible_tensors(_scale_test_tensors("cuda"))
    operand = torch.tensor(factor, dtype=torch.float64, device=factor_device)
    expected = [t.clone().mul_(operand.to(t.device)) for t in tensors]

    multi_tensor_mul_(tensors, operand)

    for actual, reference in zip(tensors, expected, strict=True):
        assert actual.dtype == reference.dtype
        assert torch.equal(actual, reference)


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
@pytest.mark.parametrize("divisor", [0.3712345678901234, 1.0, 3.0, 4.0, 6.0, 64.0])
def test_multi_tensor_div_is_bit_identical_to_per_tensor_div(divisor):
    tensors = _with_kernel_ineligible_tensors(_scale_test_tensors("cuda"))
    expected = [t.clone().div_(divisor) for t in tensors]

    multi_tensor_div_(tensors, divisor)

    for actual, reference in zip(tensors, expected, strict=True):
        assert actual.dtype == reference.dtype
        assert torch.equal(actual, reference)


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
def test_multi_tensor_mul_by_one_keeps_non_finite_values():
    values = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.5]
    tensor = torch.tensor(values, device="cuda").to(torch.bfloat16)
    before = tensor.clone()

    multi_tensor_mul_([tensor], torch.ones((), dtype=torch.float64, device="cuda"))

    assert torch.equal(tensor.isnan(), before.isnan())
    assert torch.equal(tensor.nan_to_num(nan=7.0), before.nan_to_num(nan=7.0))
    assert torch.equal(torch.signbit(tensor), torch.signbit(before))


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
@pytest.mark.parametrize("gradient_scale", [10.0, 1e-3])
def test_clip_grad_norm_impl_fused_scaling_matches_torch_clipping(gradient_scale):
    """Clipped (norm above the threshold) and unclipped (below) gradients equal PyTorch's clip_grads_with_norm_."""
    generator = torch.Generator().manual_seed(7)
    shapes = [(_CHUNK + 3,), (33, 65), (5,)]
    dtypes = [torch.bfloat16, torch.float32, torch.bfloat16]
    parameters = []
    for shape, dtype in zip(shapes, dtypes, strict=True):
        parameter = torch.nn.Parameter(torch.zeros(shape, dtype=dtype, device="cuda"))
        parameter.grad = (torch.randn(shape, generator=generator) * gradient_scale).to(dtype).to("cuda")
        parameters.append(parameter)
    reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    for ref, parameter in zip(reference, parameters, strict=True):
        ref.grad = parameter.grad.clone()
    max_norm = 1.0

    total_norm = _clip_grad_norm_impl(parameters, max_norm)

    expected_norm = sumsq_reference([p.grad for p in reference]).sqrt()
    torch.testing.assert_close(total_norm.cpu(), expected_norm, rtol=1e-12, atol=0)
    assert (expected_norm > max_norm) == (gradient_scale > 1.0)
    torch.nn.utils.clip_grads_with_norm_(reference, max_norm, total_norm, foreach=True)
    for parameter, ref in zip(parameters, reference, strict=True):
        assert torch.equal(parameter.grad, ref.grad)
