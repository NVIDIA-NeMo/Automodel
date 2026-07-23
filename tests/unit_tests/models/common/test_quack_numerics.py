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

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common.utils import (
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.shared.import_utils import safe_import

HAVE_QUACK, _ = safe_import("quack")
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(not HAVE_QUACK, reason="quack-kernels is required"),
]


def _assert_numerically_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Check a low-precision kernel result against its PyTorch reference.

    Args:
        actual: Tensor of arbitrary shape produced by a Quack kernel.
        expected: Tensor of the same shape produced by PyTorch.
    """
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    error = actual_fp32 - expected_fp32
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected_fp32).clamp_min(1e-12)
    relative_max = error.abs().amax() / expected_fp32.abs().amax().clamp_min(1e-12)
    cosine = F.cosine_similarity(actual_fp32.flatten(), expected_fp32.flatten(), dim=0)
    assert relative_l2 < 5e-3
    assert relative_max < 1e-2
    assert cosine > 0.999


@pytest.mark.parametrize(
    ("model_shape", "tokens", "in_features", "out_features", "bias"),
    [
        # One biased, odd-token model shape covers tail and bias gradients without compiling a full shape matrix in L0.
        pytest.param("gpt_oss_q", 257, 2880, 4096, True),
    ],
)
def test_quack_linear_forward_and_gradient_parity(
    model_shape: str,
    tokens: int,
    in_features: int,
    out_features: int,
    bias: bool,
) -> None:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x_data = torch.randn(tokens, in_features, device=device, dtype=dtype)
    weight_data = torch.randn(out_features, in_features, device=device, dtype=dtype) / in_features**0.5
    bias_data = torch.randn(out_features, device=device, dtype=dtype) if bias else None

    reference_x = x_data.detach().clone().requires_grad_(True)
    reference_weight = weight_data.detach().clone().requires_grad_(True)
    reference_bias = bias_data.detach().clone().requires_grad_(True) if bias_data is not None else None
    expected = F.linear(reference_x, reference_weight, reference_bias)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)

    quack_x = x_data.detach().clone().requires_grad_(True)
    quack = initialize_linear_module("quack", in_features, out_features, bias=bias, device=device, dtype=dtype)
    with torch.no_grad():
        quack.weight.copy_(weight_data)
        if bias_data is not None:
            quack.bias.copy_(bias_data)
    actual = quack(quack_x)
    actual.backward(grad_output)

    _assert_numerically_close(actual, expected)
    _assert_numerically_close(quack_x.grad, reference_x.grad)
    _assert_numerically_close(quack.weight.grad, reference_weight.grad)
    if bias:
        _assert_numerically_close(quack.bias.grad, reference_bias.grad)


@pytest.mark.parametrize(
    ("model_shape", "tokens", "hidden_size"),
    [
        # One odd-token model shape covers the RMSNorm tail path without redundant L0 kernel specializations.
        pytest.param("qwen3_moe", 257, 2048),
    ],
)
def test_quack_rms_norm_forward_and_gradient_parity(model_shape: str, tokens: int, hidden_size: int) -> None:
    torch.manual_seed(5678)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6
    x_data = torch.randn(tokens, hidden_size, device=device, dtype=dtype)
    weight_data = torch.randn(hidden_size, device=device, dtype=dtype)

    reference_x = x_data.detach().clone().requires_grad_(True)
    reference_weight = weight_data.detach().clone().requires_grad_(True)
    variance = reference_x.float().square().mean(dim=-1, keepdim=True)
    expected = (reference_x.float() * torch.rsqrt(variance + eps)).to(dtype) * reference_weight
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)

    quack_x = x_data.detach().clone().requires_grad_(True)
    quack = initialize_rms_norm_module("quack", hidden_size, eps=eps, device=device, dtype=dtype)
    with torch.no_grad():
        quack.weight.copy_(weight_data)
    actual = quack(quack_x)
    actual.backward(grad_output)

    _assert_numerically_close(actual, expected)
    _assert_numerically_close(quack_x.grad, reference_x.grad)
    _assert_numerically_close(quack.weight.grad, reference_weight.grad)
