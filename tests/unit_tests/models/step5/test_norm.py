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

"""Independent normalization and precision-boundary controls for Step-5."""

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5.norm import Step5RMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compile_norm", [False, True])
def test_norm_cpu_independent_oracle_and_gradients(dtype: torch.dtype, compile_norm: bool) -> None:
    torch.manual_seed(981)
    module = Step5RMSNorm(192, compile_norm=compile_norm)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192) + 0.00123)
    hidden = torch.randn(2, 17, 192, dtype=dtype, requires_grad=True)
    reference = F.rms_norm(hidden.float(), (192,), module.weight + 1, module.variance_epsilon).to(dtype)
    actual = module(hidden)
    upstream = torch.randn_like(hidden)
    expected_grads = torch.autograd.grad(reference, (hidden, module.weight), upstream)
    actual_grads = torch.autograd.grad(actual, (hidden, module.weight), upstream)
    for observed, expected in [(actual, reference), *zip(actual_grads, expected_grads)]:
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=2e-6)
    assert actual.dtype == dtype
    assert module.weight.dtype == torch.float32
    assert list(module.state_dict()) == ["weight"]


def test_norm_preserves_fp32_weight_and_residual_boundaries() -> None:
    torch.manual_seed(982)
    module = Step5RMSNorm(192)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192) + 0.00123)
    residual = torch.randn(2, 17, 192, dtype=torch.float32)
    correct = module(residual).bfloat16()
    early_cast = module(residual.bfloat16())
    rounded_weight = F.rms_norm(
        residual, (192,), module.weight.bfloat16().float() + 1, module.variance_epsilon
    ).bfloat16()
    assert not torch.equal(correct, early_cast)
    assert not torch.equal(correct, rounded_weight)
    assert torch.equal(module(residual), Step3p5RMSNorm.forward(module, residual))


def test_residual_norm_cast_keeps_input_and_parameter_gradients() -> None:
    torch.manual_seed(986)
    module = Step5RMSNorm(192, compile_norm=True)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192) + 0.00123)
    residual = torch.randn(2, 17, 192, requires_grad=True)
    expected = Step3p5RMSNorm.forward(module, residual).bfloat16()
    actual = module(residual, output_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (residual, module.weight), upstream)
    actual_grads = torch.autograd.grad(actual, (residual, module.weight), upstream)
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    assert residual.dtype == module.weight.dtype == torch.float32
