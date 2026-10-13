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

"""H100/SM100 execution coverage for compiled M3 normalization."""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3RMSNorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for compiled norm execution")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("hidden", [128, 6144])
def test_compiled_norm_cuda_forward_and_gradients(dtype: torch.dtype, gemma: bool, hidden: int) -> None:
    torch.manual_seed(19)
    eager = MiniMaxM3RMSNorm(hidden, gemma=gemma).cuda()
    compiled = MiniMaxM3RMSNorm(hidden, gemma=gemma, compile_norm=True).cuda()
    with torch.no_grad():
        eager.weight.uniform_(-0.5, 0.5)
        compiled.weight.copy_(eager.weight)
    x = torch.randn(2, 129, hidden, device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    grad = torch.randn_like(x)
    expected = eager(x)
    actual = compiled(y)
    expected_dx, expected_dw = torch.autograd.grad(expected, (x, eager.weight), grad)
    actual_dx, actual_dw = torch.autograd.grad(actual, (y, compiled.weight), grad)
    # Both paths accumulate in FP32; only reduction ordering can differ. BF16
    # input gradients round after that reduction, so allow one representable step.
    atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (0.032, 0.01)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_dw, expected_dw, rtol=2e-4, atol=2e-4)
