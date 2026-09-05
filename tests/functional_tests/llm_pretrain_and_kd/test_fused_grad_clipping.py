# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.components.training._fused_grad_clipping import (
    can_use_fused_grad_norm,
    fused_multi_tensor_max,
    fused_multi_tensor_scaled_l2,
)
from nemo_automodel.components.training.utils import _clip_grad_norm_impl


def test_can_use_fused_grad_norm():
    # CPU should always return False
    assert not can_use_fused_grad_norm(torch.device("cpu"))


def test_cpu_fallback_clip_grad_norm():
    """Verify that _clip_grad_norm_impl functions correctly on CPU via fallback."""
    torch.manual_seed(42)
    p1 = torch.nn.Parameter(torch.randn(10, 10))
    p1.grad = torch.randn(10, 10)
    p2 = torch.nn.Parameter(torch.randn(5))
    p2.grad = torch.randn(5)

    # Reference norm computed before clipping modifies p.grad in-place
    ref_norm = torch.linalg.vector_norm(torch.cat([p1.grad.flatten(), p2.grad.flatten()]), ord=2)

    norm = _clip_grad_norm_impl([p1, p2], max_norm=1.0, norm_type=2.0)
    assert norm > 0
    assert torch.isfinite(norm)
    torch.testing.assert_close(norm, ref_norm.to(dtype=torch.float64))


def test_empty_and_zero_numel():
    """Verify behavior on empty gradients and zero-element tensors."""
    p1 = torch.nn.Parameter(torch.empty(0))
    p1.grad = torch.empty(0)

    norm = _clip_grad_norm_impl([p1], max_norm=1.0)
    assert norm.item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_grad_clipping_cuda_parity():
    """Verify numerical parity of fused Triton kernels against PyTorch reference on CUDA."""
    device = torch.device("cuda:0")
    if not can_use_fused_grad_norm(device):
        pytest.skip("Triton not available on CUDA")

    torch.manual_seed(42)
    shapes = [(128, 128), (256,), (64, 512), (1024,)]
    params_bf16 = []
    params_fp32 = []

    for i, s in enumerate(shapes):
        p = torch.nn.Parameter(torch.randn(s, dtype=torch.bfloat16, device=device))
        p.grad = torch.randn(s, dtype=torch.bfloat16, device=device)
        params_bf16.append(p)

        p2 = torch.nn.Parameter(torch.randn(s, dtype=torch.float32, device=device))
        p2.grad = torch.randn(s, dtype=torch.float32, device=device)
        params_fp32.append(p2)

    all_params = params_bf16 + params_fp32
    ref_norm = torch.linalg.vector_norm(
        torch.cat([p.grad.float().flatten() for p in all_params]),
        ord=2,
    )

    fused_norm = _clip_grad_norm_impl(all_params, max_norm=1.0, norm_type=2.0)
    torch.testing.assert_close(fused_norm, ref_norm.to(dtype=torch.float64), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_multi_tensor_helpers():
    """Directly test fused_multi_tensor_max and fused_multi_tensor_scaled_l2 on CUDA."""
    device = torch.device("cuda:0")
    if not can_use_fused_grad_norm(device):
        pytest.skip("Triton not available on CUDA")

    grads = [
        torch.randn(100, 200, dtype=torch.bfloat16, device=device),
        torch.randn(50, dtype=torch.float32, device=device),
    ]

    # Test max
    ref_max = max(g.abs().max().item() for g in grads)
    fused_max = fused_multi_tensor_max(grads, device).item()
    assert abs(ref_max - fused_max) < 1e-4

    # Test scaled L2
    scale = torch.tensor(fused_max, dtype=torch.float64, device=device)
    ref_sq = sum((g.float() / float(scale)).square().sum().item() for g in grads)
    fused_sq = fused_multi_tensor_scaled_l2(grads, scale, device).item()
    assert abs(ref_sq - fused_sq) / ref_sq < 1e-4
