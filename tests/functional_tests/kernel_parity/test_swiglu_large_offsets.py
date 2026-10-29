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

from nemo_automodel.components.moe import swiglu_triton

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or not swiglu_triton.HAVE_TRITON,
        reason="Exercises CUDA pointer arithmetic in the Triton kernels.",
    ),
]


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("rows", [17, (2**31 // 4096) + 1])
def test_clamped_swiglu_large_offsets(rows: int, direction: str) -> None:
    """Validate rows across the signed 32-bit input-offset boundary on CUDA."""
    if rows > 17 and torch.cuda.mem_get_info()[0] < 16 * 2**30:
        pytest.skip("The large-address regression needs 16 GiB of free CUDA memory.")
    width = 4096
    inter = width // 2
    x = torch.full((rows, width), 0.25, dtype=torch.bfloat16, device="cuda")
    probs = torch.full((rows, 1), 0.5, dtype=torch.float32, device="cuda")
    indices = torch.tensor(sorted({0, rows // 2, rows - 2, rows - 1}), device="cuda")
    # Make a missing or wrapped tail row distinguishable from the first row.
    x[-1, 0] = 12.0
    x[-1, inter] = -12.0
    probs[-1] = 0.75
    ref_x = x.index_select(0, indices).float().requires_grad_()
    ref_probs = probs.index_select(0, indices).requires_grad_()
    gate, up = ref_x.chunk(2, dim=-1)
    ref_y = F.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0) * ref_probs
    if direction == "forward":
        actual = swiglu_triton._forward(x, probs, 10.0)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.index_select(0, indices).float(), ref_y.to(x.dtype).float(), rtol=0, atol=0)
    else:
        grad_y = torch.full((rows, inter), 0.125, dtype=x.dtype, device=x.device)
        grad_x, grad_probs = swiglu_triton._backward(x, probs, grad_y, 10.0)
        torch.cuda.synchronize()
        ref_y.backward(torch.full_like(ref_y, 0.125))
        torch.testing.assert_close(
            grad_x.index_select(0, indices).float(), ref_x.grad.to(x.dtype).float(), rtol=1e-2, atol=1e-3
        )
        torch.testing.assert_close(grad_probs.index_select(0, indices), ref_probs.grad, rtol=1e-4, atol=1e-3)
