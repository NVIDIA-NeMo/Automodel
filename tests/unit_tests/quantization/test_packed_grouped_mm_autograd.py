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

"""Exercise packed-weight autograd contexts without requiring a CUDA grouped GEMM."""

import pytest
import torch

from nemo_automodel.components.quantization import mxfp4, nvfp4

pytestmark = pytest.mark.cpu_only


def _grouped_mm_cpu(x: torch.Tensor, weights: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """Use CPU matmuls for contiguous expert groups.

    Args:
        x: Tensor of shape [tokens, in_dim].
        weights: Tensor of shape [experts, in_dim, out_dim].
        offs: Tensor of shape [experts], with cumulative token counts.

    Returns:
        Tensor of shape [tokens, out_dim].
    """
    chunks = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        chunks.append(x[start:end] @ weights[expert])
        start = end
    return torch.cat(chunks)


@pytest.mark.parametrize("weight_format", ["mxfp4", "nvfp4"])
def test_packed_grouped_mm_input_gradients_on_cpu(monkeypatch: pytest.MonkeyPatch, weight_format: str) -> None:
    """Autograd restores the packed tensors and returns the correct input gradient."""
    monkeypatch.setattr(torch, "_grouped_mm", _grouped_mm_cpu)
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(2, 8, 32, generator=generator)
    x = torch.randn(5, 32, generator=generator, requires_grad=True)
    offs = torch.tensor([2, 5], dtype=torch.int32)

    if weight_format == "mxfp4":
        packed, scales = mxfp4.quantize_mxfp4(weight)
        decoded = mxfp4.dequantize_mxfp4(packed, scales, x.dtype)
        output = mxfp4.MXFP4GroupedMM.apply(x, packed, scales, offs)
    else:
        packed, scales, global_scale = nvfp4.quantize_nvfp4(weight)
        decoded = nvfp4.dequantize_nvfp4(packed, scales, global_scale, x.dtype)
        output = nvfp4.NVFP4GroupedMM.apply(x, packed, scales, global_scale, offs)

    grad_output = torch.randn(output.shape, generator=generator)
    output.backward(grad_output)
    reference = torch.cat([x[:2].detach() @ decoded[0].T, x[2:].detach() @ decoded[1].T])
    expected_gradient = torch.cat([grad_output[:2] @ decoded[0], grad_output[2:] @ decoded[1]])

    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(x.grad, expected_gradient)
