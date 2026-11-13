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

"""Inkling 2 expert dequantization of CPU checkpoint tensors on a CUDA device."""

import pytest
import torch

from nemo_automodel.components.models.inkling2.quantization import (
    dequantize_mxfp8_experts,
    dequantize_nvfp4_experts,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_checkpoint_tensors_decode_on_cuda_like_on_cpu():
    generator = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (3, 8, 32), generator=generator, dtype=torch.uint8)
    nvfp4_scale = torch.randint(0, 0x7F, (3, 8, 4), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn)
    amax = torch.tensor([0.1, 0.2, 0.3])
    denominator = torch.tensor([2688.0])
    on_cuda = dequantize_nvfp4_experts(packed, nvfp4_scale, amax, denominator, device="cuda")
    assert on_cuda.device.type == "cuda"
    assert torch.equal(on_cuda.cpu(), dequantize_nvfp4_experts(packed, nvfp4_scale, amax, denominator))

    fp8 = torch.randint(0, 0x7F, (3, 8, 64), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn)
    exponents = torch.randint(100, 150, (3, 8, 2), generator=generator, dtype=torch.uint8)
    on_cuda = dequantize_mxfp8_experts(fp8, exponents, device=torch.device("cuda"))
    assert on_cuda.device.type == "cuda"
    assert torch.equal(on_cuda.cpu(), dequantize_mxfp8_experts(fp8, exponents))
