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

"""Single-GPU reference QDQ/STE and real-model LoRA QAT integration.

Requires a CUDA-enabled PyTorch build and GPU; BF16 cases additionally require
native BF16 hardware. These test ordinary PyTorch arithmetic and packed-byte
reference reload, NOT a native FP4/FP8 kernel or serving backend. Multi-rank
launches fail explicitly: no distributed training support is claimed here.
"""

import os
from pathlib import Path
from typing import Literal

import pytest
import torch
import torch.distributed as dist

from nemo_automodel.components.quantization.weight_qat import WeightQuantizationConfig
from tests.unit_tests._transformers.test_lora_qat_models import run_tiny_model_roundtrip

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-enabled PyTorch and a GPU"),
    pytest.mark.timeout(60),
]
DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(
        torch.bfloat16,
        id="bf16",
        marks=pytest.mark.skipif(
            torch.cuda.is_available() and not torch.cuda.is_bf16_supported(including_emulation=False),
            reason="requires a GPU with native BF16 support",
        ),
    ),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("architecture", ["glm5_next", "deepseek_v4"])
@pytest.mark.parametrize("fp8_block_size", [32, 128])
@pytest.mark.parametrize("target_experts", [False, True], ids=["dense", "dense-and-experts"])
def test_tiny_lora_qat_models_cuda(
    tmp_path: Path, dtype: torch.dtype, architecture: str, fp8_block_size: int, target_experts: bool
) -> None:
    run_tiny_model_roundtrip(
        architecture,
        dtype=dtype,
        device=torch.device("cuda"),
        fp8_block_size=fp8_block_size,
        directory=tmp_path,
        target_experts=target_experts,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("architecture", ["glm5_next", "deepseek_v4"])
def test_tiny_expert_only_qat_with_attention_lora_cuda(tmp_path: Path, dtype: torch.dtype, architecture: str) -> None:
    run_tiny_model_roundtrip(
        architecture,
        dtype=dtype,
        device=torch.device("cuda"),
        fp8_block_size=32,
        directory=tmp_path,
        qat_experts_only=True,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "qat_experts_only", [False, True], ids=["all-trained-targets-qat", "independent-experts-only-qat"]
)
@pytest.mark.parametrize(
    "architecture,expert_format",
    [
        pytest.param("glm5_next", "fp8", id="glm5.3-flash-fp8-128-sparse-only"),
        pytest.param("deepseek_v4", "mxfp4", id="deepseek-v4-mxfp4"),
    ],
)
def test_tiny_checkpoint_format_lora_qat_cuda(
    tmp_path: Path,
    dtype: torch.dtype,
    architecture: str,
    expert_format: Literal["mxfp4", "fp8"],
    qat_experts_only: bool,
) -> None:
    """Check checkpoint expert formats with optional dense FP8 on CUDA.

    GLM here remains one-layer sparse-only. The separate CPU hybrid case
    exercises production KDA without implying native GPU KDA coverage.
    """
    run_tiny_model_roundtrip(
        architecture,
        dtype=dtype,
        device=torch.device("cuda"),
        fp8_block_size=128,
        directory=tmp_path,
        expert_format=expert_format,
        qat_experts_only=qat_experts_only,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(128, 256), (2, 128, 256)], ids=["dense", "grouped"])
@pytest.mark.parametrize(
    "config",
    [
        WeightQuantizationConfig("fp8", (32, 32)),
        WeightQuantizationConfig("fp8", (128, 128)),
        WeightQuantizationConfig("mxfp4", (1, 32)),
    ],
    ids=["fp8-block32", "fp8-block128", "mxfp4-block1x32"],
)
def test_cuda_reference_packing_and_exact_random_ste(
    dtype: torch.dtype, shape: tuple[int, ...], config: WeightQuantizationConfig
) -> None:
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "LoRA QAT integration supports single rank only"
    assert not dist.is_initialized() or dist.get_world_size() == 1, "LoRA QAT integration supports single rank only"
    torch.manual_seed(71)
    # Noncontiguous canonical [..., out, in] weights exercise device movement
    # and grouped packing without depending on any native low-bit GEMM backend.
    weight = torch.randn((*shape[:-2], shape[-1], shape[-2]), device="cuda", dtype=dtype)
    weight = weight.transpose(-2, -1).requires_grad_()
    source = weight.detach().clone()
    quantizer = config.build().to(device="cuda", dtype=dtype)
    quantizer.eval()
    packed = quantizer.quantize(weight)
    decoded = packed.dequantize(dtype=dtype)
    evaluated = quantizer(weight)
    assert evaluated.dtype == dtype and evaluated.device == weight.device
    assert torch.isfinite(evaluated).all()
    torch.testing.assert_close(evaluated, decoded, rtol=0, atol=0)
    assert packed.payload.dtype == packed.scales.dtype == torch.uint8
    assert packed.payload.device == packed.scales.device == weight.device
    assert packed.payload.shape == (shape if config.format == "fp8" else (*shape[:-1], shape[-1] // 2))
    rows, cols = config.block_size
    assert packed.scales.shape == (*shape[:-2], shape[-2] // rows, shape[-1] // cols)
    cpu_packed = config.build().quantize(source.cpu())
    assert torch.equal(packed.payload.cpu(), cpu_packed.payload)
    assert torch.equal(packed.scales.cpu(), cpu_packed.scales)
    upstream = torch.randn_like(weight)
    quantizer.train()
    gradients = []
    for _ in range(2):
        output = quantizer(weight)
        torch.testing.assert_close(output, decoded, rtol=0, atol=0)
        (gradient,) = torch.autograd.grad(output, weight, grad_outputs=upstream)
        assert torch.equal(gradient, upstream)
        gradients.append(gradient)
    assert torch.equal(gradients[0], gradients[1])
    assert torch.equal(weight, source)
    assert not quantizer.state_dict()
