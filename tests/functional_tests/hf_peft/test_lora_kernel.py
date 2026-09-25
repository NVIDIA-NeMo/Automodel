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

"""Scheduled GPU parity coverage for the Triton LoRA kernels."""

import pytest
import torch

from nemo_automodel.components._peft.lora import LoRATritonFunction
from nemo_automodel.components._peft.lora_kernel import lora_da_dx_update_wrapper, lora_db_update_wrapper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="LoRA Triton kernels require CUDA")


def set_up_inputs(
    bs=1, seq_len=64, h=2048, lora_dim=16, scale=32, h2=1024, dtype: torch.dtype = torch.float32, device: str = "cuda"
):
    x = 0.05 * torch.randn((bs * seq_len, h), dtype=dtype, device=device)
    lora_a = 0.05 * torch.randn((lora_dim, h), dtype=dtype, device=device)
    lora_b = 0.05 * torch.randn((h2, lora_dim), dtype=dtype, device=device)
    dy = 0.001 * torch.randn((bs * seq_len, h2), dtype=dtype).to(device)

    return x, lora_a, lora_b, scale, dy


TEST_PARAMS = [
    {"bs": 1, "seq_len": 128, "h": 512, "lora_dim": 16, "scale": 32, "h2": 256, "dtype": torch.float32},
    {"bs": 1, "seq_len": 128, "h": 512, "lora_dim": 16, "scale": 32, "h2": 256, "dtype": torch.float16},
    {"bs": 1, "seq_len": 128, "h": 512, "lora_dim": 16, "scale": 32, "h2": 256, "dtype": torch.bfloat16},
    {"bs": 1, "seq_len": 100, "h": 500, "lora_dim": 8, "scale": 32, "h2": 224, "dtype": torch.float32},
    {"bs": 1, "seq_len": 100, "h": 500, "lora_dim": 8, "scale": 32, "h2": 224, "dtype": torch.float16},
    {"bs": 1, "seq_len": 100, "h": 500, "lora_dim": 8, "scale": 32, "h2": 224, "dtype": torch.bfloat16},
]


@pytest.mark.parametrize("lora_params", TEST_PARAMS)
def test_forward_kernel(lora_params):
    dtype = lora_params["dtype"]
    x, lora_a, lora_b, scale, _ = set_up_inputs(**lora_params)
    triton_fwd = LoRATritonFunction.apply(x, lora_a, lora_b, scale, dtype)
    baseline_fwd = torch.matmul(torch.matmul(x, lora_a.t()), lora_b.t()) * scale
    assert torch.allclose(triton_fwd, baseline_fwd, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("lora_params", TEST_PARAMS)
def test_da_dx_kernel(lora_params):
    dtype = lora_params["dtype"]
    x, lora_a, lora_b, scale, dy = set_up_inputs(**lora_params)
    dyb = torch.matmul(dy, lora_b)
    baseline_dlora_a = torch.matmul(x.t(), dyb) * scale
    baseline_dx = torch.matmul(dyb, lora_a) * scale

    triton_dlora_a, triton_x = lora_da_dx_update_wrapper(x.t(), dy, lora_b, lora_a, scale, dtype=dtype)
    assert torch.allclose(baseline_dlora_a, triton_dlora_a, atol=6e-2, rtol=6e-2)
    assert torch.allclose(baseline_dx, triton_x, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("lora_params", TEST_PARAMS)
def test_db_kernel(lora_params):
    dtype = lora_params["dtype"]
    x, lora_a, _, scale, dy = set_up_inputs(**lora_params)
    baseline_dlora_b = torch.matmul(dy.t(), torch.matmul(x, lora_a.t())) * scale
    triton_dlora_b = lora_db_update_wrapper(lora_a, x.t(), dy, scale, dtype=dtype)
    assert torch.allclose(baseline_dlora_b, triton_dlora_b, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("lora_dtype", [torch.bfloat16, torch.float16])
def test_memory_efficient_lora_declines_triton_on_mixed_dtype(lora_dtype):
    """Mixed-dtype LoRA must fall back off the Triton kernels and match the torch path exactly.

    The kernels hand their operands to ``tl.dot``, which asserts one dtype, and they run outside
    autocast — so the FSDP2 ``output_dtype=float32`` layout from issue #3652 (FP32 activations,
    BF16 adapters) used to abort compilation with "Both operands must be same dtype".

    This calls ``apply_memory_efficient_lora`` directly rather than going through
    ``apply_lora_to_linear_modules``: ``patch_linear_module`` force-disables Triton whenever
    TransformerEngine is importable, and the CI image builds TE, so a recipe-level test could never
    reach the kernels here.
    """
    from nemo_automodel.components._peft.lora import apply_memory_efficient_lora

    torch.manual_seed(0)
    scale = 2.5  # != 1.0 so a dropped scale cannot hide
    x = torch.randn(64, 128, dtype=torch.float32, device="cuda")
    lora_a = (0.05 * torch.randn(16, 128, device="cuda")).to(lora_dtype)
    lora_b = (0.05 * torch.randn(96, 16, device="cuda")).to(lora_dtype)
    grad_out = torch.randn(64, 96, dtype=torch.float32, device="cuda")

    def run(use_triton_kernel):
        xg = x.clone().requires_grad_(True)
        a = lora_a.clone().requires_grad_(True)
        b = lora_b.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=lora_dtype):
            out = apply_memory_efficient_lora(xg, a, b, scale, use_triton_kernel)
        (out.float() * grad_out).sum().backward()
        return out.float(), xg.grad, a.grad, b.grad

    triton_path = run(True)  # must not raise: the kernels are declined for mixed dtype
    torch_path = run(False)

    assert triton_path[1].dtype == torch.float32
    for name, got, want in zip(("out", "d_x", "d_lora_A", "d_lora_B"), triton_path, torch_path):
        # Declining means literally running the torch path, so this is exact, not approximate.
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"{name} diverged from the torch path")
