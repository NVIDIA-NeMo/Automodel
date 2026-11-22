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

"""Execute Gemma4's packed sliding mask through the CUDA FlexAttention kernel."""

from collections.abc import Callable, Iterator

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

from nemo_automodel.components.models.gemma4_moe.model import _build_packed_gemma4_causal_mask_mapping


@pytest.fixture
def _ieee_matmul() -> Iterator[None]:
    """Temporarily use IEEE math for the FP32 kernel comparison."""
    # Isolate mask semantics from the application's default TF32 precision.
    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")
@pytest.mark.usefixtures("_ieee_matmul")
def test_packed_long_image_flex_forward_backward(record_property: Callable[[str, float], None]) -> None:
    """Compare the real Flex kernel and Q/K/V gradients with a math-SDPA oracle."""
    seq_len, document_boundary, image_start, image_end, window = 1152, 1124, 2, 1122, 1024
    packed = torch.ones(1, seq_len, dtype=torch.long, device="cuda")
    packed[:, document_boundary:] = 2
    mm_types = torch.zeros_like(packed)
    mm_types[:, image_start:image_end] = 1
    block_mask = _build_packed_gemma4_causal_mask_mapping(
        packed, mm_types, dtype=torch.float32, sliding_window=window, as_block_mask=True
    )["sliding_attention"]

    # Enumerate contiguous key intervals, independently of the mask callback.
    expected_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for query in range(seq_len):
        document_start = 0 if query < document_boundary else document_boundary
        first_key = max(document_start, query - window + 1)
        last_key = image_end if image_start <= query < image_end else query + 1
        expected_mask[query, first_key:last_key] = True
    expected_mask = expected_mask[None, None].cuda()

    generator = torch.Generator(device="cuda").manual_seed(42)
    query, key, value = (
        (torch.randn(1, 2, seq_len, 16, generator=generator, device="cuda") * 0.2).requires_grad_() for _ in range(3)
    )
    grad_output = torch.randn(query.shape, generator=generator, device="cuda")
    actual = torch.compile(flex_attention, fullgraph=True)(query, key, value, block_mask=block_mask)
    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(query, key, value, attn_mask=expected_mask)
    actual_grads = torch.autograd.grad(actual, (query, key, value), grad_output)
    reference_grads = torch.autograd.grad(reference, (query, key, value), grad_output)
    record_property("output_max_abs", (actual - reference).abs().max().item())
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    for name, actual_grad, reference_grad in zip(("dq", "dk", "dv"), actual_grads, reference_grads):
        record_property(f"{name}_max_abs", (actual_grad - reference_grad).abs().max().item())
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-4, atol=1e-5)
