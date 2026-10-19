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

"""Masked attention backend selection and forward/backward numerical parity."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.components.attention.utils import initialize_attn_module_and_func


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.timeout(30)
def test_masked_sdpa_backward_matches_fp32(device: str) -> None:
    """Avoid the unsafe cuDNN path without changing attention or global flags."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sequence, valid, heads, head_dim = 126, 33, 64, 256
    generator = torch.Generator().manual_seed(1234)
    q = torch.randn((1, heads, sequence, head_dim), generator=generator).to(dtype)
    k = torch.randn((1, 4, sequence, head_dim), generator=generator).to(dtype)
    v = torch.randn(k.shape, generator=generator).to(dtype)
    mask = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    mask[:, valid:] = False
    mask = mask[None, None]
    upstream = (0.02 * torch.randn(q.shape, generator=generator)).to(dtype)
    upstream[:, :, valid:] = 0
    upstream = upstream.transpose(1, 2).contiguous().transpose(1, 2)

    reference_inputs = [tensor.float().requires_grad_() for tensor in (q, k, v)]
    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(*reference_inputs, attn_mask=mask, enable_gqa=True)
        reference.backward(upstream.float())
    expected = [reference.detach(), *(tensor.grad for tensor in reference_inputs)]
    assert all(torch.isfinite(tensor).all() for tensor in expected)

    inputs = [tensor.to(device).detach().requires_grad_() for tensor in (q, k, v)]
    _, attention = initialize_attn_module_and_func("sdpa", heads, head_dim, head_dim, head_dim**-0.5, num_gqa_groups=4)
    cudnn_enabled = torch.backends.cuda.cudnn_sdp_enabled()
    output = attention(*inputs, attn_mask=mask.to(device), is_causal=False)
    assert torch.backends.cuda.cudnn_sdp_enabled() == cudnn_enabled
    if device == "cuda":
        assert "CudnnAttentionBackward" not in type(output.grad_fn).__name__
    output.backward(upstream.to(device))
    actual = [output.detach(), *(tensor.grad for tensor in inputs)]
    assert all(torch.isfinite(tensor).all() for tensor in actual)
    for got, want in zip(actual, expected, strict=True):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(
            got.float().cpu(), want, atol=4e-3 if device == "cuda" else 1e-6, rtol=2e-2 if device == "cuda" else 1e-6
        )
