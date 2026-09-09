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

from nemo_automodel.components.moe.experts import _BIAS_GRAD_TRITON_AVAILABLE, _apply_bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_has_deterministic_bf16_bias_gradient():
    """Imbalanced BF16 routing produces the same trainable bias gradient on every CUDA backward."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    n_experts = 64
    n_tokens = 4096
    hidden = 512

    torch.manual_seed(1234)
    value = torch.randn(n_tokens, hidden, dtype=torch.bfloat16, device=device)
    bias_data = torch.randn(n_experts, hidden, dtype=torch.bfloat16, device=device)
    tokens_per_expert = torch.zeros(n_experts, dtype=torch.long, device=device)
    tokens_per_expert[0] = n_tokens
    upstream_grad = torch.randn_like(value)

    expected_grad = torch.stack(
        [segment.double().sum(dim=0) for segment in torch.split(upstream_grad, tokens_per_expert.tolist())]
    ).to(torch.bfloat16)

    first_grad = None
    for _ in range(10):
        bias = bias_data.clone().requires_grad_()
        result = _apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert)
        result.backward(upstream_grad)

        assert bias.grad is not None
        torch.testing.assert_close(bias.grad, expected_grad, rtol=0, atol=0)
        if first_grad is None:
            first_grad = bias.grad.clone()
        else:
            torch.testing.assert_close(bias.grad, first_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_preserves_fp32_probability_weighting_in_bias_gradient():
    """FP32 probability weighting remains FP32 until the BF16 bias gradient is reduced."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    n_experts = 64
    n_tokens = 4096
    hidden = 512

    torch.manual_seed(1234)
    value = torch.randn(n_tokens, hidden, dtype=torch.bfloat16, device=device)
    bias_data = torch.randn(n_experts, hidden, dtype=torch.bfloat16, device=device)
    permuted_probs = torch.rand(n_tokens, 1, dtype=torch.float32, device=device)
    tokens_per_expert = torch.zeros(n_experts, dtype=torch.long, device=device)
    tokens_per_expert[0] = n_tokens
    upstream_grad = torch.randn_like(value)

    weighted_grad = (upstream_grad.float() * permuted_probs).double()
    expected_grad = torch.stack(
        [segment.sum(dim=0) for segment in torch.split(weighted_grad, tokens_per_expert.tolist())]
    ).to(torch.bfloat16)

    bias = bias_data.clone().requires_grad_()
    result = _apply_bias(
        value,
        bias=bias,
        tokens_per_expert=tokens_per_expert,
        permuted_probs=permuted_probs,
    )
    result.backward(upstream_grad)

    assert result.dtype == torch.bfloat16
    assert bias.grad is not None
    assert bias.grad.dtype == torch.bfloat16
    torch.testing.assert_close(bias.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_triton_gradient_matches_fp64_under_cancellation():
    """Cancellation-heavy weighted gradients round to the FP64 mathematical reference."""
    assert _BIAS_GRAD_TRITON_AVAILABLE
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    n_experts = 4
    n_tokens = 4096
    hidden = 64

    value = torch.zeros(n_tokens, hidden, dtype=torch.bfloat16, device=device)
    bias = torch.zeros(n_experts, hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
    tokens_per_expert = torch.tensor([0, n_tokens, 0, 0], dtype=torch.long, device=device)
    permuted_probs = torch.ones(n_tokens, 1, dtype=torch.float32, device=device)
    permuted_probs[n_tokens // 2 :] = 0.9999
    upstream_grad = torch.ones(n_tokens, hidden, dtype=torch.bfloat16, device=device)
    upstream_grad[n_tokens // 2 :] = -1

    _apply_bias(value, bias, tokens_per_expert, permuted_probs).backward(upstream_grad)

    expected_grad = torch.zeros_like(bias)
    expected_grad[1] = (upstream_grad.float() * permuted_probs).double().sum(dim=0).to(torch.bfloat16)
    assert bias.grad is not None
    torch.testing.assert_close(bias.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_triton_handles_block_edges_empty_experts_and_noncontiguous_grad():
    """Uneven segments exercise every reduction mask against an FP64 oracle."""
    assert _BIAS_GRAD_TRITON_AVAILABLE
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    hidden = 40
    tokens_per_expert = torch.tensor([0, 1, 127, 128, 129, 4096, 17, 0], dtype=torch.long, device=device)
    n_tokens = int(tokens_per_expert.sum())

    torch.manual_seed(888)
    value = torch.randn(n_tokens, hidden, dtype=torch.bfloat16, device=device)
    bias = torch.randn(tokens_per_expert.numel(), hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
    permuted_probs = torch.rand(n_tokens, 1, dtype=torch.float32, device=device)
    upstream_grad = torch.randn(hidden, n_tokens, dtype=torch.bfloat16, device=device).transpose(0, 1)
    assert not upstream_grad.is_contiguous()

    _apply_bias(value, bias, tokens_per_expert, permuted_probs).backward(upstream_grad)

    weighted_grad = (upstream_grad.float() * permuted_probs).double()
    expected_grad = torch.stack(
        [segment.sum(dim=0) for segment in torch.split(weighted_grad, tokens_per_expert.tolist())]
    ).to(torch.bfloat16)
    assert bias.grad is not None
    torch.testing.assert_close(bias.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_triton_backward_is_inductor_fullgraph_compatible():
    """The weighted Triton backward remains inside an Inductor full graph."""
    assert _BIAS_GRAD_TRITON_AVAILABLE
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tokens_per_expert = torch.tensor([0, 2, 6, 1], dtype=torch.long, device=device)

    torch.manual_seed(7)
    value = torch.randn(9, 8, dtype=torch.bfloat16, device=device, requires_grad=True)
    bias = torch.randn(4, 8, dtype=torch.bfloat16, device=device, requires_grad=True)
    permuted_probs = torch.rand(9, 1, dtype=torch.float32, device=device, requires_grad=True)
    upstream_grad = torch.randn_like(value)
    compiled_apply_bias = torch.compile(_apply_bias, fullgraph=True)

    compiled_apply_bias(value, bias, tokens_per_expert, permuted_probs).backward(upstream_grad)

    assert value.grad is not None
    assert bias.grad is not None
    assert permuted_probs.grad is not None
