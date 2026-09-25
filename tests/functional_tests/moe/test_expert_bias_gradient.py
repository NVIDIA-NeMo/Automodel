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

from nemo_automodel.components.moe.experts import (
    _BIAS_GRAD_TRITON_AVAILABLE,
    _apply_bias,
    _checkpointed_chunked_expert_mlp,
)


def _quick_geglu(value: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Apply the GPT-OSS expert activation used by the parity tests.

    Args:
        value: Tensor of shape [tokens, 2 * intermediate] in concatenated gate/up layout.
        probs: Tensor of shape [tokens, 1] containing routing probabilities.

    Returns:
        Tensor of shape [tokens, intermediate].
    """
    gate, up = value.chunk(2, dim=-1)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0) * probs).to(value.dtype)


def _full_grouped_expert_mlp(
    hidden_states,
    gate_and_up_projs,
    down_projs,
    gate_up_proj_bias,
    down_proj_bias,
    tokens_per_expert,
    permuted_probs,
    apply_router_weight_after_down,
):
    """Compute the unchunked reference expert MLP.

    Args:
        hidden_states: Tensor of shape [tokens, hidden], grouped contiguously by expert.
        gate_and_up_projs: Tensor of shape [experts, hidden, 2 * intermediate].
        down_projs: Tensor of shape [experts, intermediate, hidden].
        gate_up_proj_bias: Optional tensor of shape [experts, 2 * intermediate].
        down_proj_bias: Optional tensor of shape [experts, hidden].
        tokens_per_expert: Tensor of shape [experts] containing contiguous row counts.
        permuted_probs: Tensor of shape [tokens, 1] containing routing probabilities.
        apply_router_weight_after_down: Whether to apply routing probabilities after the down projection.

    Returns:
        Tensor of shape [tokens, hidden].
    """
    offs = tokens_per_expert.cumsum(dim=0).to(torch.int32)
    gate_up = torch._grouped_mm(hidden_states, gate_and_up_projs, offs=offs)
    gate_up = _apply_bias(gate_up, gate_up_proj_bias, tokens_per_expert)
    activation_probs = torch.ones_like(permuted_probs) if apply_router_weight_after_down else permuted_probs
    activated = _quick_geglu(gate_up, activation_probs)
    output = torch._grouped_mm(activated, down_projs, offs=offs)
    output = _apply_bias(
        output,
        down_proj_bias,
        tokens_per_expert,
        None if apply_router_weight_after_down else permuted_probs,
    )
    if apply_router_weight_after_down:
        output = (output.float() * permuted_probs.float()).to(hidden_states.dtype)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_has_deterministic_bf16_bias_gradient():
    """Imbalanced BF16 routing produces the same trainable bias gradient on every CUDA backward."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    n_experts = 64
    n_tokens = 16384
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
    n_tokens = 16384
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
    tokens_per_expert = torch.tensor([0, 4096, 8192, 1], dtype=torch.long, device=device)
    n_tokens = int(tokens_per_expert.sum())

    torch.manual_seed(7)
    value = torch.randn(n_tokens, 8, dtype=torch.bfloat16, device=device, requires_grad=True)
    bias = torch.randn(4, 8, dtype=torch.bfloat16, device=device, requires_grad=True)
    permuted_probs = torch.rand(n_tokens, 1, dtype=torch.float32, device=device, requires_grad=True)
    upstream_grad = torch.randn_like(value)
    compiled_apply_bias = torch.compile(_apply_bias, fullgraph=True)

    compiled_apply_bias(value, bias, tokens_per_expert, permuted_probs).backward(upstream_grad)

    assert value.grad is not None
    assert bias.grad is not None
    assert permuted_probs.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_chunked_fallback_preserves_cuda_float64_gradient():
    """The non-Triton CUDA fallback keeps float64 products and accumulation."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    n_tokens = 16384
    value = torch.zeros(n_tokens, 1, dtype=torch.float64, device=device)
    bias = torch.zeros(1, 1, dtype=torch.float64, device=device, requires_grad=True)
    tokens_per_expert = torch.tensor([n_tokens], dtype=torch.long, device=device)
    permuted_probs = torch.ones(n_tokens, 1, dtype=torch.float64, device=device)
    permuted_probs[n_tokens // 2 :] = 1.0 - 1.0e-10
    upstream_grad = torch.ones_like(value)
    upstream_grad[n_tokens // 2 :] = -1.0

    _apply_bias(value, bias, tokens_per_expert, permuted_probs).backward(upstream_grad)

    expected_grad = (upstream_grad * permuted_probs).sum(dim=0)
    assert bias.grad is not None
    torch.testing.assert_close(bias.grad[0], expected_grad, rtol=1.0e-9, atol=1.0e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_large_weighted_double_backward_is_deterministic():
    """Large weighted bias second derivatives use deterministic segmented reductions."""
    assert _BIAS_GRAD_TRITON_AVAILABLE
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    n_tokens = 16384
    hidden = 8
    tokens_per_expert = torch.tensor([0, n_tokens // 2, n_tokens // 2, 0], dtype=torch.long, device=device)
    value = torch.zeros(n_tokens, hidden, dtype=torch.bfloat16, device=device)
    bias_data = torch.randn(4, hidden, dtype=torch.bfloat16, device=device)
    probs_data = torch.rand(n_tokens, 1, dtype=torch.float32, device=device)
    upstream_grad = torch.randn_like(value)
    second_upstream = torch.randn_like(probs_data)

    expected_rows = (upstream_grad.float() * second_upstream).to(torch.bfloat16)
    expected_grad = torch.stack(
        [segment.double().sum(dim=0) for segment in torch.split(expected_rows, tokens_per_expert.tolist())]
    ).to(torch.bfloat16)

    first_second_grad = None
    for _ in range(5):
        bias = bias_data.clone().requires_grad_()
        probs = probs_data.clone().requires_grad_()
        output = _apply_bias(value, bias, tokens_per_expert, probs)
        grad_probs = torch.autograd.grad(output, probs, grad_outputs=upstream_grad, create_graph=True)[0]
        second_grad = torch.autograd.grad(grad_probs, bias, grad_outputs=second_upstream)[0]

        torch.testing.assert_close(second_grad, expected_grad)
        if first_second_grad is None:
            first_second_grad = second_grad.clone()
        else:
            torch.testing.assert_close(second_grad, first_second_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("apply_router_weight_after_down", [False, True])
def test_checkpointed_chunked_expert_mlp_matches_full_cuda_gradients(
    monkeypatch,
    dtype,
    apply_router_weight_after_down,
):
    """Real grouped-MM chunks preserve forward values and every trainable gradient."""
    monkeypatch.setattr("nemo_automodel.components.moe.experts._BIAS_CHUNK_ROWS", 6)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    token_counts = torch.tensor([0, 5, 12, 0], dtype=torch.long, device=device)
    n_tokens = int(token_counts.sum())
    torch.manual_seed(2468)
    tensors = [
        torch.randn(n_tokens, 32, dtype=dtype, device=device, requires_grad=True),
        (torch.randn(4, 32, 96, dtype=dtype, device=device) * 0.02).requires_grad_(),
        (torch.randn(4, 48, 32, dtype=dtype, device=device) * 0.02).requires_grad_(),
        (torch.randn(4, 96, dtype=dtype, device=device) * 0.02).requires_grad_(),
        (torch.randn(4, 32, dtype=dtype, device=device) * 0.02).requires_grad_(),
        torch.rand(n_tokens, 1, dtype=torch.float32, device=device, requires_grad=True),
    ]
    expected_tensors = [tensor.detach().clone().requires_grad_() for tensor in tensors]

    result = _checkpointed_chunked_expert_mlp(
        *tensors[:5],
        token_counts,
        tensors[5],
        _quick_geglu,
        apply_router_weight_after_down,
    )
    expected = _full_grouped_expert_mlp(
        *expected_tensors[:5],
        token_counts,
        expected_tensors[5],
        apply_router_weight_after_down,
    )
    # Chunking changes grouped-GEMM accumulation order, whose FP32 rounding varies slightly across GPU architectures.
    tolerance = {"rtol": 2e-2, "atol": 4e-2} if dtype == torch.bfloat16 else {"rtol": 2e-5, "atol": 5e-5}
    torch.testing.assert_close(result, expected, **tolerance)

    upstream_grad = torch.randn_like(result)
    result.backward(upstream_grad)
    expected.backward(upstream_grad)
    for actual, reference in zip(tensors, expected_tensors):
        torch.testing.assert_close(actual.grad, reference.grad, **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_checkpointed_chunked_expert_mlp_does_not_materialize_full_gate_up_output():
    """Gradient-enabled peak memory stays bounded across checkpoint recomputation."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.bfloat16
    n_tokens = 32768
    dim = 256
    inter_dim = 512
    n_experts = 4
    token_counts = torch.tensor([0, 8193, 16383, 8192], dtype=torch.long, device=device)
    hidden = torch.randn(n_tokens, dim, dtype=dtype, device=device, requires_grad=True)
    gate_up = torch.randn(n_experts, dim, 2 * inter_dim, dtype=dtype, device=device, requires_grad=True)
    down = torch.randn(n_experts, inter_dim, dim, dtype=dtype, device=device, requires_grad=True)
    gate_bias = torch.randn(n_experts, 2 * inter_dim, dtype=dtype, device=device, requires_grad=True)
    down_bias = torch.randn(n_experts, dim, dtype=dtype, device=device, requires_grad=True)
    probs = torch.rand(n_tokens, 1, dtype=torch.float32, device=device, requires_grad=True)

    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = _checkpointed_chunked_expert_mlp(
        hidden,
        gate_up,
        down,
        gate_bias,
        down_bias,
        token_counts,
        probs,
        _quick_geglu,
        False,
    )
    output.backward(torch.randn_like(output))
    torch.cuda.synchronize(device)

    peak_increment = torch.cuda.max_memory_allocated(device) - baseline
    full_gate_up_bytes = n_tokens * 2 * inter_dim * hidden.element_size()
    assert output.shape == (n_tokens, dim)
    assert peak_increment < 3 * full_gate_up_bytes
    assert all(tensor.grad is not None for tensor in (hidden, gate_up, down, gate_bias, down_bias, probs))
