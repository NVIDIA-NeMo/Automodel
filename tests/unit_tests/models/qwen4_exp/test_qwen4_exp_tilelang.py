# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional H100 correctness tests for Qwen4-Exp TileLang sparse GQA."""

import pytest
import torch

from nemo_automodel.components.models.qwen4_exp.kernels._tilelang import HAS_TILELANG
from nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention import tilelang_sparse_gqa_attention
from nemo_automodel.components.models.qwen4_exp.qsa import gathered_qsa_gqa_attention, qsa_gqa_attention


def _can_run_hopper_tilelang() -> bool:
    return HAS_TILELANG and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


requires_hopper_tilelang = pytest.mark.skipif(
    not _can_run_hopper_tilelang(),
    reason="requires TileLang on an H100-class CUDA device",
)


def _forward_backward(
    operator,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value))
    output = operator(*inputs, token_ids)
    gradients = torch.autograd.grad(output, inputs, grad_output)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


@requires_hopper_tilelang
@pytest.mark.parametrize(("query_length", "kv_length", "topk"), [(17, 80, 65), (2, 4096, 2051)])
def test_tilelang_sparse_gqa_matches_oracle_forward_backward(
    query_length: int,
    kv_length: int,
    topk: int,
) -> None:
    """Cover native Qwen heads plus non-tile and released 2,051-token widths."""
    torch.manual_seed(1234 + topk)
    device = torch.device("cuda")
    query = torch.randn(1, query_length, 24, 256, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, kv_length, 2, 256, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    token_ids = torch.full((1, query_length, topk), -1, device=device, dtype=torch.int32)
    for query_idx in range(1, query_length):
        valid_width = min(topk, kv_length)
        token_ids[0, query_idx, :valid_width] = torch.randperm(kv_length, device=device)[:valid_width]
    if query_length > 1:
        # Duplicate IDs exercise FP32 atomic dK/dV accumulation.
        token_ids[0, -1, :4] = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)

    grad_storage = torch.randn(
        1,
        query_length,
        24,
        512,
        device=device,
        dtype=torch.bfloat16,
    )
    grad_output = grad_storage[..., ::2]
    assert not grad_output.is_contiguous()

    expected, expected_gradients = _forward_backward(
        gathered_qsa_gqa_attention,
        query,
        key,
        value,
        token_ids,
        grad_output,
    )
    actual, actual_gradients = _forward_backward(
        tilelang_sparse_gqa_attention,
        query,
        key,
        value,
        token_ids,
        grad_output,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-2, atol=5e-2)
    assert torch.count_nonzero(actual[:, 0]) == 0
    assert torch.count_nonzero(actual_gradients[0][:, 0]) == 0


@requires_hopper_tilelang
def test_cuda_qsa_dispatch_rejects_non_tilelang_backend() -> None:
    query = torch.randn(1, 1, 2, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    token_ids = torch.zeros(1, 1, 1, device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="requires backend.attn='tilelang'"):
        qsa_gqa_attention(query, key, value, token_ids, backend="sdpa")
