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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.kimi_k3.model import KimiRMSNorm, _apply_attn_res


def _reference_matmul_impl(prefix_sum, block_residual, projection, norm):
    """Pre-rewrite implementation: degenerate batched GEMM via torch.matmul."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    variance = values_fp32.pow(2).mean(-1, keepdim=True)
    keys = values_fp32 * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (keys * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)


def _make_inputs(tokens, blocks, hidden, dtype, seed=1234):
    torch.manual_seed(seed)
    prefix_sum = torch.randn(tokens, hidden, dtype=dtype, requires_grad=True)
    block_residual = torch.randn(tokens, blocks, hidden, dtype=dtype, requires_grad=True)
    projection = nn.Linear(hidden, 1, bias=False).to(dtype)
    norm = KimiRMSNorm(hidden)
    norm.weight.data = torch.randn(hidden).to(norm.weight.dtype)
    return prefix_sum, block_residual, projection, norm


@pytest.mark.parametrize("tokens,blocks,hidden", [(64, 3, 128), (128, 8, 64), (1, 1, 32)])
def test_forward_matches_matmul_reference_bf16(tokens, blocks, hidden):
    """bf16 forward matches the previous matmul implementation.

    The rewrite reorders fp32 accumulation, so equality is asserted to within
    one bf16 ulp; for most shapes the outputs are bitwise identical.
    """
    ps, br, proj, norm = _make_inputs(tokens, blocks, hidden, torch.bfloat16)
    out = _apply_attn_res(ps, br, proj, norm)
    ref = _reference_matmul_impl(
        ps.detach().clone().requires_grad_(True),
        br.detach().clone().requires_grad_(True),
        proj,
        norm,
    )
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=8e-3, atol=1e-5)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-6), (torch.bfloat16, 1e-2)])
def test_forward_and_grads_close_to_reference(dtype, tol):
    ps, br, proj, norm = _make_inputs(96, 4, 64, dtype)
    ps_ref = ps.detach().clone().requires_grad_(True)
    br_ref = br.detach().clone().requires_grad_(True)

    out = _apply_attn_res(ps, br, proj, norm)
    ref = _reference_matmul_impl(ps_ref, br_ref, proj, norm)
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)

    out.float().square().sum().backward()
    ref.float().square().sum().backward()
    torch.testing.assert_close(ps.grad.float(), ps_ref.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(br.grad.float(), br_ref.grad.float(), rtol=tol, atol=tol)


def test_gradients_flow_to_projection_and_norm_weights():
    ps, br, proj, norm = _make_inputs(32, 2, 48, torch.float32)
    out = _apply_attn_res(ps, br, proj, norm)
    out.sum().backward()
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0
    assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0
