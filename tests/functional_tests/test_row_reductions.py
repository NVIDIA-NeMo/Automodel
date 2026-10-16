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

"""CUDA numerical and allocator regressions for column reductions."""

import pytest
import torch

from nemo_automodel._transformers.retrieval import pool
from nemo_automodel.components.models.common.utils import sum_rows
from nemo_automodel.components.models.kimi_k3.attn_res_triton import HAVE_TRITON
from nemo_automodel.components.models.kimi_k3.situ import _attn_res_core, _AttnResTritonFunction
from nemo_automodel.components.models.ministral_bidirectional.model import pool as ministral_pool
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Exercises CUDA reduction workspace and gradients"
)


@pytest.mark.parametrize("rows", [0, 1, 512, 513, 1024, 1025])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_sum_rows_values_and_gradients(rows, dtype, strided):
    torch.manual_seed(12)
    storage = torch.randn(2, rows, 34, device="cuda", dtype=dtype)
    values = (storage[..., ::2] if strided else storage).detach().requires_grad_()
    reference = values.detach().double().sum(dim=-2).to(dtype)
    result = sum_rows(values)
    rtol, atol = (8e-3, 1e-4) if dtype in (torch.float16, torch.bfloat16) else (1e-5, 1e-5)
    torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
    upstream = torch.randn_like(result)
    actual_grad = torch.autograd.grad(result, values, upstream)[0]
    torch.testing.assert_close(actual_grad, upstream.unsqueeze(-2).expand_as(values), rtol=0, atol=0)


def test_sum_rows_preserves_low_precision_accumulation_and_empty_shapes():
    # Casting each chunk to FP16 would overflow before cancellation across chunks.
    values = torch.cat((torch.full((512, 3), 200.0), torch.full((512, 3), -200.0))).cuda().half()
    torch.testing.assert_close(sum_rows(values), torch.zeros(3, device="cuda", dtype=torch.float16), rtol=0, atol=0)
    for shape in ((0, 513, 7), (2, 513, 0)):
        values = torch.empty(shape, device="cuda")
        torch.testing.assert_close(sum_rows(values), values.sum(-2))
    values = torch.ones((513, 3), device="cuda", dtype=torch.int32)
    torch.testing.assert_close(sum_rows(values), values.sum(-2), rtol=0, atol=0)


def test_sum_rows_strided_rows_and_contiguous_reduction_axis():
    torch.manual_seed(48)
    storage = torch.randn(2, 3, 1026, 7, device="cuda")
    for values in (storage[..., ::2, :], storage.transpose(-2, -1).contiguous().transpose(-2, -1)):
        torch.testing.assert_close(sum_rows(values), values.double().sum(-2).float(), rtol=1e-5, atol=1e-5)


def test_sum_rows_higher_order_gradients():
    values = torch.randn(513, 2, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(sum_rows, (values,), fast_mode=True)
    assert torch.autograd.gradgradcheck(sum_rows, (values,), fast_mode=True)


@pytest.mark.parametrize(
    "shape,dtype", [((32769, 512), torch.float32), ((4096, 7168), torch.float32), ((1, 32769, 4096), torch.bfloat16)]
)
def test_sum_rows_forward_and_backward_workspace(shape, dtype):
    values = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn((*shape[:-2], shape[-1]), device="cuda", dtype=dtype)
    for _ in range(3):
        torch.autograd.grad(sum_rows(values), values, upstream)
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = sum_rows(values)
    gradient = torch.autograd.grad(result, values, upstream)[0]
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    # Native CUDA column sums allocate tens of MiB here. Tail slice backwards
    # would also allocate an input-sized gradient rather than a broadcast view.
    assert peak < 4 * 1024**2
    torch.testing.assert_close(gradient, upstream.unsqueeze(-2).expand_as(values), rtol=0, atol=0)


def test_sum_rows_compiled_dynamic_forward_backward():
    compiled = torch.compile(sum_rows, fullgraph=True, dynamic=True)
    for rows in (513, 1024, 1025, 17):
        values = torch.randn(2, rows, 31, device="cuda", requires_grad=True)
        upstream = torch.randn(2, 31, device="cuda")
        result = compiled(values)
        torch.testing.assert_close(result, values.double().sum(-2).float(), rtol=1e-5, atol=1e-5)
        gradient = torch.autograd.grad(result, values, upstream)[0]
        torch.testing.assert_close(gradient, upstream.unsqueeze(-2).expand_as(values), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("all_masked", [False, True])
def test_gate_aux_loss_values_and_gradients(dtype, all_masked):
    torch.manual_seed(45)
    experts, topk, tokens = 64, 4, 1025
    gate = Gate(
        MoEConfig(
            n_routed_experts=experts,
            n_shared_experts=0,
            n_activated_experts=topk,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0,
            aux_loss_coeff=0.1,
            score_func="softmax",
            route_scale=1,
            dim=16,
            inter_dim=32,
            moe_inter_dim=32,
            norm_topk_prob=False,
        )
    )
    scores = torch.rand(tokens, experts, device="cuda", dtype=dtype, requires_grad=True)
    mask = (
        torch.zeros(tokens, device="cuda", dtype=torch.bool) if all_masked else torch.rand(tokens, device="cuda") > 0.2
    )
    loads = torch.randint(0, tokens, (experts,), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = gate._compute_aux_loss(scores, loads, mask, None)
    count = mask.sum().clamp_min(1).double()
    coefficients = loads.double() * experts / (topk * count.square())
    reference = ((scores.detach().double() * mask[:, None]).sum(0) * coefficients).sum()
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss.double(), reference, rtol=1e-6, atol=1e-6)
    gradient = torch.autograd.grad(loss * 1.7, scores)[0]
    expected = (mask[:, None] * coefficients * 1.7).to(dtype)
    torch.testing.assert_close(gradient, expected, rtol=1e-6 if dtype == torch.float32 else 8e-3, atol=1e-7)


@pytest.mark.parametrize("pool_fn", [pool, ministral_pool])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_retrieval_pool_values_and_gradients(pool_fn, dtype):
    torch.manual_seed(46)
    values = torch.randn(2, 1025, 64, device="cuda", dtype=dtype, requires_grad=True)
    mask = torch.rand(2, 1025, device="cuda") > 0.2
    # Masking must suppress NaNs at padding positions, as masked_fill did before.
    with torch.no_grad():
        values.masked_fill_(~mask[..., None], float("nan"))
    result = pool_fn(values, mask, "avg")
    reference = values.detach().double().masked_fill(~mask[..., None], 0).sum(1).to(dtype) / mask.sum(1)[:, None]
    torch.testing.assert_close(result, reference, rtol=8e-3 if dtype == torch.bfloat16 else 1e-5, atol=1e-5)
    upstream = torch.randn_like(result)
    gradient = torch.autograd.grad(result, values, upstream)[0]
    expected = (upstream / mask.sum(1)[:, None])[:, None, :].expand_as(values).masked_fill(~mask[..., None], 0)
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not HAVE_TRITON, reason="Kimi attention-residual kernels require optional Triton")
def test_kimi_attention_residual_long_backward(dtype):
    torch.manual_seed(47)
    tokens, hidden = 4097, 64  # 513 partial rows, including a tail in both reductions.
    prefix = torch.randn(tokens, hidden, device="cuda", dtype=dtype, requires_grad=True)
    blocks = torch.randn(tokens, 1, hidden, device="cuda", dtype=dtype, requires_grad=True)
    norm = torch.ones(hidden, device="cuda", dtype=dtype, requires_grad=True)
    projection = (torch.randn(hidden, device="cuda", dtype=dtype) * 0.05).requires_grad_()
    reference_inputs = [value.detach().clone().requires_grad_() for value in (prefix, blocks, norm, projection)]
    reference = _attn_res_core(
        torch.cat((reference_inputs[1], reference_inputs[0].unsqueeze(1)), dim=1),
        reference_inputs[2],
        reference_inputs[3],
        1e-6,
        dtype,
    )
    result = _AttnResTritonFunction.apply(prefix, blocks, norm, projection, 1e-6)
    torch.testing.assert_close(result, reference, rtol=8e-3 if dtype == torch.bfloat16 else 1e-5, atol=1e-5)
    upstream = torch.randn_like(result)
    actual = torch.autograd.grad(result, (prefix, blocks, norm, projection), upstream)
    expected = torch.autograd.grad(reference, reference_inputs, upstream)
    for got, want in zip(actual, expected):
        scale = want.float().abs().max().clamp_min(1e-6).item()
        tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance * scale)
