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

"""The concat-free absorbed sparse-attention query matches ``cat([einsum(q_nope, w_kc), q_pe])``."""

import pytest
import torch

from nemo_automodel.components.models.glm_moe_dsa.layers import absorbed_sparse_query

TOKENS, HEADS, NOPE, ROPE, LATENT = 37, 4, 12, 4, 16


def _reference(q_nope: torch.Tensor, q_pe: torch.Tensor, w_kc: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.einsum("thd,hdc->thc", q_nope, w_kc), q_pe], dim=-1)


def _inputs(device: str, dtype: torch.dtype):
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(TOKENS, HEADS, NOPE + ROPE, generator=generator).to(device=device, dtype=dtype).requires_grad_(True)
    w_kc = torch.randn(HEADS, NOPE, LATENT, generator=generator).to(device=device, dtype=dtype).requires_grad_(True)
    return q, w_kc


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_matches_reference_values_and_gradients_on_cpu(dtype):
    q, w_kc = _inputs("cpu", dtype)
    q_nope, q_pe = torch.split(q, [NOPE, ROPE], dim=-1)  # views, as in the attention forward
    out = absorbed_sparse_query(q_nope, q_pe, w_kc)
    assert out.shape == (TOKENS, HEADS, LATENT + ROPE)
    assert out.is_contiguous() and out.dtype == dtype
    grad = torch.randn_like(out)
    out.backward(grad)

    q_ref, w_ref = _inputs("cpu", dtype)
    q_nope_ref, q_pe_ref = torch.split(q_ref, [NOPE, ROPE], dim=-1)
    ref = _reference(q_nope_ref, q_pe_ref, w_ref)
    ref.backward(grad)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(q.grad, q_ref.grad)
    torch.testing.assert_close(w_kc.grad, w_ref.grad)


def test_gradient_of_each_input_can_be_skipped():
    q, w_kc = _inputs("cpu", torch.float64)
    q_nope, q_pe = torch.split(q.detach(), [NOPE, ROPE], dim=-1)
    out = absorbed_sparse_query(q_nope, q_pe, w_kc)
    out.sum().backward()
    assert w_kc.grad is not None and torch.isfinite(w_kc.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 strided-output GEMM path is exercised on CUDA")
def test_bf16_cuda_matches_reference():
    q, w_kc = _inputs("cuda", torch.bfloat16)
    q_nope, q_pe = torch.split(q, [NOPE, ROPE], dim=-1)
    out = absorbed_sparse_query(q_nope, q_pe, w_kc)
    grad = torch.randn_like(out)
    out.backward(grad)
    q_ref, w_ref = _inputs("cuda", torch.bfloat16)
    q_nope_ref, q_pe_ref = torch.split(q_ref, [NOPE, ROPE], dim=-1)
    ref = _reference(q_nope_ref, q_pe_ref, w_ref)
    ref.backward(grad)
    assert out.is_contiguous()
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(q.grad, q_ref.grad)
    torch.testing.assert_close(w_kc.grad, w_ref.grad)
