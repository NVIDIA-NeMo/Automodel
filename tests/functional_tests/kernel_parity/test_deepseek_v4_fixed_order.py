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

"""Repeatability and independent FP64 accuracy gates for fixed-order backward."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.kernels.sparse_attention import (
    sparse_attn_tilelang,
    sparse_attn_tilelang_head_chunked,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("heads", [16, 128])
@pytest.mark.parametrize("reference_rounding", [False, True])
@pytest.mark.parametrize("invalid_index", [2**32, -(2**32)])
@pytest.mark.parametrize("control_dtype", [torch.int32, torch.int64])
def test_fixed_order_masks_int64_overflow(
    heads: int, reference_rounding: bool, invalid_index: int, control_dtype: torch.dtype
) -> None:
    """An invalid wide index must behave exactly like the masked sentinel."""
    results = []
    for index, dtype in ((-1, control_dtype), (invalid_index, torch.int64)):
        q = torch.zeros(1, 1, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kv = torch.stack((torch.zeros(128, device="cuda"), torch.ones(128, device="cuda")))
        kv = kv.unsqueeze(0).to(torch.bfloat16).requires_grad_()
        sink = torch.zeros(heads, device="cuda", requires_grad=True)
        indices = torch.tensor([[[1, index]]], device="cuda", dtype=dtype)
        if heads > 64:
            output = sparse_attn_tilelang_head_chunked(
                q, kv, sink, indices, 64,
                reference_rounding=reference_rounding,
                backward_impl="deterministic",
                backward_workspace_bytes=32 * 128 * 4,
            )
        else:
            output = sparse_attn_tilelang(
                q, kv, sink, indices,
                reference_rounding=reference_rounding,
                backward_impl="deterministic",
                backward_workspace_bytes=32 * 128 * 4,
            )
        gradients = torch.autograd.grad(output, (q, kv, sink), torch.ones_like(output))
        assert torch.equal(output, torch.full_like(output, 0.5))
        assert torch.count_nonzero(gradients[1][:, 0]) == 0
        assert torch.equal(gradients[1][:, 1], torch.full_like(gradients[1][:, 1], heads / 2))
        results.append((output.detach(), *(value.detach() for value in gradients)))
    assert all(torch.equal(actual, control) for actual, control in zip(results[1], results[0], strict=True))


def _reference(q, kv, sink, indices, incoming):
    """Differentiate a float64 sparse-softmax reference.

    Args:
        q: BF16 [batch, query, heads, dim] queries.
        kv: BF16 [batch, key, dim] shared KV.
        sink: FP32 [heads] logits.
        indices: Int32 [batch, query, slots] indices, -1 padding.
        incoming: BF16 [batch, query, heads, dim] output gradient.

    Returns:
        FP64 output and gradients with the respective input shapes.
    """
    q, kv, sink = [x.double().detach().requires_grad_() for x in (q, kv, sink)]
    selected = kv[torch.arange(q.shape[0], device=q.device)[:, None, None], indices.clamp_min(0).long()]
    scores = torch.einsum("bqhd,bqkd->bqhk", q, selected) * q.shape[-1] ** -0.5
    scores = scores.masked_fill((indices < 0)[:, :, None, :], -torch.inf)
    logits = torch.cat((scores, sink[None, None, :, None].expand(*scores.shape[:-1], 1)), dim=-1)
    output = torch.einsum("bqhk,bqkd->bqhd", logits.softmax(-1)[..., :-1], selected)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        grads = torch.autograd.grad(output, (q, kv, sink), incoming.double())
    finally:
        torch.use_deterministic_algorithms(previous)
    return output.detach(), tuple(x.detach() for x in grads)


@pytest.mark.parametrize(
    "batch,heads,dim,sequence,slots", [(1, 16, 128, 7, 17), (2, 32, 128, 9, 33), (1, 64, 512, 17, 64)]
)
def test_fixed_order_repeatability_and_accuracy(batch, heads, dim, sequence, slots):
    torch.manual_seed(1017)
    q = torch.randn(batch, sequence, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, 97, dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda")
    # Repeated valid keys test both within-query duplicates and cross-query contention.
    indices = torch.randint(0, 17, (batch, sequence, slots), device="cuda", dtype=torch.int32)
    indices[:, :, -3:] = -1
    indices[:, 0, :] = -1
    incoming = torch.randn(batch, sequence, dim, heads, device="cuda", dtype=q.dtype).transpose(2, 3)
    reference, reference_grads = _reference(q, kv, sink, indices, incoming)
    repeated_reference, repeated_grads = _reference(q, kv, sink, indices, incoming)
    assert torch.equal(reference, repeated_reference)
    assert all(torch.equal(a, b) for a, b in zip(reference_grads, repeated_grads, strict=True))
    first = None
    for repeat in range(6):
        inputs = [x.detach().clone().requires_grad_() for x in (q, kv, sink)]
        output = sparse_attn_tilelang(
            *inputs,
            indices,
            reference_rounding=True,
            backward_impl="deterministic",
            # Force several chunks and a tail.
            backward_workspace_bytes=batch * ((slots + 31) // 32 * 32) * dim * 4 * 4,
        )
        grads = torch.autograd.grad(output, inputs, incoming)
        actual = (output.detach(), *[g.detach() for g in grads])
        if first is None:
            first = actual
        for value, previous in zip(actual, first, strict=True):
            assert torch.equal(value, previous), f"repeat {repeat} differs"
            assert torch.isfinite(value).all()
        torch.testing.assert_close(output.double(), reference, rtol=2e-2, atol=2e-2)
        for value, expected in zip(grads, reference_grads, strict=True):
            # Existing sparse-kernel elementwise gate plus a stricter norm gate.
            torch.testing.assert_close(value.double(), expected, rtol=5e-2, atol=5e-2)
            assert ((value.double() - expected).norm() / expected.norm().clamp_min(1e-30)).item() < 0.01
        assert torch.count_nonzero(grads[0][:, 0]) == 0
        assert torch.count_nonzero(grads[1][:, 17:]) == 0
    atomic = sparse_attn_tilelang(q, kv, sink, indices, reference_rounding=True)
    assert torch.equal(first[0], atomic), "backward option changed forward"


def test_fixed_order_head_chunking():
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention

    torch.manual_seed(52)
    q = torch.randn(1, 5, 128, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 41, 512, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(128, device="cuda")
    indices = torch.randint(0, 41, (1, 5, 32), device="cuda", dtype=torch.int32)
    incoming = torch.randn_like(q)
    expected, reference_grads = _reference(q, kv, sinks, indices, incoming)
    first = None
    for _ in range(6):
        inputs = [x.detach().clone().requires_grad_() for x in (q, kv, sinks)]
        output = dsv4_sparse_attention(
            *inputs, indices, 512**-0.5, backend="tilelang", reference_rounding=True, backward_impl="deterministic"
        )
        grads = torch.autograd.grad(output, inputs, incoming)
        if first is None:
            first = tuple(g.detach().clone() for g in grads)
        assert all(torch.equal(g, old) for g, old in zip(grads, first, strict=True))
        torch.testing.assert_close(output.double(), expected, rtol=2e-2, atol=2e-2)
        for value, reference in zip(grads, reference_grads, strict=True):
            torch.testing.assert_close(value.double(), reference, rtol=5e-2, atol=5e-2)
            assert ((value.double() - reference).norm() / reference.norm().clamp_min(1e-30)).item() < 0.01


def test_fixed_order_reuses_compilation_across_lengths(monkeypatch):
    """Changing batches, KV lengths and tail sizes must reuse backward kernels."""
    from nemo_automodel.components.models.deepseek_v4.kernels import fixed_order_sparse_mla as fixed
    from nemo_automodel.components.models.deepseek_v4.kernels import tilelang_sparse_mla_bwd as backward

    torch.manual_seed(1017)
    # Observe the returned compiled kernels through the lazy import shim.
    # The identity must stay stable even when runtime tensor lengths change.
    factories = ((backward, "preprocess"), (backward, "bwd"), (fixed, "_reduce_kv"), (fixed, "_reduce_sink"))
    seen = {name: set() for _, name in factories}

    def observe(factory, name):
        def wrapped(*args, **kwargs):
            kernel = factory(*args, **kwargs)
            seen[name].add(id(kernel))
            return kernel

        return wrapped

    for owner, name in factories:
        monkeypatch.setattr(owner, name, observe(getattr(owner, name), name))
    cached = None
    for batch, sequence, keys in ((1, 7, 41), (2, 9, 53), (1, 13, 97)):
        q = torch.randn(batch, sequence, 16, 128, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(batch, keys, 128, device="cuda", dtype=q.dtype)
        sink = torch.randn(16, device="cuda")
        indices = torch.randint(0, keys, (batch, sequence, 32), device="cuda", dtype=torch.int32)
        indices[:, 0] = -1
        incoming = torch.randn_like(q)
        _, expected = _reference(q, kv, sink, indices, incoming)
        first = None
        for _ in range(2):
            inputs = [value.detach().clone().requires_grad_() for value in (q, kv, sink)]
            output = sparse_attn_tilelang(
                *inputs,
                indices,
                reference_rounding=True,
                backward_impl="deterministic",
                backward_workspace_bytes=batch * 32 * 128 * 4 * 4,
            )
            grads = torch.autograd.grad(output, inputs, incoming)
            for actual, reference in zip(grads, expected, strict=True):
                torch.testing.assert_close(actual.double(), reference, rtol=5e-2, atol=5e-2)
                assert ((actual.double() - reference).norm() / reference.norm().clamp_min(1e-30)).item() < 0.01
            if first is None:
                first = grads
            else:
                assert all(torch.equal(a, b) for a, b in zip(first, grads, strict=True))
        current = {name: frozenset(kernels) for name, kernels in seen.items()}
        if cached is None:
            cached = current
        else:
            assert current == cached, "a new sequence/batch shape compiled another backward kernel"
