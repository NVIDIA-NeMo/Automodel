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

"""Independent gradient audit for the original-inference sparse rounding mode."""

import importlib.util
import json
import math

import pytest
import torch

# Fixed before the first GPU audit. These bounds cover BF16 arithmetic against
# an FP32 mathematical oracle; they do not permit changing the LSE convention.
GRAD_RELATIVE_RMSE_LIMIT = 1e-2
GRAD_COSINE_MINIMUM = 0.999
requires_tilelang = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("tilelang") is None,
    reason="requires CUDA and TileLang",
)


def _torch_attention(q, kv, sink, indices, scale):
    """Compute an independent FP32 softmax over sparse slots and a zero-value sink.

    Duplicate indices remain separate softmax slots. Autograd then adds both
    the key and value paths back to each shared KV row through indexed reads.
    """
    valid = (indices >= 0) & (indices < kv.shape[1])
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    selected = kv.float()[batch, indices.clamp(0, kv.shape[1] - 1).long()]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), selected) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    sink_logits = sink[None, None, :, None].expand(*scores.shape[:-1], 1)
    probabilities = torch.cat((scores, sink_logits), dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probabilities, selected)


@pytest.mark.parametrize("duplicates", [1, 2])
def test_gradient_oracle_matches_analytic_sink_and_duplicate_case(duplicates):
    """Validate the oracle against the one-key softmax derivative, including masks."""
    q = torch.zeros(1, 1, 1, 64, requires_grad=True)
    kv = torch.ones(1, 1, 64, requires_grad=True)
    sink = torch.zeros(1, requires_grad=True)
    indices = torch.tensor([[[*([0] * duplicates), -1, 1]]], dtype=torch.int32)
    output = _torch_attention(q, kv, sink, indices, 64**-0.5)
    dq, dkv, dsink = torch.autograd.grad(output.sum(), (q, kv, sink))
    probability = duplicates / (duplicates + 1)
    derivative = duplicates / (duplicates + 1) ** 2
    torch.testing.assert_close(output, torch.full_like(output, probability))
    torch.testing.assert_close(dq, torch.full_like(dq, 8 * derivative))
    torch.testing.assert_close(dkv, torch.full_like(dkv, probability))
    torch.testing.assert_close(dsink, torch.full_like(dsink, -64 * derivative))


def _inputs(heads, dim, sparse, seed):
    """Build causal rows or window-plus-compressed rows with duplicate gather IDs."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    sequence = 19 if sparse else 17
    batch_size = 1 if dim == 512 else 2
    compressed = 11 if sparse else 0
    kv_length = sequence + compressed
    q = torch.randn(batch_size, sequence, heads, dim, generator=generator, device="cuda").bfloat16()
    kv = torch.randn(batch_size, kv_length, dim, generator=generator, device="cuda").bfloat16()
    sink = torch.linspace(-3, 3, heads, device="cuda")
    slots = 81 if sparse else sequence
    indices = torch.full((batch_size, sequence, slots), -1, dtype=torch.int32, device="cuda")
    for position in range(sequence):
        if sparse:
            window = torch.arange(max(0, position - 7), position + 1, device="cuda", dtype=torch.int32)
            indices[:, position, : window.numel()] = window
            count = min(compressed, (position + 1) // 2)
            if count:
                # Deliberate duplicates cross the 64-slot block boundary.
                indices[:, position, 8:] = sequence + torch.arange(slots - 8, device="cuda") % count
                indices[:, position, 13::17] = kv_length + 7  # invalid upper-bound slots
        else:
            indices[:, position, : position + 1] = torch.arange(position + 1, device="cuda")
    upstream = torch.randn(q.shape, generator=generator, device="cuda").bfloat16()
    return q, kv, sink, indices, upstream


def _metrics(actual, expected):
    actual, expected = actual.double().flatten(), expected.double().flatten()
    difference = actual - expected
    reference_norm = torch.linalg.vector_norm(expected)
    actual_norm = torch.linalg.vector_norm(actual)
    return {
        "relative_rmse": (torch.linalg.vector_norm(difference) / reference_norm).item(),
        "cosine": (torch.dot(actual, expected) / (actual_norm * reference_norm)).item(),
        "max_abs": difference.abs().max().item(),
        "reference_l2": reference_norm.item(),
    }


@requires_tilelang
@pytest.mark.parametrize("seed", [101, 202])
@pytest.mark.parametrize("heads,dim,sparse", [(4, 64, False), (8, 64, True), (64, 512, False), (64, 512, True)])
def test_reference_rounding_gradients_match_fp32_oracle(heads, dim, sparse, seed, record_property, monkeypatch):
    """Check all three gradients, including the production H64/D512 head chunking."""
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    q, kv, sink, indices, upstream = _inputs(heads, dim, sparse, seed)
    native_inputs = tuple(value.detach().clone().requires_grad_() for value in (q, kv, sink))
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in (q, kv, sink))
    scale = dim**-0.5
    native = dsv4_sparse_attention(*native_inputs, indices, scale, backend="tilelang", reference_rounding=True)
    reference = _torch_attention(*reference_inputs, indices, scale)
    native_gradients = torch.autograd.grad(native, native_inputs, upstream)
    reference_gradients = torch.autograd.grad(reference, reference_inputs, upstream.float())
    report = {
        "heads": heads,
        "dim": dim,
        "sparse_duplicates": sparse,
        "seed": seed,
        "relative_rmse_limit": GRAD_RELATIVE_RMSE_LIMIT,
        "cosine_minimum": GRAD_COSINE_MINIMUM,
        "forward": _metrics(native, reference),
        "gradients": {
            name: _metrics(actual, expected)
            for name, actual, expected in zip(("q", "kv", "sink"), native_gradients, reference_gradients)
        },
    }
    record_property("gradient_audit", json.dumps(report))
    print(json.dumps({"gradient_audit": report}), flush=True)
    for name, metrics in report["gradients"].items():
        assert all(math.isfinite(value) for value in metrics.values()), (name, metrics)
        assert metrics["relative_rmse"] <= GRAD_RELATIVE_RMSE_LIMIT, (name, metrics)
        assert metrics["cosine"] >= GRAD_COSINE_MINIMUM, (name, metrics)
