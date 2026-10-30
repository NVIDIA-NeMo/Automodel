# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed DSA dispatch, boundary isolation and independent FP32 gradient checks."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common.cudnn_sparse_attention import is_cudnn_sparse_attention_available
from nemo_automodel.components.models.glm5_next import layers as layer_module
from nemo_automodel.components.models.glm5_next.backend import Glm5NextBackendConfig
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext
from nemo_automodel.components.models.glm5_next.layers import Glm5NextSparseAttention


def reference_document(
    x: torch.Tensor, model: Glm5NextSparseAttention, topk: int, pool: int, tail: bool
) -> torch.Tensor:
    """Independently evaluate expanded attention with explicit pool selection.

    Args:
        x: One document's states [tokens, hidden].
        model: Source projection and norm weights.
        topk: Selected complete-pool token budget.
        pool: Number of tokens per complete pool.
        tail: Whether to include the incomplete causal tail.

    Returns:
        Independent document attention [tokens, hidden].
    """
    n = x.shape[0]
    c = model.config

    def norm(z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Normalize states without using the production RMSNorm helper.

        Args:
            z: States [tokens, channels].
            weight: Scale [channels].

        Returns:
            Normalized states [tokens, channels].
        """
        return z * (z.square().mean(-1, keepdim=True) + c.rms_norm_eps).rsqrt() * weight

    qr = norm(F.linear(x, model.q_a_proj.weight), model.q_a_layernorm.weight)
    query = F.linear(qr, model.q_b_proj.weight).reshape(n, c.num_attention_heads, c.qk_nope_head_dim)
    latent = norm(F.linear(x, model.kv_a_proj_with_mqa.weight), model.kv_a_layernorm.weight)
    expanded = F.linear(latent, model.kv_b_proj.weight).reshape(
        n, c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim
    )
    keys, values = expanded.split([c.qk_nope_head_dim, c.v_head_dim], -1)
    with torch.no_grad():
        ix = model.indexer
        ik = F.layer_norm(F.linear(x, ix.wk.weight), (ix.head_dim,), ix.k_norm.weight, ix.k_norm.bias, 1e-6)
        gates = F.linear(x, ix.index_kpool_compress_gate)
        pools = []
        for start in range(0, n - pool + 1, pool):
            probability = (gates[start : start + pool] + ix.index_kpool_compress_ape).softmax(0)
            pools.append((probability * ik[start : start + pool]).sum(0))
        pool_keys = torch.stack(pools) if pools else ik[:0]
        iq = F.linear(qr, ix.wq_b.weight).reshape(n, ix.n_heads, ix.head_dim)
        iw = F.linear(x, ix.weights_proj.weight) / ix.n_heads**0.5
        selected = []
        for pos in range(n):
            visible = (pos + 1) // pool
            count = min(visible, topk // pool)
            if count:
                scores = (iq[pos] @ pool_keys[:visible].T / ix.head_dim**0.5).relu()
                scores = (scores * iw[pos, :, None]).sum(0)
                chosen = scores.argsort(descending=True)[:count]
                indices = [int(p) * pool + j for p in chosen for j in range(pool)]
            else:
                indices = []
            if tail:
                indices.extend(range(visible * pool, pos + 1))
            selected.append(sorted(indices))
    outputs = []
    for pos, indices in enumerate(selected):
        if not indices:
            outputs.append(values.sum(0) * 0)
            continue
        kk, vv = keys[indices].transpose(0, 1), values[indices].transpose(0, 1)
        scores = (query[pos, :, None, :] * kk).sum(-1) / c.qk_nope_head_dim**0.5
        outputs.append((scores.softmax(-1)[..., None] * vv).sum(1))
    return F.linear(torch.stack(outputs).reshape(n, -1), model.o_proj.weight)


def reference(
    x: torch.Tensor,
    model: Glm5NextSparseAttention,
    rows: list[list[int]],
    topk: int,
    pool: int,
    tail: bool,
    gap: int = 0,
) -> torch.Tensor:
    """Evaluate independent documents without the production packing helpers.

    Args:
        x: Packed states [batch, global_sequence, hidden].
        model: Source projection and norm weights.
        rows: Document lengths for each batch row.
        topk: Selected complete-pool token budget.
        pool: Number of tokens per complete pool.
        tail: Whether to include the incomplete causal tail.
        gap: Padding tokens between documents.

    Returns:
        Packed attention [batch, global_sequence, hidden], zero at padding.
    """
    out = torch.zeros_like(x)
    for row, lengths in enumerate(rows):
        start = 0
        for n in lengths:
            out[row, start : start + n] = reference_document(x[row, start : start + n], model, topk, pool, tail)
            start += n + gap
    return out


def reference_absorbed_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    *,
    all_rows_nonempty: bool = False,
) -> torch.Tensor:
    """Provide a differentiable FP32 math substitute for a CPU dispatch check.

    Args:
        q: Absorbed queries [queries, heads, kv_rank].
        kv: Document latent K/V [keys, 1, kv_rank].
        indices: Selected raw indices [queries, 1, selection_width].
        scale: Original attention scale.
        all_rows_nonempty: Original backend flag, not needed by the oracle.

    Returns:
        Math attention [queries, heads, kv_rank]. This is not a CUDA kernel test.
    """
    raw = indices.squeeze(1).long()
    valid = raw >= 0
    selected = kv[:, 0][raw.clamp_min(0)]
    logits = torch.einsum("qhc,qkc->qhk", q, selected) * scale
    logits = logits.masked_fill(~valid[:, None], torch.finfo(logits.dtype).min)
    probabilities = logits.softmax(-1) * valid[:, None]
    return torch.einsum("qhk,qkc->qhc", probabilities, selected)


@pytest.mark.parametrize("pool,tail", [(4, True), (4, False), (2, True)])
@pytest.mark.parametrize(
    "rows",
    [
        [[5, 3, 17, 7], [3, 21, 5]],
        [[3, 2], []],
        [[7, 7, 7, 7, 4], [3, 5, 5, 5, 7, 7]],
    ],
)
@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("gap", [0, 3])
def test_packed_attention_matches_independent_document_oracle(
    pool: int, tail: bool, rows: list[list[int]], ac: bool, gap: int
) -> None:
    cfg = Glm5NextTextConfig(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        index_kpool=pool,
        index_kpool_always_select_tail=tail,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        dtype=torch.float32,
    )
    backend = Glm5NextBackendConfig(attn="cudnn", linear="torch", rms_norm="torch")
    torch.manual_seed(195)
    model = Glm5NextSparseAttention(cfg, 0, backend)
    model.init_weights(torch.device("cpu"), 0.15)
    # Avoid exact ReLU-zero score ties in the independently sorted oracle.
    with torch.no_grad():
        model.q_a_proj.weight.abs_()
        model.indexer.wq_b.weight.abs_()
        model.indexer.weights_proj.weight.abs_()
        model.indexer.k_norm.bias.fill_(4)
    oracle = copy.deepcopy(model)
    full = torch.rand(2, 64, 32) + 0.1
    dy = torch.randn_like(full) / 32
    docs = torch.zeros(2, 64, dtype=torch.int32)
    for row, lengths in enumerate(rows):
        cursor = 0
        for j, n in enumerate(lengths):
            docs[row, cursor : cursor + n] = j + 1
            cursor += n + gap
    context = Glm5NextPackedContext(docs)
    expected_x = full.clone().requires_grad_()
    expected = reference(expected_x, oracle, rows, 4, pool, tail, gap)
    expected.backward(dy)
    x = full.clone().requires_grad_()

    def forward(value: torch.Tensor) -> torch.Tensor:
        """Run the model with the fixed document layout.

        Args:
            value: Packed states [batch, sequence, hidden].

        Returns:
            Attention states [batch, sequence, hidden].
        """
        return model(value, packed_context=context, padding_mask=docs.eq(0))

    with (
        patch.object(layer_module, "is_cudnn_sparse_attention_available", return_value=True),
        patch.object(layer_module, "cudnn_sparse_attention", side_effect=reference_absorbed_attention) as kernel,
    ):
        actual = checkpoint(forward, x, use_reentrant=False) if ac else forward(x)
        # Every nonempty row is one packed attention invocation, including mixed
        # dense/sparse documents. This fails on the old per-document implementation.
        assert kernel.call_count == sum(bool(row) for row in rows)
        actual.backward(dy)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=3e-6, rtol=3e-5)
    for (name, param), (expected_name, ref_param) in zip(
        model.named_parameters(), oracle.named_parameters(), strict=True
    ):
        assert name == expected_name
        if ref_param.grad is None:
            if name.startswith("indexer."):
                assert param.grad is None
            elif param.grad is not None:
                torch.testing.assert_close(param.grad, torch.zeros_like(param.grad), atol=0, rtol=0)
        else:
            torch.testing.assert_close(param.grad, ref_param.grad, atol=3e-6, rtol=3e-5, msg=name)


def _relative_errors(observed: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """Return RMS and worst-case error relative to the oracle's RMS.

    Args:
        observed: Result of any shape.
        reference: FP32 oracle with the same shape.

    Returns:
        ``(rms_error / rms_reference, max_abs_error / rms_reference)``.
    """
    scale = reference.double().square().mean().sqrt().clamp_min(1e-12)
    error = observed.double() - reference.double()
    return float(error.square().mean().sqrt() / scale), float(error.abs().max() / scale)


def _assert_within_bf16_budget(kernel: torch.Tensor, eager: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    """Bound the kernel's error by the error of eager BF16 math on the same inputs.

    Eager BF16 attention rounds its inputs, scores and probabilities like the kernel, so its
    error against the FP32 oracle is the BF16 budget. The kernel may round dS/dQ differently
    (measured up to about 2.5x the eager error on the query projections), so it gets 4x that
    budget with small floors, plus an absolute 5% RMS ceiling that catches layout or scale bugs.

    Args:
        kernel: Kernel output or gradient of any shape.
        eager: Eager BF16 result with the same shape.
        reference: FP32 oracle with the same shape.
        name: Label used in assertion messages.
    """
    assert kernel.shape == eager.shape == reference.shape, name
    assert torch.isfinite(kernel).all(), name
    kernel_rms, kernel_max = _relative_errors(kernel, reference)
    eager_rms, eager_max = _relative_errors(eager, reference)
    assert kernel_rms <= max(4 * eager_rms, 0.01), (name, kernel_rms, eager_rms)
    assert kernel_max <= max(4 * eager_max, 0.06), (name, kernel_max, eager_max)
    assert kernel_rms <= 0.05, (name, kernel_rms)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cudnn_sparse_attention_available(),
    reason="requires CUDA and the cuDNN/FlashMLA sparse-attention kernels",
)
@pytest.mark.parametrize("tail", [True, False])
@pytest.mark.parametrize("rows", [[[5, 3, 17, 7], [3, 21, 5]], [[7, 7, 7, 7, 4], []]])
@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("gap", [0, 3])
def test_packed_attention_cuda_kernel_matches_independent_document_oracle(
    tail: bool, rows: list[list[int]], ac: bool, gap: int
) -> None:
    """Run the real BF16 cuDNN/FlashMLA path on packed documents with padding gaps.

    The kernel is compared with an FP32 oracle on the same BF16-rounded weights and inputs,
    within the error that eager BF16 attention math shows against that oracle. ``index_topk``
    exceeds every document's complete pools, so each query selects all visible pools and
    BF16 score ties cannot change the selected set; KPool selection is covered by the indexer
    tests.
    """
    pool = 4
    cfg = Glm5NextTextConfig(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=32,
        index_kpool=pool,
        index_kpool_always_select_tail=tail,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        dtype=torch.float32,
    )
    backend = Glm5NextBackendConfig(attn="cudnn", linear="torch", rms_norm="torch")
    torch.manual_seed(197)
    model = Glm5NextSparseAttention(cfg, 0, backend)
    model.init_weights(torch.device("cpu"), 0.15)
    model.to(device="cuda", dtype=torch.bfloat16)
    # The oracle runs in FP32 on the same BF16-rounded weights and inputs.
    oracle = copy.deepcopy(model).float()
    eager = copy.deepcopy(model)
    full = (torch.rand(2, 64, 32, device="cuda") + 0.1).to(torch.bfloat16)
    dy = (torch.randn(2, 64, 32, device="cuda") / 32).to(torch.bfloat16)
    docs = torch.zeros(2, 64, dtype=torch.int32, device="cuda")
    for row, lengths in enumerate(rows):
        cursor = 0
        for j, n in enumerate(lengths):
            docs[row, cursor : cursor + n] = j + 1
            cursor += n + gap
    context = Glm5NextPackedContext(docs)
    expected_x = full.float().requires_grad_()
    expected = reference(expected_x, oracle, rows, cfg.index_topk, pool, tail, gap)
    expected.backward(dy.float())
    eager_x = full.clone().requires_grad_()
    with patch.object(layer_module, "cudnn_sparse_attention", side_effect=reference_absorbed_attention):
        eager_output = eager(eager_x, packed_context=context, padding_mask=docs.eq(0))
    eager_output.backward(dy)
    x = full.clone().requires_grad_()

    def forward(value: torch.Tensor) -> torch.Tensor:
        """Run the model with the fixed document layout.

        Args:
            value: Packed BF16 states [batch, sequence, hidden].

        Returns:
            Attention states [batch, sequence, hidden].
        """
        return model(value, packed_context=context, padding_mask=docs.eq(0))

    with patch.object(layer_module, "cudnn_sparse_attention", wraps=layer_module.cudnn_sparse_attention) as kernel:
        actual = checkpoint(forward, x, use_reentrant=False) if ac else forward(x)
        assert kernel.call_count == sum(bool(row) for row in rows)
        actual.backward(dy)

    assert torch.isfinite(actual).all()
    assert actual[docs.eq(0)].eq(0).all(), "padding positions must be exactly zero"
    if not tail:
        # Queries before their document's first complete pool select nothing and must contribute zero.
        for row, lengths in enumerate(rows):
            cursor = 0
            for n in lengths:
                assert actual[row, cursor : cursor + min(n, pool - 1)].eq(0).all()
                cursor += n + gap
    _assert_within_bf16_budget(actual.float(), eager_output.float(), expected, "output")
    _assert_within_bf16_budget(x.grad.float(), eager_x.grad.float(), expected_x.grad, "input gradient")
    for (name, param), (expected_name, ref_param), (_, eager_param) in zip(
        model.named_parameters(), oracle.named_parameters(), eager.named_parameters(), strict=True
    ):
        assert name == expected_name
        if name.startswith("indexer."):
            assert param.grad is None, name
        elif ref_param.grad is not None:
            _assert_within_bf16_budget(param.grad.float(), eager_param.grad.float(), ref_param.grad, name)
