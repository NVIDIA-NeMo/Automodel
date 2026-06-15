# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for gemma4 context-parallel ring attention helpers.

These exercise the CPU-testable pure helpers directly and use mocks/stubs for
the FlexAttention kernel and torch.distributed collectives so the ring logic can
be covered without a GPU or a real process group.
"""

import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa


# ---------------------------------------------------------------------------
# gemma4_vision_group_ids
# ---------------------------------------------------------------------------
def test_vision_group_ids_text_only_all_negative_one():
    ids = torch.zeros(1, 5, dtype=torch.long)  # all text
    out = cpa.gemma4_vision_group_ids(ids)
    assert torch.equal(out, torch.full((1, 5), -1, dtype=out.dtype))


def test_vision_group_ids_two_blocks_get_distinct_ids():
    # text, [vision block 0], text, [vision block 1]
    ids = torch.tensor([[0, 1, 1, 0, 2, 2]], dtype=torch.long)
    out = cpa.gemma4_vision_group_ids(ids)
    assert out.tolist() == [[-1, 0, 0, -1, 1, 1]]


def test_vision_group_ids_block_at_sequence_start():
    ids = torch.tensor([[1, 1, 0]], dtype=torch.long)
    out = cpa.gemma4_vision_group_ids(ids)
    assert out.tolist() == [[0, 0, -1]]


# ---------------------------------------------------------------------------
# _base_gemma4_cp_mask
# ---------------------------------------------------------------------------
def _idx_grid(q_len, kv_len):
    q = torch.arange(q_len).view(q_len, 1).expand(q_len, kv_len)
    kv = torch.arange(kv_len).view(1, kv_len).expand(q_len, kv_len)
    return q, kv


def test_base_mask_causal():
    module = SimpleNamespace(sliding_window=None)
    ctx = SimpleNamespace(seq_global_start=0, is_causal=True)
    q, kv = _idx_grid(4, 4)
    allowed = cpa._base_gemma4_cp_mask(module, ctx, q, kv)
    # lower-triangular (kv <= q)
    assert torch.equal(allowed, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_base_mask_non_causal_all_allowed():
    module = SimpleNamespace(sliding_window=None)
    ctx = SimpleNamespace(seq_global_start=0, is_causal=False)
    q, kv = _idx_grid(3, 3)
    allowed = cpa._base_gemma4_cp_mask(module, ctx, q, kv)
    assert bool(allowed.all())


def test_base_mask_sliding_window_limits_lookback():
    module = SimpleNamespace(sliding_window=2)
    ctx = SimpleNamespace(seq_global_start=0, is_causal=True)
    q, kv = _idx_grid(4, 4)
    allowed = cpa._base_gemma4_cp_mask(module, ctx, q, kv)
    # causal AND (q - kv) < 2
    for qi in range(4):
        for ki in range(4):
            expected = (ki <= qi) and (qi - ki) < 2
            assert bool(allowed[qi, ki]) == expected


def test_base_mask_global_offset_shifts_causal_window():
    module = SimpleNamespace(sliding_window=None)
    ctx = SimpleNamespace(seq_global_start=4, is_causal=True)  # local q starts at global 4
    q, kv = _idx_grid(2, 2)
    allowed = cpa._base_gemma4_cp_mask(module, ctx, q, kv, kv_global_start=0)
    # q_global in {4,5}, kv_global in {0,1} -> all allowed (kv_global <= q_global)
    assert bool(allowed.all())


# ---------------------------------------------------------------------------
# _metadata_like / _detach_metadata
# ---------------------------------------------------------------------------
def test_metadata_like_preserves_shape_and_none():
    md = {"a": torch.randn(2, 3), "b": None}
    out = cpa._metadata_like(md)
    assert out["a"].shape == (2, 3)
    assert out["b"] is None
    # empty_like -> independent storage
    assert out["a"].data_ptr() != md["a"].data_ptr()


def test_detach_metadata_detaches_and_keeps_none():
    t = torch.randn(2, 2, requires_grad=True)
    md = {"a": t, "b": None}
    out = cpa._detach_metadata(md)
    assert out["a"].requires_grad is False
    assert out["a"].is_contiguous()
    assert out["b"] is None


# ---------------------------------------------------------------------------
# _merge_flex_chunk  (online-softmax merge)
# ---------------------------------------------------------------------------
def test_merge_flex_chunk_first_chunk_returns_step():
    out_step = torch.randn(1, 2, 3, 4)
    lse_step = torch.randn(1, 2, 3)
    out, lse = cpa._merge_flex_chunk(None, None, out_step, lse_step)
    assert out is out_step and lse is lse_step


def test_merge_flex_chunk_matches_reference_softmax():
    # Two KV chunks merged online must equal a single full-softmax attention.
    torch.manual_seed(0)
    b, h, q, d = 1, 2, 3, 4
    kv = 5
    query = torch.randn(b, h, q, d)
    key = torch.randn(b, h, kv, d)
    value = torch.randn(b, h, kv, d)
    scale = 1.0 / math.sqrt(d)

    def chunk_out_lse(k, v):
        scores = torch.matmul(query, k.transpose(-1, -2)) * scale
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return out, lse

    o1, l1 = chunk_out_lse(key[:, :, :2], value[:, :, :2])
    o2, l2 = chunk_out_lse(key[:, :, 2:], value[:, :, 2:])
    merged_out, merged_lse = cpa._merge_flex_chunk(o1, l1, o2, l2)

    full_out, full_lse = chunk_out_lse(key, value)
    assert torch.allclose(merged_out, full_out, atol=1e-5)
    assert torch.allclose(merged_lse, full_lse, atol=1e-5)


# ---------------------------------------------------------------------------
# _zero_if_none
# ---------------------------------------------------------------------------
def test_zero_if_none_returns_grad_when_present():
    g = torch.randn(2, 2)
    assert cpa._zero_if_none(g, torch.empty(2, 2)) is g


def test_zero_if_none_returns_zeros_when_none():
    like = torch.randn(2, 3)
    out = cpa._zero_if_none(None, like)
    assert torch.equal(out, torch.zeros_like(like))


# ---------------------------------------------------------------------------
# attach_gemma4_cp_ring_attention
# ---------------------------------------------------------------------------
def test_attach_sets_metadata_keys_and_method():
    module = torch.nn.Linear(2, 2)
    cpa.attach_gemma4_cp_ring_attention(module)
    assert module._cp_manual_metadata_keys == (
        "mm_token_type_ids",
        "_packed_seq_ids",
        "padding_mask",
        "_gemma4_vision_group_ids",
    )
    assert module._cp_manual_metadata_seq_dims == {
        "mm_token_type_ids": 1,
        "_packed_seq_ids": 1,
        "padding_mask": 1,
        "_gemma4_vision_group_ids": 1,
    }
    assert callable(module.run_cp_manual_attention)
    assert callable(module.setup_cp_attention)


def test_setup_cp_attention_installs_model_owned_sdpa_swap(monkeypatch):
    """setup_cp_attention swaps SDPA -> ring on the module, captures the attention
    kwargs into _cp_manual_metadata, and restores SDPA after the forward."""
    import torch.nn.functional as F

    captured = {}

    def fake_ring(module, query, key, value, *, cp_mesh, attn_mask, dropout_p, is_causal, scale, enable_gqa, kwargs):
        captured["meta"] = dict(module._cp_manual_metadata)
        captured["uses_hook"] = module._cp_uses_attention_hook
        return torch.zeros_like(query)

    monkeypatch.setattr(cpa, "_gemma4_cp_manual_attention", fake_ring)

    class _Attn(torch.nn.Module):
        def forward(self, q, k, v, **kw):
            return F.scaled_dot_product_attention(q, k, v)

    attn = _Attn()
    cpa.attach_gemma4_cp_ring_attention(attn)  # binds metadata keys + setup_cp_attention
    original = F.scaled_dot_product_attention
    attn.setup_cp_attention(object())  # model installs its own SDPA-swap hook (cp_mesh stubbed)
    assert attn._cp_uses_attention_hook is True

    q = torch.randn(1, 2, 4, 8)
    out = attn(q, q.clone(), q.clone(), mm_token_type_ids=torch.tensor([[1, 0, 1, 0]]))
    assert torch.equal(out, torch.zeros_like(q))  # ring ran -> SDPA was swapped
    assert torch.equal(captured["meta"]["mm_token_type_ids"], torch.tensor([[1, 0, 1, 0]]))
    assert captured["uses_hook"] is True
    # post-hook restored SDPA and cleared the per-call metadata
    assert F.scaled_dot_product_attention is original
    assert attn._cp_manual_metadata == {}


# ---------------------------------------------------------------------------
# _compiled_flex_attention  (caches the compiled wrapper)
# ---------------------------------------------------------------------------
def test_compiled_flex_attention_is_cached():
    module = torch.nn.Linear(2, 2)
    sentinel = object()
    with mock.patch("torch.compile", return_value=sentinel) as compile_mock:
        first = cpa._compiled_flex_attention(module)
        second = cpa._compiled_flex_attention(module)
    assert first is sentinel and second is sentinel
    compile_mock.assert_called_once()  # only compiled once, then cached on the module


# ---------------------------------------------------------------------------
# _collect_ring_kv_chunks  (cp_size == 1 takes no exchange path)
# ---------------------------------------------------------------------------
def test_collect_ring_kv_chunks_single_rank_no_exchange():
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    ctx = SimpleNamespace(
        key=key,
        value=value,
        metadata={"padding_mask": None},
        cp_rank=0,
        cp_size=1,
        cp_group=object(),
    )
    with mock.patch.object(cpa, "_ring_exchange") as ring:
        chunks = cpa._collect_ring_kv_chunks(ctx)
    ring.assert_not_called()
    assert len(chunks) == 1
    owner, k, v, md = chunks[0]
    assert owner == 0
    assert torch.equal(k, key) and torch.equal(v, value)


def test_collect_ring_kv_chunks_two_ranks_rotates_owner():
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    ctx = SimpleNamespace(
        key=key,
        value=value,
        metadata={"_packed_seq_ids": torch.ones(1, 3, dtype=torch.long)},
        cp_rank=0,
        cp_size=2,
        cp_group=object(),
    )
    with mock.patch.object(cpa, "_ring_exchange") as ring:
        chunks = cpa._collect_ring_kv_chunks(ctx)
    # one exchange between the two steps
    ring.assert_called_once()
    assert [c[0] for c in chunks] == [0, 1]  # owner rotates rank-1


# ---------------------------------------------------------------------------
# _ring_exchange / _direct_exchange  (mock torch.distributed)
# ---------------------------------------------------------------------------
def test_ring_exchange_noop_on_empty():
    # returns immediately without touching torch.distributed
    cpa._ring_exchange([], cp_group=object(), cp_rank=0, cp_size=2)


def _patch_distributed():
    """Patch the torch.distributed primitives used by the exchange helpers."""
    reqs = [mock.Mock()]
    p = mock.patch.multiple(
        torch.distributed,
        get_process_group_ranks=mock.DEFAULT,
        P2POp=mock.DEFAULT,
        isend=mock.DEFAULT,
        irecv=mock.DEFAULT,
        batch_isend_irecv=mock.DEFAULT,
    )
    return p, reqs


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_ring_exchange_issues_send_and_recv(cp_rank):
    send_t = torch.randn(2, 2)
    recv_t = torch.empty(2, 2)
    with mock.patch.multiple(
        torch.distributed,
        get_process_group_ranks=mock.MagicMock(return_value=[0, 1]),
        P2POp=mock.MagicMock(side_effect=lambda *a, **k: ("op", a)),
        isend=mock.MagicMock(),
        irecv=mock.MagicMock(),
        batch_isend_irecv=mock.MagicMock(return_value=[mock.Mock()]),
    ):
        cpa._ring_exchange([(send_t, recv_t)], cp_group=object(), cp_rank=cp_rank, cp_size=2)
        # one send + one recv op constructed
        assert torch.distributed.P2POp.call_count == 2
        torch.distributed.batch_isend_irecv.assert_called_once()


def test_direct_exchange_noop_on_empty():
    cpa._direct_exchange([], cp_group=object(), cp_rank=0, send_cp_rank=1, recv_cp_rank=1)


def test_direct_exchange_issues_ops():
    send_t = torch.randn(2, 2)
    recv_t = torch.empty(2, 2)
    with mock.patch.multiple(
        torch.distributed,
        get_process_group_ranks=mock.MagicMock(return_value=[0, 1]),
        P2POp=mock.MagicMock(side_effect=lambda *a, **k: ("op", a)),
        isend=mock.MagicMock(),
        irecv=mock.MagicMock(),
        batch_isend_irecv=mock.MagicMock(return_value=[mock.Mock()]),
    ):
        cpa._direct_exchange([(send_t, recv_t)], cp_group=object(), cp_rank=1, send_cp_rank=0, recv_cp_rank=0)
        assert torch.distributed.P2POp.call_count == 2
        torch.distributed.batch_isend_irecv.assert_called_once()


# ---------------------------------------------------------------------------
# Real FlexAttention paths (CPU eager) — flex chunk / ring forward / autograd /
# manual-attention entry. The "compiled" flex fn is replaced by the eager
# flex_attention so no GPU or torch.compile is needed.
# ---------------------------------------------------------------------------
from torch.nn.attention.flex_attention import flex_attention as _eager_flex  # noqa: E402


def _flex_module(sliding_window=None, use_bidirectional_attention=None):
    """Attention-module stub whose 'compiled' flex fn is the eager kernel."""
    module = SimpleNamespace(
        sliding_window=sliding_window,
        config=SimpleNamespace(use_bidirectional_attention=use_bidirectional_attention),
        _gemma4_cp_compiled_flex_attn=_eager_flex,
    )
    return module


def _make_ctx(
    module,
    *,
    q_heads=2,
    kv_heads=2,
    seq=4,
    head_dim=4,
    cp_size=1,
    cp_rank=0,
    is_causal=True,
    dropout_p=0.0,
    metadata=None,
    scale=None,
):
    torch.manual_seed(0)
    query = torch.randn(1, q_heads, seq, head_dim)
    key = torch.randn(1, kv_heads, seq, head_dim)
    value = torch.randn(1, kv_heads, seq, head_dim)
    return cpa.CPRingAttentionContext(
        module=module,
        query=query,
        key=key,
        value=value,
        cp_mesh=None,
        cp_group=object(),
        cp_size=cp_size,
        cp_rank=cp_rank,
        seq_local=seq,
        seq_full=seq * cp_size,
        seq_global_start=cp_rank * seq,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale if scale is not None else 1.0 / math.sqrt(head_dim),
        enable_gqa=(q_heads != kv_heads),
        kwargs={},
        metadata=metadata if metadata is not None else {},
        metadata_seq_dims={},
    )


def test_flex_chunk_simple_causal_matches_sdpa():
    module = _flex_module()
    ctx = _make_ctx(module, seq=6, head_dim=8)
    out, lse, empty_rows, padded = cpa._run_gemma4_flex_chunk(
        module, ctx, key_chunk=ctx.key, value_chunk=ctx.value, metadata_chunk={}, kv_global_start=0
    )
    assert out.shape == ctx.query.shape
    assert empty_rows is None
    assert padded == 8
    ref = torch.nn.functional.scaled_dot_product_attention(ctx.query, ctx.key, ctx.value, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-4)


def test_flex_chunk_pads_non_power_of_two_head_dim():
    module = _flex_module()
    ctx = _make_ctx(module, seq=4, head_dim=3, scale=None)  # head_dim 3 -> padded 4
    out, lse, _, padded = cpa._run_gemma4_flex_chunk(
        module, ctx, key_chunk=ctx.key, value_chunk=ctx.value, metadata_chunk={}, kv_global_start=0
    )
    assert padded == 4
    assert out.shape[-1] == 3  # sliced back to original head_dim


def test_flex_chunk_gqa_kv_heads_smaller():
    module = _flex_module()
    ctx = _make_ctx(module, q_heads=4, kv_heads=2, seq=4, head_dim=8)
    out, _, _, _ = cpa._run_gemma4_flex_chunk(
        module, ctx, key_chunk=ctx.key, value_chunk=ctx.value, metadata_chunk={}, kv_global_start=0
    )
    assert out.shape == ctx.query.shape


def test_flex_chunk_padding_mask_zeros_empty_query_rows():
    module = _flex_module()
    padding_mask = torch.tensor([[False, False, True, True]])  # last two rows are padding
    ctx = _make_ctx(module, seq=4, head_dim=8, metadata={"padding_mask": padding_mask})
    out, _, empty_rows, _ = cpa._run_gemma4_flex_chunk(
        module,
        ctx,
        key_chunk=ctx.key,
        value_chunk=ctx.value,
        metadata_chunk={"padding_mask": padding_mask},
        kv_global_start=0,
    )
    assert empty_rows is not None and bool(empty_rows.any())
    # padded query rows must be zeroed in the output
    assert torch.count_nonzero(out[:, :, 2:, :]) == 0


def test_flex_chunk_packed_seq_ids_branch():
    module = _flex_module()
    packed = torch.tensor([[1, 1, 2, 2]])
    ctx = _make_ctx(module, seq=4, head_dim=8, metadata={"_packed_seq_ids": packed})
    out, _, empty_rows, _ = cpa._run_gemma4_flex_chunk(
        module,
        ctx,
        key_chunk=ctx.key,
        value_chunk=ctx.value,
        metadata_chunk={"_packed_seq_ids": packed},
        kv_global_start=0,
    )
    assert out.shape == ctx.query.shape


def test_flex_chunk_vision_bidirectional_branch():
    module = _flex_module(sliding_window=1024, use_bidirectional_attention="vision")
    mm = torch.tensor([[0, 1, 1, 0]])  # one vision block
    ctx = _make_ctx(module, seq=4, head_dim=8, metadata={"mm_token_type_ids": mm})
    out, _, _, _ = cpa._run_gemma4_flex_chunk(
        module,
        ctx,
        key_chunk=ctx.key,
        value_chunk=ctx.value,
        metadata_chunk={"mm_token_type_ids": mm},
        kv_global_start=0,
    )
    assert out.shape == ctx.query.shape


def test_flex_chunk_kernel_options_typeerror_falls_back():
    # head_dim>256 -> use_small_flex_blocks -> kernel_options added; stub raises
    # TypeError mentioning kernel_options on first call, succeeds without it.
    calls = {"n": 0}

    def stub(q, k, v, return_lse=True, **kw):
        calls["n"] += 1
        if "kernel_options" in kw:
            raise TypeError("unexpected keyword argument 'kernel_options'")
        b, h, s, d = q.shape
        return torch.zeros(b, h, s, d), torch.zeros(b, h, s)

    module = SimpleNamespace(
        sliding_window=None,
        config=SimpleNamespace(use_bidirectional_attention=None),
        _gemma4_cp_compiled_flex_attn=stub,
    )
    ctx = _make_ctx(module, seq=4, head_dim=300)  # padded 512 > 256
    out, _, _, padded = cpa._run_gemma4_flex_chunk(
        module, ctx, key_chunk=ctx.key, value_chunk=ctx.value, metadata_chunk={}, kv_global_start=0
    )
    assert padded == 512
    assert calls["n"] == 2  # first call raised, retried without kernel_options
    assert out.shape[-1] == 300


def test_flex_chunk_wraps_unexpected_error_in_runtimeerror():
    def boom(*a, **k):
        raise ValueError("kaboom")

    module = SimpleNamespace(
        sliding_window=None,
        config=SimpleNamespace(use_bidirectional_attention=None),
        _gemma4_cp_compiled_flex_attn=boom,
    )
    ctx = _make_ctx(module, seq=4, head_dim=8)
    with pytest.raises(RuntimeError, match="Gemma4 CP ring requires FlexAttention"):
        cpa._run_gemma4_flex_chunk(
            module, ctx, key_chunk=ctx.key, value_chunk=ctx.value, metadata_chunk={}, kv_global_start=0
        )


def test_ring_forward_single_rank_matches_sdpa():
    cpa._GEMMA4_CP_FLEX_RING_OK_LOGGED = False  # exercise the one-time log path
    module = _flex_module()
    ctx = _make_ctx(module, seq=6, head_dim=8)
    out = cpa._run_gemma4_cp_ring_attention_forward(module, ctx)
    ref = torch.nn.functional.scaled_dot_product_attention(ctx.query, ctx.key, ctx.value, is_causal=True)
    assert out.shape == ctx.query.shape
    assert torch.allclose(out, ref, atol=1e-4)
    assert cpa._GEMMA4_CP_FLEX_RING_OK_LOGGED is True


def test_ring_forward_rejects_dropout():
    module = _flex_module()
    ctx = _make_ctx(module, dropout_p=0.1)
    with pytest.raises(NotImplementedError, match="dropout"):
        cpa._run_gemma4_cp_ring_attention_forward(module, ctx)


def test_ring_forward_raises_when_no_chunks():
    module = _flex_module()
    ctx = _make_ctx(module)
    with mock.patch.object(cpa, "_collect_ring_kv_chunks", return_value=[]):
        with pytest.raises(RuntimeError, match="no output chunks"):
            cpa._run_gemma4_cp_ring_attention_forward(module, ctx)


def _sdpa_flex_surrogate(module, ctx, *, key_chunk, value_chunk, metadata_chunk, kv_global_start):
    """Differentiable stand-in for _run_gemma4_flex_chunk.

    Eager flex_attention has no CPU backward, so the autograd-Function tests
    swap in this causal-SDPA surrogate to exercise the ring autograd plumbing
    (save_for_backward, chunk grad collection, cross-rank grad exchange).
    """
    scale = ctx.scale
    scores = torch.matmul(ctx.query, key_chunk.transpose(-1, -2)) * scale
    q_global = torch.arange(ctx.seq_local).view(-1, 1) + ctx.seq_global_start
    kv_global = torch.arange(key_chunk.shape[2]).view(1, -1) + kv_global_start
    allowed = kv_global <= q_global
    scores = scores.masked_fill(~allowed, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    out = torch.matmul(weights, value_chunk)
    return out, lse, None, ctx.query.shape[-1]


def test_ring_autograd_single_rank_grads_match_sdpa(monkeypatch):
    monkeypatch.setattr(cpa, "_run_gemma4_flex_chunk", _sdpa_flex_surrogate)
    module = _flex_module()
    ctx = _make_ctx(module, seq=6, head_dim=8)
    q = ctx.query.clone().requires_grad_(True)
    k = ctx.key.clone().requires_grad_(True)
    v = ctx.value.clone().requires_grad_(True)
    ring_ctx = cpa.replace(ctx, query=q, key=k, value=v)
    out = cpa._run_gemma4_cp_ring_attention(module, ring_ctx)
    out.sum().backward()

    qr = ctx.query.clone().requires_grad_(True)
    kr = ctx.key.clone().requires_grad_(True)
    vr = ctx.value.clone().requires_grad_(True)
    ref = torch.nn.functional.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    ref.sum().backward()
    assert torch.allclose(out, ref, atol=1e-4)
    assert torch.allclose(q.grad, qr.grad, atol=1e-4)
    assert torch.allclose(k.grad, kr.grad, atol=1e-4)
    assert torch.allclose(v.grad, vr.grad, atol=1e-4)


def test_ring_autograd_two_ranks_exchanges_grads(monkeypatch):
    # cp_size=2 exercises the ring KV collection and the backward grad
    # cross-rank exchange. Collectives are replaced by local copies so a single
    # process can run both ring steps.
    def fake_exchange(tensors, **kwargs):
        for send_t, recv_t in tensors:
            recv_t.copy_(send_t)

    monkeypatch.setattr(cpa, "_run_gemma4_flex_chunk", _sdpa_flex_surrogate)
    monkeypatch.setattr(cpa, "_ring_exchange", fake_exchange)
    monkeypatch.setattr(cpa, "_direct_exchange", fake_exchange)

    module = _flex_module()
    ctx = _make_ctx(module, seq=4, head_dim=8, cp_size=2, cp_rank=0)
    q = ctx.query.clone().requires_grad_(True)
    k = ctx.key.clone().requires_grad_(True)
    v = ctx.value.clone().requires_grad_(True)
    ring_ctx = cpa.replace(ctx, query=q, key=k, value=v)
    out = cpa._run_gemma4_cp_ring_attention(module, ring_ctx)
    out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert not torch.isnan(k.grad).any()


def test_ring_backward_raises_when_no_chunks(monkeypatch):
    monkeypatch.setattr(cpa, "_run_gemma4_flex_chunk", _sdpa_flex_surrogate)
    module = _flex_module()
    ctx = _make_ctx(module, seq=4, head_dim=8)
    q = ctx.query.clone().requires_grad_(True)
    ring_ctx = cpa.replace(ctx, query=q, key=ctx.key.clone(), value=ctx.value.clone())
    out = cpa._run_gemma4_cp_ring_attention(module, ring_ctx)
    # force the backward chunk collection to be empty
    monkeypatch.setattr(cpa, "_collect_ring_kv_chunks", lambda c: [])
    with pytest.raises(RuntimeError, match="no output chunks"):
        out.sum().backward()


def test_flex_chunk_scale_none_with_head_dim_padding():
    # scale=None AND padded head_dim != orig -> default scale is derived (line 269)
    module = _flex_module()
    ctx = cpa.replace(_make_ctx(module, seq=4, head_dim=3), scale=None)
    out, _, _, padded = cpa._run_gemma4_flex_chunk(
        module, ctx, key_chunk=ctx.key, value_chunk=ctx.value, metadata_chunk={}, kv_global_start=0
    )
    assert padded == 4 and out.shape[-1] == 3


def test_flex_chunk_typeerror_without_kernel_options_reraised():
    # TypeError not mentioning kernel_options must propagate (line 295) -> RuntimeError wrap
    def stub(q, k, v, return_lse=True, **kw):
        raise TypeError("totally unrelated")

    module = SimpleNamespace(
        sliding_window=None,
        config=SimpleNamespace(use_bidirectional_attention=None),
        _gemma4_cp_compiled_flex_attn=stub,
    )
    ctx = _make_ctx(module, seq=4, head_dim=300)  # small-blocks path adds kernel_options
    with pytest.raises(RuntimeError, match="Gemma4 CP ring requires FlexAttention"):
        cpa._run_gemma4_flex_chunk(
            module, ctx, key_chunk=ctx.key, value_chunk=ctx.value, metadata_chunk={}, kv_global_start=0
        )


def test_ring_forward_masks_empty_query_rows():
    # padding_mask -> empty_query_rows -> forward masks the accumulated output (line 368)
    module = _flex_module()
    padding_mask = torch.tensor([[False, False, True, True]])
    ctx = _make_ctx(module, seq=4, head_dim=8, metadata={"padding_mask": padding_mask})
    out = cpa._run_gemma4_cp_ring_attention_forward(module, ctx)
    assert torch.count_nonzero(out[:, :, 2:, :]) == 0


def test_ring_backward_masks_empty_query_rows(monkeypatch):
    # surrogate that reports empty query rows -> backward masked_fill (line 448)
    def surrogate_empty(module, ctx, *, key_chunk, value_chunk, metadata_chunk, kv_global_start):
        out, lse, _, padded = _sdpa_flex_surrogate(
            module,
            ctx,
            key_chunk=key_chunk,
            value_chunk=value_chunk,
            metadata_chunk=metadata_chunk,
            kv_global_start=kv_global_start,
        )
        empty = torch.zeros(ctx.query.shape[0], ctx.seq_local, dtype=torch.bool)
        empty[:, -1] = True
        return out, lse, empty, padded

    monkeypatch.setattr(cpa, "_run_gemma4_flex_chunk", surrogate_empty)
    module = _flex_module()
    ctx = _make_ctx(module, seq=4, head_dim=8)
    q = ctx.query.clone().requires_grad_(True)
    ring_ctx = cpa.replace(ctx, query=q, key=ctx.key.clone(), value=ctx.value.clone())
    out = cpa._run_gemma4_cp_ring_attention(module, ring_ctx)
    out.sum().backward()
    assert q.grad is not None


def test_manual_attention_entry_single_rank(monkeypatch):
    module = _flex_module()
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    cp_mesh = SimpleNamespace(get_group=lambda: object(), size=lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    out = cpa._gemma4_cp_manual_attention(
        module,
        q,
        k,
        v,
        cp_mesh=cp_mesh,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=None,
        enable_gqa=False,
        kwargs={},
    )
    assert out.shape == q.shape


def test_manual_attention_entry_sets_gqa_when_head_counts_differ(monkeypatch):
    module = _flex_module()
    q = torch.randn(1, 4, 4, 8)  # 4 q-heads
    k = torch.randn(1, 2, 4, 8)  # 2 kv-heads -> enable_gqa forced True
    v = torch.randn(1, 2, 4, 8)
    cp_mesh = SimpleNamespace(get_group=lambda: object(), size=lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    out = cpa._gemma4_cp_manual_attention(
        module,
        q,
        k,
        v,
        cp_mesh=cp_mesh,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=None,
        enable_gqa=False,
        kwargs={},
    )
    assert out.shape == q.shape
