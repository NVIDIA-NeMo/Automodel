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

"""CPU unit tests for the Gemma4 context-parallel ring attention -- FFPA CuTeDSL backend.

Companion to ``test_cp_attention.py`` (the Flex ring backend + the helpers shared by
both backends: vision group ids, base CP mask, metadata/exchange plumbing, block-mask
cache, FSDP guard, model-owned SDPA-swap wiring). The FFPA CuTeDSL kernel is CUDA-only,
so these substitute differentiable flash-style surrogates over the THD / dense layouts
and validate the FFPA ring geometry (segment construction, dense/varlen dispatch, online
merge, manual backward, all-pad safety, segment cache). Real-kernel parity lives in the
GPU functional tests.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention as _eager_flex

from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa


def _flex_module(sliding_window=None):
    return SimpleNamespace(
        sliding_window=sliding_window,
        config=SimpleNamespace(use_bidirectional_attention=None),
        _gemma4_cp_compiled_flex_attn=_eager_flex,
    )


def _make_ctx(module, *, q_heads=2, kv_heads=2, seq=4, head_dim=4, cp_size=1, cp_rank=0, metadata=None):
    torch.manual_seed(0)
    return cpa.CPRingAttentionContext(
        module=module,
        query=torch.randn(1, q_heads, seq, head_dim),
        key=torch.randn(1, kv_heads, seq, head_dim),
        value=torch.randn(1, kv_heads, seq, head_dim),
        cp_mesh=None,
        cp_group=object(),
        cp_size=cp_size,
        cp_rank=cp_rank,
        seq_local=seq,
        seq_full=seq * cp_size,
        seq_global_start=cp_rank * seq,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=1.0 / math.sqrt(head_dim),
        enable_gqa=(q_heads != kv_heads),
        kwargs={},
        metadata=metadata if metadata is not None else {},
        metadata_seq_dims={},
    )


def _requires_grad(ctx):
    q = ctx.query.clone().requires_grad_(True)
    k = ctx.key.clone().requires_grad_(True)
    v = ctx.value.clone().requires_grad_(True)
    return q, k, v, cpa.replace(ctx, query=q, key=k, value=v)


# Flash-style varlen surrogates over packed [T, H, D] + cu_seqlens
def _segments(seg):
    cu_q, cu_k = seg["cu_q"].tolist(), seg["cu_k"].tolist()
    for i in range(len(cu_q) - 1):
        yield cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]


def _ffpa_varlen_fwd_surrogate(q_pack, k_pack, v_pack, seg, *, scale, causal):
    Tq, Hq, D = q_pack.shape
    g = Hq // k_pack.shape[1]
    out = torch.zeros(Tq, Hq, D, dtype=q_pack.dtype)
    lse = torch.full((Hq, Tq), float("-inf"), dtype=torch.float32)
    for a, b, c, d in _segments(seg):
        qs = q_pack[a:b].float()
        ks = k_pack[c:d].repeat_interleave(g, dim=1).float()
        vs = v_pack[c:d].repeat_interleave(g, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            nq, nk = b - a, d - c
            qpos, kpos = torch.arange(nq).view(-1, 1), torch.arange(nk).view(1, -1)
            scores = scores.masked_fill(kpos > qpos + (nk - nq), float("-inf"))
        lse[:, a:b] = torch.logsumexp(scores, dim=-1)
        out[a:b] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vs).to(q_pack.dtype)
    return out, lse


def _ffpa_varlen_bwd_surrogate(grad_out, q_pack, k_pack, v_pack, out_global, lse_global, seg, *, scale, causal):
    Tq, Hq, D = q_pack.shape
    Tk, Hkv, _ = k_pack.shape
    g = Hq // Hkv
    dq = torch.zeros(Tq, Hq, D)
    dk = torch.zeros(Tk, Hkv, D)
    dv = torch.zeros(Tk, Hkv, D)
    for a, b, c, d in _segments(seg):
        qs = q_pack[a:b].float()
        ks = k_pack[c:d].repeat_interleave(g, dim=1).float()
        vs = v_pack[c:d].repeat_interleave(g, dim=1).float()
        go, og, lg = grad_out[a:b].float(), out_global[a:b].float(), lse_global[:, a:b]
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            nq, nk = b - a, d - c
            qpos, kpos = torch.arange(nq).view(-1, 1), torch.arange(nk).view(1, -1)
            scores = scores.masked_fill(kpos > qpos + (nk - nq), float("-inf"))
        p = torch.exp(scores - lg.unsqueeze(-1))
        dp = torch.einsum("qhd,khd->hqk", go, vs)
        delta = (go * og).sum(dim=-1).transpose(0, 1)
        ds = p * (dp - delta.unsqueeze(-1))
        dq[a:b] = scale * torch.einsum("hqk,khd->qhd", ds, ks)
        dk[c:d] = (scale * torch.einsum("hqk,qhd->khd", ds, qs)).view(d - c, Hkv, g, D).sum(2)
        dv[c:d] = torch.einsum("hqk,qhd->khd", p, go).view(d - c, Hkv, g, D).sum(2)
    return dq.to(q_pack.dtype), dk.to(k_pack.dtype), dv.to(v_pack.dtype)


# Flash-style *dense* surrogates over [B, H, S, D] (single-document ring path)
def _ffpa_dense_fwd_surrogate(q, k, v, *, scale, causal):
    B, Hq, S, D = q.shape
    g = Hq // k.shape[1]
    Sk = k.shape[2]
    ks = k.repeat_interleave(g, dim=1).float()
    vs = v.repeat_interleave(g, dim=1).float()
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), ks) * scale
    if causal:
        qpos, kpos = torch.arange(S).view(-1, 1), torch.arange(Sk).view(1, -1)
        scores = scores.masked_fill(kpos > qpos, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B, Hq, S]
    out = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), vs).to(q.dtype)
    return out, lse


def _ffpa_dense_bwd_surrogate(grad_out, q, k, v, out_global, lse_global, *, scale, causal):
    B, Hq, S, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    g = Hq // Hkv
    ks = k.repeat_interleave(g, dim=1).float()
    vs = v.repeat_interleave(g, dim=1).float()
    go, og = grad_out.float(), out_global.float()
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), ks) * scale
    if causal:
        qpos, kpos = torch.arange(S).view(-1, 1), torch.arange(Sk).view(1, -1)
        scores = scores.masked_fill(kpos > qpos, float("-inf"))
    p = torch.exp(scores - lse_global.unsqueeze(-1))  # [B, Hq, S, Sk]
    dp = torch.einsum("bhqd,bhkd->bhqk", go, vs)
    delta = (go * og).sum(dim=-1)  # [B, Hq, S]
    ds = p * (dp - delta.unsqueeze(-1))
    dq = scale * torch.einsum("bhqk,bhkd->bhqd", ds, ks)
    dk = (scale * torch.einsum("bhqk,bhqd->bhkd", ds, q.float())).view(B, Hkv, g, Sk, D).sum(2)
    dv = torch.einsum("bhqk,bhqd->bhkd", p, go).view(B, Hkv, g, Sk, D).sum(2)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _packed_causal_ref(query, key, value, doc_ids, scale):
    """Block-diagonal (same-document) causal SDPA reference; pad rows zeroed."""
    _, _, Sq, _ = query.shape
    Sk = key.shape[2]
    scores = torch.einsum("bhqd,bhkd->bhqk", query.float(), key.float()) * scale
    qpos, kpos = torch.arange(Sq).view(-1, 1), torch.arange(Sk).view(1, -1)
    q_doc, k_doc = doc_ids[0, :Sq].view(-1, 1), doc_ids[0, :Sk].view(1, -1)
    allowed = (kpos <= qpos) & (q_doc == k_doc) & (q_doc > 0)
    scores = scores.masked_fill(~allowed.view(1, 1, Sq, Sk), float("-inf"))
    p = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    out = torch.einsum("bhqk,bhkd->bhqd", p, value.float())
    return (out * (q_doc > 0).view(1, 1, Sq, 1)).to(query.dtype)


def _sdpa_flex_surrogate(module, ctx, *, key_chunk, value_chunk, metadata_chunk, kv_global_start):
    """Differentiable causal-SDPA stand-in for _run_gemma4_flex_chunk (eager flex has no CPU backward)."""
    scores = torch.matmul(ctx.query, key_chunk.transpose(-1, -2)) * ctx.scale
    q_global = torch.arange(ctx.seq_local, device=ctx.query.device).view(-1, 1) + ctx.seq_global_start
    kv_global = torch.arange(key_chunk.shape[2], device=key_chunk.device).view(1, -1) + kv_global_start
    scores = scores.masked_fill(kv_global > q_global, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.matmul(weights, value_chunk), lse, None, ctx.query.shape[-1]


# online-softmax merge
def test_merge_flex_chunk_matches_full_softmax():
    torch.manual_seed(0)
    query, key, value = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    scale = 0.5

    def out_lse(k, v):
        s = torch.matmul(query, k.transpose(-1, -2)) * scale
        return torch.matmul(torch.softmax(s, dim=-1), v), torch.logsumexp(s, dim=-1)

    o1, l1 = out_lse(key[:, :, :2], value[:, :, :2])
    o2, l2 = out_lse(key[:, :, 2:], value[:, :, 2:])
    merged_out, merged_lse = cpa._merge_flex_chunk(o1, l1, o2, l2)
    full_out, full_lse = out_lse(key, value)
    assert torch.allclose(merged_out, full_out, atol=1e-5)
    assert torch.allclose(merged_lse, full_lse, atol=1e-5)


# _build_packed_ring_segments -- cross-shard per-document pairing
@pytest.mark.parametrize(
    "q_ids,k_ids,exp_q,exp_k,exp_cuq,exp_cuk",
    [
        ([1, 1, 2, 2], [1, 1, 2, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 4], [0, 2, 4]),  # diagonal two docs
        ([1, 1, 2, 2], [2, 2, 3, 3], [2, 3], [0, 1], [0, 2], [0, 2]),  # only shared doc 2
        ([1, 1, 0, 0], [1, 1, 0, 0], [0, 1], [0, 1], [0, 2], [0, 2]),  # padding (doc 0) excluded
        ([1, 2], [1, 1], [0], [0, 1], [0, 1], [0, 2]),  # straddling doc: unequal q/k lengths
    ],
)
def test_build_packed_ring_segments(q_ids, k_ids, exp_q, exp_k, exp_cuq, exp_cuk):
    seg = cpa._build_packed_ring_segments(torch.tensor([q_ids]), torch.tensor([k_ids]))
    assert seg["q_index"].tolist() == exp_q
    assert seg["k_index"].tolist() == exp_k
    assert seg["cu_q"].tolist() == exp_cuq
    assert seg["cu_k"].tolist() == exp_cuk


def test_build_packed_ring_segments_no_shared_doc_returns_none():
    assert cpa._build_packed_ring_segments(torch.tensor([[1, 1]]), torch.tensor([[2, 2]])) is None


@pytest.mark.parametrize(
    "q_ids,k_ids,exp_local,exp_cross",
    [
        ([1, 1, 1, 1], [1, 1, 1, 1], True, True),  # one full document, no pad => dense in either role
        ([1, 1, 2, 2], [1, 1, 2, 2], False, False),  # two docs in the q shard => cross-doc masking needed
        ([1, 1, 0, 0], [1, 1, 0, 0], True, False),  # single doc + pad suffix: local dense, cross needs full k
        ([1, 1, 1, 1], [2, 1, 1, 1], True, False),  # k is not one full doc => only the local chunk is dense
    ],
)
def test_build_packed_ring_segments_dense_flags(q_ids, k_ids, exp_local, exp_cross):
    seg = cpa._build_packed_ring_segments(torch.tensor([q_ids]), torch.tensor([k_ids]))
    assert seg["dense_local"] is exp_local
    assert seg["dense_cross"] is exp_cross


def test_dense_ring_single_doc_matches_causal_sdpa_fwd_bwd(monkeypatch):
    # Single-document (unpacked) shard takes the dense path; ring fwd+bwd must equal plain causal SDPA.
    monkeypatch.setattr(cpa, "_ffpa_dense_fwd", _ffpa_dense_fwd_surrogate)
    monkeypatch.setattr(cpa, "_ffpa_dense_bwd", _ffpa_dense_bwd_surrogate)
    doc_ids = torch.ones(1, 6, dtype=torch.long)
    ctx = _make_ctx(_flex_module(), q_heads=4, kv_heads=2, seq=6, head_dim=8, metadata={"_packed_seq_ids": doc_ids})

    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4FFPAVarlenRingAttention.apply(q, k, v, ring_ctx)
    out.sum().backward()

    qr = ctx.query.clone().requires_grad_(True)
    kr = ctx.key.clone().requires_grad_(True)
    vr = ctx.value.clone().requires_grad_(True)
    ref = torch.nn.functional.scaled_dot_product_attention(qr, kr, vr, is_causal=True, scale=ctx.scale, enable_gqa=True)
    ref.sum().backward()

    assert torch.allclose(out, ref, atol=1e-5)
    for got, exp in ((q, qr), (k, kr), (v, vr)):
        assert torch.allclose(got.grad, exp.grad, atol=1e-5)


# eligibility -- rank-uniform gate (collective safety)
def _elig_module(sliding_window=None, use_ffpa=True):
    return SimpleNamespace(_gemma4_cp_use_ffpa=use_ffpa, sliding_window=sliding_window)


def _elig_ctx(*, head_dim=512, packed=True):
    md = {"_packed_seq_ids": torch.tensor([[1, 1, 1, 1]])} if packed else {}
    return SimpleNamespace(
        is_causal=True, metadata=md, query=torch.zeros(1, 1, 4, head_dim, dtype=torch.bfloat16), scale=0.0625
    )


@pytest.mark.parametrize(
    "avail,mod_kw,ctx_kw,expected",
    [
        (True, {}, {}, True),  # happy path
        (True, {}, {"packed": False}, False),  # no _packed_seq_ids
        (True, {"sliding_window": 512}, {}, False),  # sliding-window layer
        (True, {"use_ffpa": False}, {}, False),  # backend flag off
        (True, {}, {"head_dim": 256}, False),  # wrong head_dim
        (False, {}, {}, False),  # kernel unavailable
    ],
)
def test_ring_use_ffpa_varlen(monkeypatch, avail, mod_kw, ctx_kw, expected):
    monkeypatch.setattr(cpa, "_ffpa_varlen_ring_available", lambda: avail)
    assert cpa._ring_use_ffpa_varlen(_elig_module(**mod_kw), _elig_ctx(**ctx_kw)) is expected


# Flex ring forward (eager flex kernel, no GPU)
def test_flex_ring_forward_single_rank_matches_sdpa():
    module = _flex_module()
    ctx = _make_ctx(module, seq=6, head_dim=8)
    out = cpa._run_gemma4_cp_ring_attention_forward(module, ctx)
    ref = torch.nn.functional.scaled_dot_product_attention(ctx.query, ctx.key, ctx.value, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-4)


# FFPA varlen ring autograd (surrogate kernel)
def test_varlen_ring_packed_matches_blockdiag_sdpa_fwd_bwd(monkeypatch):
    monkeypatch.setattr(cpa, "_ffpa_varlen_forward_chunk", _ffpa_varlen_fwd_surrogate)
    monkeypatch.setattr(cpa, "_ffpa_varlen_backward_chunk", _ffpa_varlen_bwd_surrogate)
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 0]])  # doc1(3) | doc2(2) | pad(1)
    ctx = _make_ctx(_flex_module(), seq=6, head_dim=8, metadata={"_packed_seq_ids": doc_ids})

    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4FFPAVarlenRingAttention.apply(q, k, v, ring_ctx)
    out.sum().backward()

    qr = ctx.query.clone().requires_grad_(True)
    kr = ctx.key.clone().requires_grad_(True)
    vr = ctx.value.clone().requires_grad_(True)
    ref = _packed_causal_ref(qr, kr, vr, doc_ids, ctx.scale)
    ref.sum().backward()

    real = doc_ids[0] > 0  # compare real-token rows only (pad rows are 0 on both sides)
    assert torch.allclose(out, ref, atol=1e-5)
    for got, exp in ((q, qr), (k, kr), (v, vr)):
        assert torch.allclose(got.grad[:, :, real], exp.grad[:, :, real], atol=1e-5)


def test_varlen_ring_all_pad_shard_zeros_and_zero_grads(monkeypatch):
    # A full-pad CP shard (no shared doc) must return 0 / zero-grad, NOT raise:
    # a one-rank raise desyncs the rank-uniform dK/dV p2p and hangs multi-GPU training.
    monkeypatch.setattr(cpa, "_ffpa_varlen_forward_chunk", _ffpa_varlen_fwd_surrogate)
    monkeypatch.setattr(cpa, "_ffpa_varlen_backward_chunk", _ffpa_varlen_bwd_surrogate)
    ctx = _make_ctx(
        _flex_module(), seq=4, head_dim=8, metadata={"_packed_seq_ids": torch.zeros(1, 4, dtype=torch.long)}
    )
    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4FFPAVarlenRingAttention.apply(q, k, v, ring_ctx)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert torch.count_nonzero(q.grad) == torch.count_nonzero(k.grad) == torch.count_nonzero(v.grad) == 0


# per-step segment cache
def test_ring_segment_cache_builds_once_then_hits(monkeypatch):
    cpa._RING_SEGMENT_CACHE.clear()
    cpa._RING_SEGMENT_GEN[0] = None
    calls = []
    real = cpa._build_packed_ring_segments

    def counting(q, k):
        calls.append(1)
        return real(q, k)

    monkeypatch.setattr(cpa, "_build_packed_ring_segments", counting)
    ids = torch.tensor([[1, 1, 2, 2]])  # same object => same data_ptr generation
    for _ in range(3):
        cpa._cached_ring_segments(ids, ids, cp_rank=0, owner=0, cp_size=1)
    assert len(calls) == 1


# dispatch
def test_ring_dispatch_routes_to_flex_when_not_eligible(monkeypatch):
    monkeypatch.setattr(cpa, "_ring_use_ffpa_varlen", lambda m, c: False)
    monkeypatch.setattr(cpa, "_run_gemma4_flex_chunk", _sdpa_flex_surrogate)
    ctx = _make_ctx(_flex_module(), seq=4, head_dim=8)
    assert cpa._run_gemma4_cp_ring_attention(ctx.module, ctx).shape == ctx.query.shape


def test_ring_dispatch_routes_to_varlen_when_eligible(monkeypatch):
    # Two distinct documents in the shard => not single-doc => varlen (THD) path.
    monkeypatch.setattr(cpa, "_ring_use_ffpa_varlen", lambda m, c: True)
    monkeypatch.setattr(cpa, "_ffpa_varlen_forward_chunk", _ffpa_varlen_fwd_surrogate)
    ctx = _make_ctx(_flex_module(), seq=4, head_dim=8, metadata={"_packed_seq_ids": torch.tensor([[1, 1, 2, 2]])})
    assert cpa._run_gemma4_cp_ring_attention(ctx.module, ctx).shape == ctx.query.shape


def test_ring_dispatch_routes_to_dense_when_single_doc(monkeypatch):
    # One full document (no pad) => dense kernel path (zero gather/scatter).
    monkeypatch.setattr(cpa, "_ring_use_ffpa_varlen", lambda m, c: True)
    monkeypatch.setattr(cpa, "_ffpa_dense_fwd", _ffpa_dense_fwd_surrogate)
    ctx = _make_ctx(_flex_module(), seq=4, head_dim=8, metadata={"_packed_seq_ids": torch.tensor([[1, 1, 1, 1]])})
    assert cpa._run_gemma4_cp_ring_attention(ctx.module, ctx).shape == ctx.query.shape


# model wiring
def test_attach_sets_ffpa_flag_and_cp_seam():
    attn = torch.nn.Module()
    cpa.attach_gemma4_cp_ring_attention(attn, use_ffpa=True)
    assert attn._gemma4_cp_use_ffpa is True
    assert callable(attn.setup_cp_attention) and callable(attn.run_cp_manual_attention)

    default = torch.nn.Module()
    cpa.attach_gemma4_cp_ring_attention(default)
    assert default._gemma4_cp_use_ffpa is False
