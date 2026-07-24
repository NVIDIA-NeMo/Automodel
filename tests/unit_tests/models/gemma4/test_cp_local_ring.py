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

"""CPU unit tests for the Gemma4 CP sliding-window local-kernel ring (cp_local_ring).

Covers the sliding-window comms reduction (``_ring_num_prior_chunks`` and its symmetric
use in forward collection / backward dK/dV routing) and the FlashAttention-2 local-kernel
backend. The FA2 kernel is CUDA-only, so the four ``_fa_ll_*`` entry points are replaced by
differentiable sliding-window SDPA surrogates (autograd via ``torch.autograd.grad``); the
tests then exercise the neighborhood concat, THD segment gather/scatter, no-recompute
save/restore, all-pad safety, owner grad routing, and the reduction geometry. Real-kernel
layout/window/numerics parity belongs in the GPU tier.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa
from nemo_automodel.components.models.gemma4_moe import cp_local_ring as clr


def _sliding_module(sliding_window):
    return SimpleNamespace(sliding_window=sliding_window, _gemma4_cp_sliding_backend="fa")


def _make_ctx(module, *, q_heads=2, kv_heads=2, seq=4, head_dim=8, cp_size=1, cp_rank=0, metadata=None):
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


# ---------------------------------------------------------------------------
# Differentiable sliding-window SDPA references + FA2 low-level surrogates.
# ---------------------------------------------------------------------------
def _br_window(nq, nk, win, device):
    """Bottom-right causal + left-window boolean mask, shape [nq, nk] (q is the kv suffix)."""
    qglob = torch.arange(nq, device=device).view(-1, 1) + (nk - nq)
    kpos = torch.arange(nk, device=device).view(1, -1)
    return (kpos <= qglob) & (qglob - kpos <= win - 1)


def _masked_attn(q, k, v, scale, allowed):
    """GQA attention with a boolean ``allowed`` [Sq, Sk] mask. q [B,Hq,Sq,D], k/v [B,Hkv,Sk,D]."""
    g = q.shape[1] // k.shape[1]
    ks, vs = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, ks) * scale
    scores = scores.masked_fill(~allowed.view(1, 1, *allowed.shape), float("-inf"))
    p = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.einsum("bhqk,bhkd->bhqd", p, vs)


def _sliding_packed(qp, kp, vp, seg, scale, win):
    """Block-diagonal THD sliding attention. qp [Tq,Hq,D], kp/vp [Tk,Hkv,D], per seg cu_seqlens."""
    g = qp.shape[1] // kp.shape[1]
    cu_q, cu_k = seg["cu_q"].tolist(), seg["cu_k"].tolist()
    outs = []
    for i in range(len(cu_q) - 1):
        a, b, c, d = cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]
        ks, vs = kp[c:d].repeat_interleave(g, dim=1), vp[c:d].repeat_interleave(g, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qp[a:b], ks) * scale
        scores = scores.masked_fill(~_br_window(b - a, d - c, win, qp.device), float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vs))
    return torch.cat(outs, dim=0)


def _fa_dense_fwd_surrogate(q, k, v, scale, win):
    with torch.enable_grad():
        ins = tuple(t.detach().requires_grad_(True) for t in (q, k, v))
        out = _masked_attn(*ins, scale, _br_window(q.shape[2], k.shape[2], win, q.device))
    return out.detach(), {"ins": ins, "out": out}


def _fa_dense_bwd_surrogate(grad_out, st, scale, win):
    return torch.autograd.grad(st["out"], st["ins"], grad_out)


def _fa_varlen_fwd_surrogate(qp, kp, vp, seg, scale, win):
    with torch.enable_grad():
        ins = tuple(t.detach().requires_grad_(True) for t in (qp, kp, vp))
        out = _sliding_packed(*ins, seg, scale, win)
    return out.detach(), {"ins": ins, "out": out}


def _fa_varlen_bwd_surrogate(dout_p, st, seg, scale, win):
    return torch.autograd.grad(st["out"], st["ins"], dout_p)


@pytest.fixture
def _fa_surrogates(monkeypatch):
    """Swap the four CUDA-only FA2 low-level kernels for CPU differentiable surrogates."""
    monkeypatch.setattr(clr, "_fa_ll_dense_fwd", _fa_dense_fwd_surrogate)
    monkeypatch.setattr(clr, "_fa_ll_dense_bwd", _fa_dense_bwd_surrogate)
    monkeypatch.setattr(clr, "_fa_ll_varlen_fwd", _fa_varlen_fwd_surrogate)
    monkeypatch.setattr(clr, "_fa_ll_varlen_bwd", _fa_varlen_bwd_surrogate)


# ---------------------------------------------------------------------------
# _ring_num_prior_chunks -- the comms-reduction hop count (pure, no kernel)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_ring_num_prior_chunks_global_takes_full_rotation(cp_size):
    """Global attention collects every prior CP rank's KV chunk."""
    ctx = SimpleNamespace(module=_sliding_module(None), cp_size=cp_size, seq_local=4)
    assert cpa._ring_num_prior_chunks(ctx) == cp_size - 1


@pytest.mark.parametrize(
    "window,seq_local,cp_size,expected",
    [
        (6, 4, 8, 2),  # ceil(6/4)=2 prior chunks, far short of the full 7
        (4, 4, 8, 1),  # exactly one chunk back
        (5, 4, 8, 2),  # straddles into the second prior chunk
        (100, 4, 8, 7),  # window >= full seq -> clamped to cp_size-1
        (6, 4, 2, 1),  # clamped to cp_size-1 even when ceil says 2
    ],
)
def test_ring_num_prior_chunks_sliding_reduces(window, seq_local, cp_size, expected):
    """Sliding attention collects only prior chunks intersecting its window."""
    ctx = SimpleNamespace(module=_sliding_module(window), cp_size=cp_size, seq_local=seq_local)
    assert cpa._ring_num_prior_chunks(ctx) == expected


# ---------------------------------------------------------------------------
# _concat_neighborhood / _owner_grad_dict / _fold_pad (pure)
# ---------------------------------------------------------------------------
def test_concat_neighborhood_drops_future_owners_and_orders_ascending():
    """Neighborhood concatenation causally drops future chunks and sorts retained owners."""
    ctx = SimpleNamespace(cp_rank=2)
    chunks = [(owner, torch.full((1, 2, 3, 4), float(owner)), torch.zeros(1, 2, 3, 4), {}) for owner in (2, 1, 3, 0)]
    k_nb, v_nb, nb_ids, valid_owners = clr._concat_neighborhood(ctx, chunks)
    assert valid_owners == [0, 1, 2]  # owner 3 (> cp_rank) causally dropped, rest ascending
    assert nb_ids is None  # unpacked chunks -> dense path
    assert k_nb.shape[2] == 9 and [k_nb[0, 0, i, 0].item() for i in (0, 3, 6)] == [0.0, 1.0, 2.0]


def test_concat_neighborhood_packed_folds_padding_into_ids():
    """Packed neighborhood concatenation maps padded tokens to sequence ID zero."""
    ctx = SimpleNamespace(cp_rank=1)
    md0 = {"_packed_seq_ids": torch.tensor([[1, 1]]), "padding_mask": None}
    md1 = {"_packed_seq_ids": torch.tensor([[2, 2]]), "padding_mask": torch.tensor([[False, True]])}
    chunks = [
        (0, torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4), md0),
        (1, torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4), md1),
    ]
    _, _, nb_ids, valid_owners = clr._concat_neighborhood(ctx, chunks)
    assert valid_owners == [0, 1]
    assert nb_ids.tolist() == [[1, 1, 2, 0]]  # owner-1 pad position folded to id 0


def test_owner_grad_dict_fills_missing_owners_with_zeros():
    """Owner gradient maps supply zero chunks for owners with no contribution."""
    present = torch.ones(1, 2, 3, 4)
    got = clr._owner_grad_dict({3: present}, cp_rank=3, cp_size=4, n_prior=2, zeros_like=torch.empty(1, 2, 3, 4))
    assert set(got) == {3, 2, 1}  # owners (cp_rank - d) for d in 0..n_prior
    assert torch.equal(got[3], present)
    assert torch.count_nonzero(got[2]) == torch.count_nonzero(got[1]) == 0


def test_fold_pad_returns_same_object_without_mask_and_zeros_with_mask():
    """Padding folding preserves unmasked IDs and zeroes IDs at masked positions."""
    ids = torch.tensor([[1, 1, 2]])
    assert clr._fold_pad(ids, None) is ids
    assert clr._fold_pad(ids, torch.tensor([[False, True, False]])).tolist() == [[1, 0, 2]]


# ---------------------------------------------------------------------------
# Full local-kernel ring autograd Function -- fwd+bwd parity vs sliding SDPA.
# cp_size=1 (own chunk only, n_prior=0) exercises compute + owner routing.
# ---------------------------------------------------------------------------
def test_local_ring_dense_single_doc_matches_sliding_sdpa_fwd_bwd(_fa_surrogates):
    """Dense local-ring forward and backward match sliding-window SDPA on one CP rank."""
    win = 4
    ctx = _make_ctx(_sliding_module(win), q_heads=4, kv_heads=2, seq=6, head_dim=8)  # unpacked -> dense
    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4LocalKernelRingAttention.apply(q, k, v, ring_ctx)
    out.sum().backward()

    qr, kr, vr = (t.clone().requires_grad_(True) for t in (ctx.query, ctx.key, ctx.value))
    allowed = _br_window(6, 6, win, ctx.query.device)  # standard causal+window (nq==nk)
    ref = _masked_attn(qr, kr, vr, ctx.scale, allowed)
    ref.sum().backward()

    assert torch.allclose(out, ref, atol=1e-5)
    for got, exp in ((q, qr), (k, kr), (v, vr)):
        assert torch.allclose(got.grad, exp.grad, atol=1e-5)


def test_local_ring_varlen_packed_matches_blockdiag_sliding_sdpa_fwd_bwd(_fa_surrogates):
    """Packed local-ring forward and backward match document-isolated sliding SDPA."""
    win = 2  # < doc length so the window actually restricts within a document
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 0]])  # doc1(3) | doc2(2) | pad(1)
    ctx = _make_ctx(
        _sliding_module(win), q_heads=4, kv_heads=2, seq=6, head_dim=8, metadata={"_packed_seq_ids": doc_ids}
    )
    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4LocalKernelRingAttention.apply(q, k, v, ring_ctx)
    out.sum().backward()

    qr, kr, vr = (t.clone().requires_grad_(True) for t in (ctx.query, ctx.key, ctx.value))
    qpos, kpos = torch.arange(6).view(-1, 1), torch.arange(6).view(1, -1)
    qd, kd = doc_ids[0].view(-1, 1), doc_ids[0].view(1, -1)
    allowed = (kpos <= qpos) & (qpos - kpos <= win - 1) & (qd == kd) & (qd > 0)
    ref = _masked_attn(qr, kr, vr, ctx.scale, allowed) * (qd.view(1, 1, 6, 1) > 0)
    ref.sum().backward()

    real = doc_ids[0] > 0  # pad rows are 0 on both sides; compare real-token grads only
    assert torch.allclose(out, ref, atol=1e-5)
    for got, exp in ((q, qr), (k, kr), (v, vr)):
        assert torch.allclose(got.grad[:, :, real], exp.grad[:, :, real], atol=1e-5)


def test_local_ring_all_pad_shard_zeros_and_zero_grads(_fa_surrogates):
    """An all-padding shard safely returns zero outputs and zero gradients."""
    # A full-pad shard must return 0 / zero-grad, NOT raise: a one-rank raise desyncs the
    # rank-uniform dK/dV p2p and hangs multi-GPU training.
    ctx = _make_ctx(
        _sliding_module(4), seq=4, head_dim=8, metadata={"_packed_seq_ids": torch.zeros(1, 4, dtype=torch.long)}
    )
    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4LocalKernelRingAttention.apply(q, k, v, ring_ctx)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert torch.count_nonzero(q.grad) == torch.count_nonzero(k.grad) == torch.count_nonzero(v.grad) == 0


# ---------------------------------------------------------------------------
# Comms reduction: only the required chunks are collected, and dropping the
# rest is lossless (no in-window key lives in a dropped chunk).
# ---------------------------------------------------------------------------
def test_collect_ring_kv_chunks_sliding_collects_only_required(monkeypatch):
    """Sliding collection exchanges fewer KV chunks than global attention."""
    calls = {"n": 0}

    def fake_exchange(tensors, **kwargs):
        calls["n"] += 1
        for send_t, recv_t in tensors:
            recv_t.copy_(send_t)

    monkeypatch.setattr(cpa, "_ring_exchange", fake_exchange)
    md = {"_packed_seq_ids": torch.ones(1, 4, dtype=torch.long)}
    # cp_size=4, seq_local=4, window=6 -> n_prior=2: own + 2 prior, owner 0 dropped.
    sliding_ctx = _make_ctx(_sliding_module(6), seq=4, head_dim=8, cp_size=4, cp_rank=3, metadata=md)
    chunks = cpa._collect_ring_kv_chunks(sliding_ctx)
    assert [c[0] for c in chunks] == [3, 2, 1] and calls["n"] == 2

    calls["n"] = 0  # a global layer at the same geometry still takes the full rotation
    global_ctx = cpa.replace(sliding_ctx, module=_sliding_module(None))
    assert [c[0] for c in cpa._collect_ring_kv_chunks(global_ctx)] == [3, 2, 1, 0] and calls["n"] == 3


def test_reduction_is_lossless_vs_full_sequence_sliding_sdpa(_fa_surrogates):
    """Dropping out-of-window CP chunks preserves full-sequence sliding-attention output."""
    torch.manual_seed(0)
    sl, cp_size, win, D = 4, 4, 6, 8
    scale = 1.0 / math.sqrt(D)
    ctx = cpa.replace(
        _make_ctx(_sliding_module(win), q_heads=4, kv_heads=2, seq=sl, head_dim=D, cp_size=cp_size, cp_rank=3),
        scale=scale,
    )
    assert cpa._ring_num_prior_chunks(ctx) == 2  # owner 0 is dropped by the reduction

    q_local = torch.randn(1, 4, sl, D)
    k_full, v_full = torch.randn(1, 2, sl * cp_size, D), torch.randn(1, 2, sl * cp_size, D)
    chunks = [(o, k_full[:, :, o * sl : (o + 1) * sl], v_full[:, :, o * sl : (o + 1) * sl], {}) for o in (3, 2, 1)]
    out, _ = clr.sliding_ring_compute_fa_fwd(cpa.replace(ctx, query=q_local), chunks)

    # Reference attends the FULL sequence (incl. dropped owner 0); equality proves owner 0
    # held no in-window key for rank 3's queries (global positions 12..15).
    qpos = (torch.arange(sl) + 3 * sl).view(-1, 1)
    kpos = torch.arange(sl * cp_size).view(1, -1)
    ref = _masked_attn(q_local, k_full, v_full, scale, (kpos <= qpos) & (qpos - kpos <= win - 1))
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Per-step segment cache around the neighborhood build
# ---------------------------------------------------------------------------
def test_sliding_segment_cache_builds_once_then_hits(monkeypatch):
    """Repeated segment requests reuse one cached packed-ring segmentation."""
    clr._RING_SEGMENT_CACHE.clear()
    cpa._RING_SEGMENT_GEN[0] = None
    calls = []
    real = clr._build_packed_ring_segments

    def counting(q, k):
        calls.append(1)
        return real(q, k)

    monkeypatch.setattr(clr, "_build_packed_ring_segments", counting)
    ids = torch.tensor([[1, 1, 2, 2]])  # same object => same data_ptr generation
    ctx = SimpleNamespace(cp_rank=0, cp_size=1, metadata={"_packed_seq_ids": ids})
    for _ in range(3):
        clr._cached_sliding_segments(ctx, ids, ids, 0)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Dispatch + attach wiring
# ---------------------------------------------------------------------------
def test_run_dispatch_routes_sliding_fa_to_local_kernel(monkeypatch):
    """Dispatch selects the local FA kernel only for FA-backed sliding layers."""
    monkeypatch.setattr(cpa, "_ring_use_ffpa_varlen", lambda m, c: False)

    class _Local:
        @staticmethod
        def apply(q, k, v, ctx):
            return "LOCAL"

    class _Flex:
        @staticmethod
        def apply(q, k, v, ctx):
            return "FLEX"

    monkeypatch.setattr(cpa, "_Gemma4LocalKernelRingAttention", _Local)
    monkeypatch.setattr(cpa, "_Gemma4FlexRingAttention", _Flex)
    ctx = SimpleNamespace(query=torch.zeros(1, 2, 4, 8), key=torch.zeros(1, 2, 4, 8), value=torch.zeros(1, 2, 4, 8))

    assert (
        cpa._run_gemma4_cp_ring_attention(SimpleNamespace(_gemma4_cp_sliding_backend="fa", sliding_window=512), ctx)
        == "LOCAL"
    )
    assert (
        cpa._run_gemma4_cp_ring_attention(SimpleNamespace(_gemma4_cp_sliding_backend="flex", sliding_window=512), ctx)
        == "FLEX"
    )
    # fa only applies to sliding layers; a global (sliding_window None) layer stays on flex.
    assert (
        cpa._run_gemma4_cp_ring_attention(SimpleNamespace(_gemma4_cp_sliding_backend="fa", sliding_window=None), ctx)
        == "FLEX"
    )


def test_attach_sets_sliding_backend():
    """Attention attachment records the requested backend and defaults to flex."""
    fa = torch.nn.Module()
    cpa.attach_gemma4_cp_ring_attention(fa, sliding_backend="fa")
    assert fa._gemma4_cp_sliding_backend == "fa"

    default = torch.nn.Module()
    cpa.attach_gemma4_cp_ring_attention(default)
    assert default._gemma4_cp_sliding_backend == "flex"


# ---------------------------------------------------------------------------
# GPU smoke tests: the REAL FlashAttention-2 kernel (no surrogate), single GPU.
# Validates the kernel numerics + BHSD/THD layout transposes + window args that
# the CPU surrogates cannot. cp_size=1 => zero ring exchange and no process
# group, so the exact per-shard kernel call is exercised on one device.
# ---------------------------------------------------------------------------
_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention-2 kernel requires CUDA")


def _to_cuda_bf16(ctx):
    return cpa.replace(
        ctx,
        query=ctx.query.to("cuda", torch.bfloat16),
        key=ctx.key.to("cuda", torch.bfloat16),
        value=ctx.value.to("cuda", torch.bfloat16),
    )


@_GPU
@pytest.mark.run_only_on("GPU")
def test_local_ring_dense_matches_sliding_sdpa_on_gpu():
    """Real dense FA2 forward and backward match FP32 sliding SDPA on one GPU."""
    pytest.importorskip("flash_attn")
    win, seq, D = 8, 32, 64  # seq > win so the window genuinely restricts
    ctx = _to_cuda_bf16(_make_ctx(_sliding_module(win), q_heads=4, kv_heads=2, seq=seq, head_dim=D))
    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4LocalKernelRingAttention.apply(q, k, v, ring_ctx)
    out.sum().backward()

    qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (ctx.query, ctx.key, ctx.value))
    ref = _masked_attn(qr, kr, vr, ctx.scale, _br_window(seq, seq, win, qr.device))
    ref.sum().backward()

    assert torch.allclose(out.float(), ref, atol=2e-2)  # bf16 kernel vs fp32 reference
    for got, exp in ((q, qr), (k, kr), (v, vr)):
        assert torch.allclose(got.grad.float(), exp.grad, atol=3e-2)


@_GPU
@pytest.mark.run_only_on("GPU")
def test_local_ring_varlen_matches_blockdiag_sliding_sdpa_on_gpu():
    """Real packed FA2 forward and backward match document-isolated SDPA on one GPU."""
    pytest.importorskip("flash_attn")
    win, D = 4, 64
    doc_ids = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]], device="cuda")  # doc1(10) | doc2(6)
    seq = doc_ids.shape[1]
    ctx = _to_cuda_bf16(
        _make_ctx(
            _sliding_module(win), q_heads=4, kv_heads=2, seq=seq, head_dim=D, metadata={"_packed_seq_ids": doc_ids}
        )
    )
    q, k, v, ring_ctx = _requires_grad(ctx)
    out = cpa._Gemma4LocalKernelRingAttention.apply(q, k, v, ring_ctx)
    out.sum().backward()

    qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (ctx.query, ctx.key, ctx.value))
    qpos, kpos = torch.arange(seq, device="cuda").view(-1, 1), torch.arange(seq, device="cuda").view(1, -1)
    qd, kd = doc_ids[0].view(-1, 1), doc_ids[0].view(1, -1)
    ref = _masked_attn(qr, kr, vr, ctx.scale, (kpos <= qpos) & (qpos - kpos <= win - 1) & (qd == kd))
    ref.sum().backward()

    assert torch.allclose(out.float(), ref, atol=2e-2)
    for got, exp in ((q, qr), (k, kr), (v, vr)):
        assert torch.allclose(got.grad.float(), exp.grad, atol=3e-2)
