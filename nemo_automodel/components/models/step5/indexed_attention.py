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

"""Region-indexed GQA with online softmax and bounded backward storage.

Each program handles one query and one KV group, reusing selected K/V across
that group's query heads. Only selected regions are read. Backward recomputes
probabilities and reduces K/V gradients by region in FP32; no attention matrix or
sequence-squared membership tensor is allocated.
"""

import torch
from torch.autograd.function import once_differentiable

from nemo_automodel.components.models.step5.fa4 import fa4_attention
from nemo_automodel.components.models.step5.sparse import IndexedRegionLayout, RegionSelection
from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TL, tl = safe_import("triton.language")

if _HAVE_TRITON and _HAVE_TL:

    @triton.jit
    def _attention(
        Q,
        K,
        V,
        IDS,
        COUNTS,
        STARTS,
        ENDS,
        POS,
        KPOS,
        PAD,
        DOC,
        KDOC,
        OUT,
        LSE,
        DO,
        DQ,
        QUERIES: tl.constexpr,
        KEYS: tl.constexpr,
        GROUPS: tl.constexpr,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        REGIONS: tl.constexpr,
        SLOTS: tl.constexpr,
        REGION_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
        BACKWARD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        K_BATCH: tl.constexpr,
        K_TOKEN: tl.constexpr,
        K_HEAD: tl.constexpr,
        V_BATCH: tl.constexpr,
        V_TOKEN: tl.constexpr,
        V_HEAD: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Triton implementation of the tensor contract in selected_attention."""
        qi, bg = tl.program_id(0), tl.program_id(1)
        batch, group = bg // GROUPS, bg % GROUPS
        h = tl.arange(0, BLOCK_H)
        d0, d1 = tl.arange(0, 128), 128 + tl.arange(0, 64)
        q_base = ((batch * QUERIES + qi) * GROUPS * HEADS + group * HEADS + h[:, None]) * DIM
        qo0, qo1 = q_base + d0[None, :], q_base + d1[None, :]
        q0 = tl.load(Q + qo0, h[:, None] < HEADS, other=0)
        q1 = tl.load(Q + qo1, h[:, None] < HEADS, other=0)
        query_position = tl.load(POS + batch * QUERIES + qi)
        query_document = tl.load(DOC + batch * QUERIES + qi)
        row = (batch * GROUPS + group) * QUERIES + qi
        count = tl.load(COUNTS + row)
        lse_offset = (batch * QUERIES + qi) * GROUPS * HEADS + group * HEADS + h
        if BACKWARD:
            out0 = tl.load(OUT + qo0, h[:, None] < HEADS, other=0).to(tl.float32)
            out1 = tl.load(OUT + qo1, h[:, None] < HEADS, other=0).to(tl.float32)
            do0 = tl.load(DO + qo0, h[:, None] < HEADS, other=0)
            do1 = tl.load(DO + qo1, h[:, None] < HEADS, other=0)
            delta = tl.sum(out0 * do0.to(tl.float32), 1) + tl.sum(out1 * do1.to(tl.float32), 1)
            lse = tl.load(LSE + lse_offset, h < HEADS, other=float("inf"))
            dq0 = tl.zeros((BLOCK_H, 128), tl.float32)
            dq1 = tl.zeros((BLOCK_H, 64), tl.float32)
        else:
            maximum = tl.full((BLOCK_H,), float("-inf"), tl.float32)
            normalizer = tl.zeros((BLOCK_H,), tl.float32)
            acc0 = tl.zeros((BLOCK_H, 128), tl.float32)
            acc1 = tl.zeros((BLOCK_H, 64), tl.float32)

        for offset in range(0, count * REGION_SIZE, BLOCK_K):
            compact = offset + tl.arange(0, BLOCK_K)
            slot, within = compact // REGION_SIZE, compact % REGION_SIZE
            region = tl.load(IDS + row * SLOTS + slot, slot < count, other=REGIONS)
            previous = tl.load(IDS + row * SLOTS + slot - 1, (slot > 0) & (slot < count), other=-1)
            valid_region = (slot < count) & (region < REGIONS) & (region != previous)
            first = tl.load(STARTS + batch * REGIONS + region, valid_region, other=KEYS)
            end = tl.load(ENDS + batch * REGIONS + region, valid_region, other=0)
            ki = first + within
            valid = valid_region & (ki < end) & (ki < KEYS)
            kp = tl.load(KPOS + batch * KEYS + ki, valid, other=0)
            kd = tl.load(KDOC + batch * KEYS + ki, valid, other=-1)
            padding = tl.load(PAD + batch * KEYS + ki, valid, other=1)
            valid = valid & (kp <= query_position) & (kd == query_document) & ~padding
            kb = batch * K_BATCH + ki[:, None] * K_TOKEN + group * K_HEAD
            vb = batch * V_BATCH + ki[:, None] * V_TOKEN + group * V_HEAD
            k0 = tl.load(K + kb + d0[None, :], valid[:, None], other=0)
            k1 = tl.load(K + kb + d1[None, :], valid[:, None], other=0)
            v0 = tl.load(V + vb + d0[None, :], valid[:, None], other=0)
            v1 = tl.load(V + vb + d1[None, :], valid[:, None], other=0)
            scores = tl.dot(q0, tl.trans(k0))
            scores = tl.dot(q1, tl.trans(k1), acc=scores) * SCALE
            scores = tl.where(valid[None, :] & (h[:, None] < HEADS), scores, float("-inf"))
            if BACKWARD:
                probability = tl.exp(scores - lse[:, None])
                dp = tl.dot(do0, tl.trans(v0))
                dp = tl.dot(do1, tl.trans(v1), acc=dp)
                ds = (probability * (dp - delta[:, None]) * SCALE).to(q0.dtype)
                dq0 += tl.dot(ds, k0)
                dq1 += tl.dot(ds, k1)
            else:
                next_max = tl.maximum(maximum, tl.max(scores, 1))
                safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
                correction = tl.exp(maximum - safe_max)
                probability = tl.exp(scores - safe_max[:, None])
                normalizer = normalizer * correction + tl.sum(probability, 1)
                acc0 = acc0 * correction[:, None] + tl.dot(probability.to(v0.dtype), v0)
                acc1 = acc1 * correction[:, None] + tl.dot(probability.to(v1.dtype), v1)
                maximum = next_max
        if BACKWARD:
            tl.store(DQ + qo0, dq0, h[:, None] < HEADS)
            tl.store(DQ + qo1, dq1, h[:, None] < HEADS)
        else:
            denominator = tl.where(normalizer > 0, normalizer, 1.0)[:, None]
            tl.store(OUT + qo0, acc0 / denominator, h[:, None] < HEADS)
            tl.store(OUT + qo1, acc1 / denominator, h[:, None] < HEADS)
            lse = tl.where(normalizer > 0, maximum + tl.log(normalizer), float("inf"))
            tl.store(LSE + lse_offset, lse, h < HEADS)

    @triton.jit
    def _edge_counts(
        IDS,
        COUNTS,
        TOTAL: tl.constexpr,
        SLOTS: tl.constexpr,
        QUERIES: tl.constexpr,
        REGIONS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        slot = i % SLOTS
        region = tl.load(IDS + i, i < TOTAL, other=REGIONS)
        previous = tl.load(IDS + i - 1, (i < TOTAL) & (slot > 0), other=-1)
        bg = i // (SLOTS * QUERIES)
        valid = (i < TOTAL) & (region < REGIONS) & (region != previous)
        tl.atomic_add(COUNTS + bg * REGIONS + region, 1, valid, sem="relaxed")

    @triton.jit
    def _edge_scatter(
        IDS,
        OFFSETS,
        CURSOR,
        EDGES,
        TOTAL: tl.constexpr,
        SLOTS: tl.constexpr,
        QUERIES: tl.constexpr,
        REGIONS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        slot = i % SLOTS
        region = tl.load(IDS + i, i < TOTAL, other=REGIONS)
        previous = tl.load(IDS + i - 1, (i < TOTAL) & (slot > 0), other=-1)
        bg = i // (SLOTS * QUERIES)
        valid = (i < TOTAL) & (region < REGIONS) & (region != previous)
        row = bg * REGIONS + region
        begin = tl.load(OFFSETS + row, valid, other=0)
        position = tl.atomic_add(CURSOR + row, 1, valid, sem="relaxed")
        tl.store(EDGES + begin + position, (i // SLOTS) % QUERIES, valid)

    @triton.jit
    def _delta(OUT, DO, DELTA, ROWS: tl.constexpr, BH: tl.constexpr):
        row = tl.program_id(0) * BH + tl.arange(0, BH)
        d0 = tl.arange(0, 128)
        d1 = 128 + tl.arange(0, 64)
        a = tl.load(OUT + row[:, None] * 192 + d0[None, :], row[:, None] < ROWS, other=0).to(tl.float32)
        b = tl.load(DO + row[:, None] * 192 + d0[None, :], row[:, None] < ROWS, other=0).to(tl.float32)
        c = tl.load(OUT + row[:, None] * 192 + d1[None, :], row[:, None] < ROWS, other=0).to(tl.float32)
        d = tl.load(DO + row[:, None] * 192 + d1[None, :], row[:, None] < ROWS, other=0).to(tl.float32)
        tl.store(DELTA + row, tl.sum(a * b, 1) + tl.sum(c * d, 1), row < ROWS)

    @triton.jit
    def _region_backward(
        Q,
        K,
        V,
        DO,
        LSE,
        DELTA,
        OFFSETS,
        EDGES,
        STARTS,
        ENDS,
        POS,
        KPOS,
        PAD,
        DOC,
        KDOC,
        DK,
        DV,
        QUERIES: tl.constexpr,
        KEYS: tl.constexpr,
        GROUPS: tl.constexpr,
        HEADS: tl.constexpr,
        REGIONS: tl.constexpr,
        REGION_SIZE: tl.constexpr,
        SCALE: tl.constexpr,
        K_BATCH: tl.constexpr,
        K_TOKEN: tl.constexpr,
        K_HEAD: tl.constexpr,
        V_BATCH: tl.constexpr,
        V_TOKEN: tl.constexpr,
        V_HEAD: tl.constexpr,
        BN: tl.constexpr,
    ):
        region = tl.program_id(0)
        bg = tl.program_id(1)
        batch = bg // GROUPS
        group = bg % GROUPS
        first = tl.load(STARTS + batch * REGIONS + region)
        end = tl.load(ENDS + batch * REGIONS + region)
        ki = first + tl.arange(0, 16)
        valid_key = (tl.arange(0, 16) < REGION_SIZE) & (ki < end) & (ki < KEYS)
        kp = tl.load(KPOS + batch * KEYS + ki, valid_key, other=0)
        kd = tl.load(KDOC + batch * KEYS + ki, valid_key, other=-1)
        padding = tl.load(PAD + batch * KEYS + ki, valid_key, other=1)
        valid_key = valid_key & ~padding
        d0 = tl.arange(0, 128)
        d1 = 128 + tl.arange(0, 64)
        kb = batch * K_BATCH + ki[:, None] * K_TOKEN + group * K_HEAD
        vb = batch * V_BATCH + ki[:, None] * V_TOKEN + group * V_HEAD
        k0 = tl.load(K + kb + d0[None, :], valid_key[:, None], other=0)
        k1 = tl.load(K + kb + d1[None, :], valid_key[:, None], other=0)
        v0 = tl.load(V + vb + d0[None, :], valid_key[:, None], other=0)
        v1 = tl.load(V + vb + d1[None, :], valid_key[:, None], other=0)
        dk0 = tl.zeros((16, 128), tl.float32)
        dk1 = tl.zeros((16, 64), tl.float32)
        dv0 = tl.zeros((16, 128), tl.float32)
        dv1 = tl.zeros((16, 64), tl.float32)
        begin = tl.load(OFFSETS + bg * REGIONS + region)
        stop = tl.load(OFFSETS + bg * REGIONS + region + 1)
        n = tl.arange(0, BN)
        for offset in range(0, (stop - begin) * HEADS, BN):
            headentry = offset + n
            edge = begin + headentry // HEADS
            head = headentry % HEADS
            valid = edge < stop
            query = tl.load(EDGES + edge, valid, other=0)
            qp = tl.load(POS + batch * QUERIES + query, valid, other=0)
            qd = tl.load(DOC + batch * QUERIES + query, valid, other=-2)
            qrow = (batch * QUERIES + query) * GROUPS * HEADS + group * HEADS + head
            qb = qrow[:, None] * 192
            q0 = tl.load(Q + qb + d0[None, :], valid[:, None], other=0)
            q1 = tl.load(Q + qb + d1[None, :], valid[:, None], other=0)
            do0 = tl.load(DO + qb + d0[None, :], valid[:, None], other=0)
            do1 = tl.load(DO + qb + d1[None, :], valid[:, None], other=0)
            lse = tl.load(LSE + qrow, valid, other=float("inf"))
            delta = tl.load(DELTA + qrow, valid, other=0)
            scores = tl.dot(k0, tl.trans(q0))
            scores = tl.dot(k1, tl.trans(q1), acc=scores) * SCALE
            allowed = valid_key[:, None] & valid[None, :] & (kp[:, None] <= qp[None, :]) & (kd[:, None] == qd[None, :])
            probability = tl.where(allowed, tl.exp(scores - lse[None, :]), 0.0)
            dp = tl.dot(v0, tl.trans(do0))
            dp = tl.dot(v1, tl.trans(do1), acc=dp)
            ds = (probability * (dp - delta[None, :]) * SCALE).to(q0.dtype)
            dk0 += tl.dot(ds, q0)
            dk1 += tl.dot(ds, q1)
            prob = probability.to(do0.dtype)
            dv0 += tl.dot(prob, do0)
            dv1 += tl.dot(prob, do1)
        gb = ((batch * KEYS + ki[:, None]) * GROUPS + group) * 192
        # Regions partition physical keys; invalid/padded keys receive explicit zero.
        store = (tl.arange(0, 16) < REGION_SIZE) & (ki < end) & (ki < KEYS)
        tl.store(DK + gb + d0[None, :], dk0, store[:, None])
        tl.store(DK + gb + d1[None, :], dk1, store[:, None])
        tl.store(DV + gb + d0[None, :], dv0, store[:, None])
        tl.store(DV + gb + d1[None, :], dv1, store[:, None])


def _inverse_edges(ids: torch.Tensor, regions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert discrete routes using storage proportional to selected edges.

    Args:
        ids: Sorted int32 CUDA IDs [batch, kv_groups, local_queries, slots].
            Adjacent duplicate and sentinel IDs do not create duplicate edges.
        regions: Number of valid region IDs; also the sentinel ID.

    Returns:
        Int32 row offsets [batch * kv_groups * regions + 1] and query indices
        [ids.numel()]. Only the offset-delimited prefix of indices is initialized.
        Atomic insertion changes edge order but never the selected key sets.
    """
    batch, groups, queries, slots = ids.shape
    if max(ids.numel(), batch * groups * regions) > torch.iinfo(torch.int32).max:
        raise ValueError("Step5 inverse-edge metadata exceeds int32 indexing capacity")
    counts = torch.zeros(batch * groups * regions, device=ids.device, dtype=torch.int32)
    _edge_counts[(triton.cdiv(ids.numel(), 256),)](ids, counts, ids.numel(), slots, queries, regions, 256)
    offsets = torch.empty(counts.numel() + 1, device=ids.device, dtype=torch.int32)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, dtype=torch.int32, out=offsets[1:])
    cursor = torch.zeros_like(counts)
    edges = torch.empty(ids.numel(), device=ids.device, dtype=torch.int32)
    _edge_scatter[(triton.cdiv(ids.numel(), 256),)](
        ids, offsets, cursor, edges, ids.numel(), slots, queries, regions, 256
    )
    return offsets, edges


class _SelectedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ids: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        positions: torch.Tensor,
        key_positions: torch.Tensor,
        padding: torch.Tensor,
        documents: torch.Tensor,
        key_documents: torch.Tensor,
        region_size: int,
    ) -> torch.Tensor:
        """Evaluate selected attention and retain linear-size backward state.

        Args:
            ctx: Autograd-owned state.
            query: BF16 CUDA tensor [batch, local_queries, query_heads, 192].
            key: BF16 CUDA tensor [batch, global_keys, kv_heads, 192].
            value: Tensor shaped like key; K/V may be strided unbind views.
            ids: Sorted int32 IDs [batch, kv_heads, local_queries, slots].
            starts: Int32 region starts [batch, regions].
            ends: Int32 exclusive region ends [batch, regions].
            positions: Document positions [batch, local_queries].
            key_positions: Document positions [batch, global_keys].
            padding: Boolean flags [batch, global_keys], true for padding.
            documents: Document IDs [batch, local_queries].
            key_documents: Document IDs [batch, global_keys].
            region_size: Number of valid tokens per region.

        Returns:
            Independent BF16 output [batch, local_queries, query_heads, 192].
        """
        batch, queries, heads, dim = query.shape
        groups, regions = key.shape[2], starts.shape[1]
        query = query.contiguous()
        # Preserve the joint CP K/V allocation; do not copy global keys/values.

        metadata = tuple(
            x.contiguous() for x in (ids, starts, ends, positions, key_positions, padding, documents, key_documents)
        )
        ids, starts, ends, positions, key_positions, padding, documents, key_documents = metadata
        counts = (ids < regions).sum(-1, dtype=torch.int32)
        output = torch.empty_like(query)
        lse = torch.empty((batch, queries, heads), device=query.device, dtype=torch.float32)
        launch = dict(
            QUERIES=queries,
            KEYS=key.shape[1],
            GROUPS=groups,
            HEADS=heads // groups,
            DIM=dim,
            REGIONS=regions,
            SLOTS=ids.shape[-1],
            REGION_SIZE=region_size,
            SCALE=dim**-0.5,
            BLOCK_H=max(16, triton.next_power_of_2(heads // groups)),
            K_BATCH=key.stride(0),
            K_TOKEN=key.stride(1),
            K_HEAD=key.stride(2),
            V_BATCH=value.stride(0),
            V_TOKEN=value.stride(1),
            V_HEAD=value.stride(2),
            BLOCK_K=64,
            num_warps=4,
        )
        _attention[(queries, batch * groups)](
            query,
            key,
            value,
            ids,
            counts,
            starts,
            ends,
            positions,
            key_positions,
            padding,
            documents,
            key_documents,
            output,
            lse,
            output,
            query,
            BACKWARD=False,
            **launch,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            ids,
            counts,
            starts,
            ends,
            positions,
            key_positions,
            padding,
            documents,
            key_documents,
            output,
            lse,
        )
        ctx.launch = launch
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Return Q/K/V gradients in their input layouts; metadata is discrete.

        Args:
            ctx: Saved state from forward.
            grad_output: Tensor of shape [batch, local_queries, query_heads, head_dim].

        Returns:
            Q gradient [batch, local_queries, query_heads, head_dim], K/V
            gradients [batch, global_keys, kv_heads, head_dim], then None for
            each discrete metadata input. Inverse-edge construction can change
            reduction order, so deterministic and higher-order backward are unsupported.
        """
        (
            query,
            key,
            value,
            ids,
            counts,
            starts,
            ends,
            positions,
            key_positions,
            padding,
            documents,
            key_documents,
            output,
            lse,
        ) = ctx.saved_tensors
        dq = torch.empty_like(query)
        dk = torch.zeros(key.shape, device=key.device, dtype=key.dtype)
        dv = torch.zeros(value.shape, device=value.device, dtype=value.dtype)
        grad_output = grad_output.contiguous()
        offsets, edges = _inverse_edges(ids, starts.shape[1])
        delta = torch.empty_like(lse)
        _delta[(triton.cdiv(lse.numel(), 16),)](output, grad_output, delta, lse.numel(), 16)
        _attention[(query.shape[1], query.shape[0] * key.shape[2])](
            query,
            key,
            value,
            ids,
            counts,
            starts,
            ends,
            positions,
            key_positions,
            padding,
            documents,
            key_documents,
            output,
            lse,
            grad_output,
            dq,
            BACKWARD=True,
            **ctx.launch,
        )
        _region_backward[(starts.shape[1], query.shape[0] * key.shape[2])](
            query,
            key,
            value,
            grad_output,
            lse,
            delta,
            offsets,
            edges,
            starts,
            ends,
            positions,
            key_positions,
            padding,
            documents,
            key_documents,
            dk,
            dv,
            QUERIES=query.shape[1],
            KEYS=key.shape[1],
            GROUPS=key.shape[2],
            HEADS=query.shape[2] // key.shape[2],
            REGIONS=starts.shape[1],
            REGION_SIZE=ctx.launch["REGION_SIZE"],
            SCALE=query.shape[-1] ** -0.5,
            K_BATCH=key.stride(0),
            K_TOKEN=key.stride(1),
            K_HEAD=key.stride(2),
            V_BATCH=value.stride(0),
            V_TOKEN=value.stride(1),
            V_HEAD=value.stride(2),
            BN=64,
            num_warps=4,
        )
        return dq, dk, dv, *([None] * 9)


def selected_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_padding: torch.Tensor,
    *,
    selection: RegionSelection,
    region_size: int,
    query_start: int = 0,
) -> torch.Tensor:
    """Attend to selected regions, with rowwise CSR/FA4 dispatch for mixed B2.

    Eligible sparse rows use the Triton kernel; their noneligible partner uses
    the established FA4 prefix, all-padding, or sparse implementation. Q/K/V
    row views share input storage; autograd still assembles full-batch gradients.

    Args:
        query: BF16 CUDA tensor [batch, local_queries, query_heads, head_dim].
        key: Same-dtype CUDA tensor [batch, global_keys, kv_heads, head_dim].
        value: Same shape/dtype/device as key. Query heads must divide into KV groups.
        positions: Document-local int tensor [batch, local_queries].
        key_positions: Document-local int tensor [batch, global_keys].
        key_padding: Boolean tensor [batch, global_keys], true for padding.
        selection: Sorted region IDs and document metadata described by RegionSelection.
            Adjacent duplicate IDs are deduplicated; sentinel IDs are ignored.
        query_start: Physical global key offset of the first local query.
        region_size: Eight valid tokens per contiguous region. A trailing padded
            region can contain more physical keys, whose gradients stay zero.

    Returns:
        Independent tensor [batch, local_queries, query_heads, head_dim].
        Inputs are not mutated. First-order gradients cover Q, K, and V.
    """
    if not (_HAVE_TRITON and _HAVE_TL):
        raise ImportError("Step5 selected attention requires Triton")
    if not query.is_cuda or query.dtype != torch.bfloat16:
        raise ValueError("Step5 selected attention requires CUDA BF16 Q/K/V")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise ValueError("Step5 indexed attention currently supports SM90")
    if region_size != 8:
        raise ValueError("Step5 indexed attention requires eight-token regions")
    if query.shape[-1] != 192:
        raise ValueError("Step5 indexed attention requires the native 192-dimensional head")
    if any(x.dtype != query.dtype or x.device != query.device or x.stride(-1) != 1 for x in (key, value)):
        raise ValueError("Step5 indexed attention requires matching dtypes and contiguous head dimensions")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Step5 selected attention requires matching Q/K/V batch and head dimensions")
    if query.shape[2] % key.shape[2] or selection.region_ids.shape[:3] != (
        query.shape[0],
        key.shape[2],
        query.shape[1],
    ):
        raise ValueError("Step5 selected attention requires one route set per query and KV group")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Step5 selected attention uses nondeterministic inverse-edge ordering")
    layout = selection.indexed_layout
    if layout is None:
        raise ValueError("Step5 indexed attention requires shared physical region bounds")
    if layout.starts.shape != layout.ends.shape or layout.starts.shape != (key.shape[0], selection.region_count):
        raise ValueError("Step5 indexed region bounds do not match batch and region count")
    if any(x.dtype != torch.int32 or x.device != query.device for x in (layout.starts, layout.ends)):
        raise ValueError("Step5 indexed region bounds must be CUDA int32 tensors")
    if layout.row_eligible and len(layout.row_eligible) != query.shape[0]:
        raise ValueError("Indexed row eligibility must match the batch")
    if query.shape[0] == 2 and layout.row_eligible and not all(layout.row_eligible):
        if len(selection.causal_row_layouts) != 2:
            raise ValueError("Mixed indexed B2 dispatch requires two row proofs")
        # Unbind differentiable buffers once: separate slices would each create
        # a zero-filled full-batch K/V gradient before accumulating them.
        query_rows, key_rows, value_rows = query.unbind(0), key.unbind(0), value.unbind(0)
        outputs = []
        for row, eligible in enumerate(layout.row_eligible):
            proof = selection.causal_row_layouts[row]
            row_selection = RegionSelection(
                selection.region_ids[row : row + 1],
                selection.token_regions[row : row + 1],
                selection.query_documents[row : row + 1],
                selection.key_documents[row : row + 1],
                selection.region_count,
                all_regions_selected=proof is not None,
                causal_layout=proof if proof is not None and proof.valid_queries else None,
                all_padding_key=proof.padding_key if proof is not None and not proof.valid_queries else None,
                indexed_layout=IndexedRegionLayout(layout.starts[row : row + 1], layout.ends[row : row + 1]),
            )
            q, k, v = query_rows[row][None], key_rows[row][None], value_rows[row][None]
            if eligible:
                output = selected_attention(
                    q,
                    k,
                    v,
                    positions[row : row + 1],
                    key_positions[row : row + 1],
                    key_padding[row : row + 1],
                    selection=row_selection,
                    region_size=region_size,
                    query_start=query_start,
                )
            else:
                output = fa4_attention(
                    q,
                    k,
                    v,
                    positions[row : row + 1],
                    key_positions[row : row + 1],
                    key_padding[row : row + 1],
                    selection.key_documents[row : row + 1],
                    query_start=query_start,
                    window=None,
                    selection=row_selection,
                )
            outputs.append(output.squeeze(0))
        return torch.stack(outputs, dim=0)
    return _SelectedAttention.apply(
        query,
        key,
        value,
        selection.region_ids,
        layout.starts,
        layout.ends,
        positions,
        key_positions,
        key_padding,
        selection.query_documents,
        selection.key_documents,
        region_size,
    )
