#!/usr/bin/env -S /home/jovyan/Automodel/.pixi/envs/nemo/bin/python
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

"""4-rank end-to-end (real NCCL ring) throughput: FFPA backend vs Flex backend.

Confirms the per-chunk aux numbers aggregate over a whole ring. For each shard layout
(padded single-doc / aligned multi-doc / small-straddle) every rank runs the full
``_Gemma4FFPAVarlenRingAttention`` and ``_Gemma4FlexRingAttention`` (forward + backward,
all ``cp_size`` chunks rotated by p2p) and we time forward, backward, and total. Step time
is the bottleneck rank (max over ranks, since ranks sync at each ring exchange). A parity
gate (FFPA vs Flex out + dQ/dK/dV) precedes timing so the comparison is apples-to-apples.

Run::

    torchrun --standalone --nproc-per-node=4 \
        tests/functional_tests/models/gemma4/run_ffpa_vs_flex_throughput_cp.py
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention

from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa


def _ctx(module, q, k, v, *, rank, cp_size, S_local, scale, Hq, Hkv, ids):
    return cpa.CPRingAttentionContext(
        module=module,
        query=q,
        key=k,
        value=v,
        cp_mesh=None,
        cp_group=dist.group.WORLD,
        cp_size=cp_size,
        cp_rank=rank,
        seq_local=S_local,
        seq_full=S_local * cp_size,
        seq_global_start=rank * S_local,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
        enable_gqa=(Hq != Hkv),
        kwargs={},
        metadata={"_packed_seq_ids": ids},
        metadata_seq_dims={},
    )


def _scenario_ids(kind, S_full, S_local, cp_size, dev):
    ids = torch.zeros(1, S_full, dtype=torch.long, device=dev)
    if kind == "padded":  # one doc spanning all but a half-shard tail pad (only last rank padded)
        ids[:, : S_full - S_local // 2] = 1
    elif kind == "multidoc":  # docs aligned to half-shards: each rank = 2 full docs, no straddle
        half = S_local // 2
        for d in range(S_full // half):
            ids[:, d * half : (d + 1) * half] = d + 1
    elif kind == "straddle":  # filler docs split by an 8-token straddle doc on every CP boundary
        cuts = {0, S_full}
        for b in range(1, cp_size):
            cuts.add(b * S_local - 4)
            cuts.add(b * S_local + 4)
        edges = sorted(cuts)
        for doc_id, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            ids[:, a:b] = doc_id
    return ids


def _time(once, *, warmup, iters):
    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        once()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters  # ms/call


def main() -> int:
    if not torch.cuda.is_available() or not cpa._ffpa_varlen_ring_available():
        print("[skip] CUDA + FFPA CuTeDSL required", file=sys.stderr)
        return 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dt = torch.bfloat16
    B, Hq, Hkv, D = 1, 8, 4, 512
    S_local = 2048
    S_full = S_local * cp_size
    scale = 1.0 / math.sqrt(256)
    s0 = rank * S_local

    torch.manual_seed(0)
    q_full = torch.randn(B, Hq, S_full, D, device=dev, dtype=dt)
    k_full = torch.randn(B, Hkv, S_full, D, device=dev, dtype=dt)
    v_full = torch.randn(B, Hkv, S_full, D, device=dev, dtype=dt)
    torch.manual_seed(1)
    grad = torch.randn(B, Hq, S_local, D, device=dev, dtype=dt)
    module = SimpleNamespace(
        sliding_window=None,
        config=SimpleNamespace(use_bidirectional_attention=None),
        _gemma4_cp_compiled_flex_attn=torch.compile(flex_attention, dynamic=True),
    )

    def make(ids):
        q = q_full[:, :, s0 : s0 + S_local].clone().requires_grad_(True)
        k = k_full[:, :, s0 : s0 + S_local].clone().requires_grad_(True)
        v = v_full[:, :, s0 : s0 + S_local].clone().requires_grad_(True)
        ctx = _ctx(module, q, k, v, rank=rank, cp_size=cp_size, S_local=S_local, scale=scale, Hq=Hq, Hkv=Hkv, ids=ids)
        return q, k, v, ctx

    def run(fn, ids, with_bwd):
        q, k, v, ctx = make(ids)
        out = fn.apply(q, k, v, ctx)
        if with_bwd:
            out.backward(grad)

    all_ok = True
    if rank == 0:
        print(
            f"ring throughput  cp_size={cp_size}  B={B} Hq={Hq} Hkv={Hkv} D={D} S_local={S_local} S_full={S_full} {dt}"
        )
        print(
            f"{'scenario':<10} {'fwd ffpa/flex (ms)':>22} {'bwd ffpa/flex (ms)':>22} "
            f"{'total ffpa/flex':>20} {'speedup':>9}"
        )

    for kind in ("padded", "multidoc", "straddle"):
        ids = _scenario_ids(kind, S_full, S_local, cp_size, dev)[:, s0 : s0 + S_local].contiguous()

        # ---- parity gate (FFPA vs Flex) ----
        q, k, v, ctx = make(ids)
        of = cpa._Gemma4FFPAVarlenRingAttention.apply(q, k, v, ctx)
        of.backward(grad)
        gq_f, gk_f, gv_f = q.grad, k.grad, v.grad
        q2, k2, v2, ctx2 = make(ids)
        ox = cpa._Gemma4FlexRingAttention.apply(q2, k2, v2, ctx2)
        ox.backward(grad)
        close = all(
            torch.allclose(a.float(), b.float(), atol=3e-2, rtol=2e-2)
            for a, b in ((of, ox), (gq_f, q2.grad), (gk_f, k2.grad), (gv_f, v2.grad))
        )
        all_ok = all_ok and close

        dist.barrier()
        f_ffpa = _time(lambda: run(cpa._Gemma4FFPAVarlenRingAttention, ids, False), warmup=3, iters=8)
        t_ffpa = _time(lambda: run(cpa._Gemma4FFPAVarlenRingAttention, ids, True), warmup=3, iters=8)
        f_flex = _time(lambda: run(cpa._Gemma4FlexRingAttention, ids, False), warmup=3, iters=8)
        t_flex = _time(lambda: run(cpa._Gemma4FlexRingAttention, ids, True), warmup=3, iters=8)
        b_ffpa, b_flex = t_ffpa - f_ffpa, t_flex - f_flex

        # bottleneck rank = max over ranks (all ranks sync at each ring exchange)
        stats = torch.tensor([f_ffpa, f_flex, b_ffpa, b_flex, t_ffpa, t_flex], device=dev)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        f_a, f_x, b_a, b_x, t_a, t_x = stats.tolist()
        if rank == 0:
            print(
                f"{kind:<10} {f_a:9.2f} / {f_x:9.2f}    {b_a:9.2f} / {b_x:9.2f}    "
                f"{t_a:8.2f} / {t_x:8.2f}   {t_x / t_a:6.2f}x  parity={'ok' if close else 'FAIL'}"
            )

    flag = torch.tensor([1 if all_ok else 0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    dist.barrier()
    if rank == 0:
        print("[ALL RANKS PARITY OK]" if int(flag.item()) == 1 else "[PARITY FAILED]")
    dist.destroy_process_group()
    return 0 if int(flag.item()) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
