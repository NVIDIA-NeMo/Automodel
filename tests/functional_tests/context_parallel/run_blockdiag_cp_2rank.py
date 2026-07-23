#!/usr/bin/env python
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

"""2-rank (real NCCL) forward+backward parity for block-diagonal varlen CP.

Runs the full production path -- ``configure_cp_varlen`` ->
``make_cp_blockdiag_batch_and_ctx`` (real DeviceMesh / process group) ->
``cp_blockdiag_sdpa`` -- for every KV-exchange collective (``allgather`` full
K/V all-gather with reduce-scatter backward, ``halo`` left-neighbor p2p, and
``a2a`` needed-only all-to-all-v, both selectable at cp=2 via the
``kv_exchange`` knob) and both varlen kernels (flash / TE) plus the dense
fallback. Each case is checked against a single-process dense block-causal
reference computed from identical full tensors on every rank: local outputs,
the input-embedding gradient slice, and the cross-rank-summed per-parameter
gradients of a tiny q/k/v/o attention module. Scenarios cover a document
straddling the rank boundary, an entire rank chunk of padding (the collective
no-hang case), and a single document spanning both ranks.

Run::

    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_blockdiag_cp_2rank.py
"""

from __future__ import annotations

import sys
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.blockdiag_cp import (
    configure_cp_varlen,
    cp_blockdiag_sdpa,
    make_cp_blockdiag_batch_and_ctx,
)
from nemo_automodel.components.distributed.blockdiag_cp import kernels as bd_kernels
from nemo_automodel.components.distributed.blockdiag_cp import runtime as bd_runtime
from nemo_automodel.components.distributed.blockdiag_cp import state as bd_state

B, S, E, HQ, HKV, D = 1, 256, 128, 8, 4, 64

# (name, per-document token counts, zero-id pad tail); lengths sum to S.
SCENARIOS = [
    ("straddle+pad", [100, 80, 44], 32),  # doc 2 straddles the rank boundary at 128
    ("rank1-all-pad", [120], 136),  # rank 1's whole chunk is padding
    ("single-doc", [256], 0),  # one document spans both ranks
]

# (attn_backend, kv_exchange, dtype); dense exercises the masked-SDPA fallback in
# fp32 for a tight tolerance, flash/TE are the bf16 varlen kernels.
CASES = [("dense", "allgather", torch.float32)] + [
    (backend, exch, torch.bfloat16) for backend in ("flash", "te") for exch in ("allgather", "halo", "a2a")
]


class _TinyAttn(nn.Module):
    """Minimal q/k/v/o-projection attention block for CP-vs-dense grad parity."""

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.q_proj = nn.Linear(E, HQ * D, bias=False, dtype=dtype, device=device)
        self.k_proj = nn.Linear(E, HKV * D, bias=False, dtype=dtype, device=device)
        self.v_proj = nn.Linear(E, HKV * D, bias=False, dtype=dtype, device=device)
        self.o_proj = nn.Linear(HQ * D, E, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, sdpa_fn: Callable) -> torch.Tensor:
        """Project, attend (GQA), and re-project one sequence chunk.

        Args:
            x: Hidden states ``[B, L, E]`` (``B`` = batch, ``L`` = this chunk's
                sequence length -- full ``S`` for the reference, local shard for
                CP -- ``E`` = embedding dim).
            sdpa_fn: SDPA-compatible callable applied to ``q`` ``[B, HQ, L, D]``
                and ``k``/``v`` ``[B, HKV, L, D]``.

        Returns:
            Output hidden states ``[B, L, E]``.
        """
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, HQ, D).transpose(1, 2)  # [B, HQ, L, D]
        k = self.k_proj(x).view(b, s, HKV, D).transpose(1, 2)  # [B, HKV, L, D]
        v = self.v_proj(x).view(b, s, HKV, D).transpose(1, 2)
        o = sdpa_fn(q, k, v, enable_gqa=True)  # [B, HQ, L, D]
        return self.o_proj(o.transpose(1, 2).reshape(b, s, HQ * D))


def _dense_blockdiag_sdpa(doc_ids: torch.Tensor) -> Callable:
    """Single-process dense block-causal SDPA reference over the full sequence.

    Args:
        doc_ids: Per-position document ids ``[1, S]`` (0 == padding) on the full
            packed sequence.

    Returns:
        An SDPA-compatible callable: ``q`` ``[B, Hq, S, D]``, ``k``/``v``
        ``[B, Hkv, S, D]`` -> ``[B, Hq, S, D]`` under the per-document causal mask.
    """

    def fn(q, k, v, enable_gqa=False):
        if enable_gqa and k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        allow = bd_kernels._cp_blockdiag_mask(doc_ids, 0, q.shape[2], k.shape[2], q.shape[0])
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allow)

    return fn


def _make_doc_ids(seg_lens: list[int], pad: int, device) -> torch.Tensor:
    """Build ``[1, S]`` document ids: 1-based per-document runs + a zero pad tail."""
    ids = []
    for d, n in enumerate(seg_lens, start=1):
        ids += [d] * n
    ids += [0] * pad
    assert len(ids) == S, (len(ids), S)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def _rel_diff(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs difference between same-shape tensors, scaled by max |ref| (>= 1e-3)."""
    d = (got.float() - ref.float()).abs().max().item()
    return d / max(ref.float().abs().max().item(), 1e-3)


def _run_case(cp_mesh, device, backend, exch, dtype, seg_lens, pad):
    """Run one (backend, kv_exchange, scenario) parity case on this rank.

    Returns:
        ``(ok, detail)``: whether every check passed locally, and a per-metric
        relative-diff summary string.
    """
    rank = dist.get_rank()
    world = cp_mesh.size()
    local_len = S // world
    off = rank * local_len
    tol_out = 1e-3 if dtype is torch.float32 else 2e-2
    tol_grad = 2e-3 if dtype is torch.float32 else 4e-2

    doc_ids = _make_doc_ids(seg_lens, pad, device)  # [1, S]
    torch.manual_seed(1234)  # identical params on every rank and for the reference
    model = _TinyAttn(dtype, device)
    torch.manual_seed(7)
    x_data = torch.randn(B, S, E, device=device, dtype=dtype)
    torch.manual_seed(11)
    g = torch.randn(B, S, E, device=device, dtype=dtype)  # upstream grad for out [B, S, E]
    g[:, doc_ids[0] == 0, :] = 0  # padding rows carry no upstream gradient

    # ---- single-process dense reference (full sequence, same weights) ----
    x_ref = x_data.clone().requires_grad_(True)
    out_ref = model(x_ref, _dense_blockdiag_sdpa(doc_ids))
    (out_ref * g).sum().backward()
    ref_param_grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()}
    ref_x_grad = x_ref.grad.detach().float().clone()
    model.zero_grad(set_to_none=True)

    # ---- CP run through the production batch/ctx + SDPA path ----
    configure_cp_varlen(attn_backend=backend, kv_exchange=exch)
    x_cp = x_data.clone().requires_grad_(True)
    batch = {"inputs_embeds": x_cp, "_packed_seq_ids": doc_ids.clone()}
    ctx, sharded = make_cp_blockdiag_batch_and_ctx(cp_mesh, None, batch)
    x_loc = sharded["inputs_embeds"]  # [B, local_len, E], differentiable slice of x_cp
    with ctx():
        step_state = bd_state._CP_BLOCKDIAG_STATE.get()
        # Pin the KV-exchange path actually taken (memoized; reused by the sdpa call).
        path, _, why = bd_runtime._select_kv_exchange_path(
            step_state,
            step_state["group"],
            step_state["doc_ids"],
            local_len,
            device,
            step_state["row_offset"],
            query_dtype=dtype,
        )
        expected = exch if backend != "dense" else "allgather"
        if path != expected:
            return False, f"kv path {path} != expected {expected} ({why})"
        out_loc = model(x_loc, cp_blockdiag_sdpa)  # [B, local_len, E]
        (out_loc * g[:, off : off + local_len]).sum().backward()

    ok = True
    msgs = [f"path={path}"]
    real = doc_ids[0, off : off + local_len] > 0
    if real.any():
        rel = _rel_diff(out_loc.detach()[0][real], out_ref.detach()[0, off : off + local_len][real])
        ok &= rel < tol_out
        msgs.append(f"out={rel:.1e}")
    rel = _rel_diff(x_cp.grad[:, off : off + local_len], ref_x_grad[:, off : off + local_len])
    ok &= rel < tol_grad
    msgs.append(f"dx={rel:.1e}")
    for n, p in model.named_parameters():
        # An all-padding rank's queries are disconnected from its zero output (only
        # K/V keep the 0-weighted grad attachment), so q_proj may have no grad there.
        g32 = torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.detach().float().clone()
        dist.all_reduce(g32, op=dist.ReduceOp.SUM)  # rank contributions sum to the full-batch grad
        rel = _rel_diff(g32, ref_param_grads[n])
        ok &= rel < tol_grad
        msgs.append(f"d{n.split('.')[0]}={rel:.1e}")
    model.zero_grad(set_to_none=True)
    return ok, " ".join(msgs)


def main() -> int:
    """Run the case matrix; return 0 iff every case passed on every rank."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("[skip] block-diagonal CP 2-rank parity requires 2 CUDA devices", file=sys.stderr)
        return 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise SystemExit(f"expected torchrun --nproc-per-node=2, got world_size={world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    failures = []
    for backend, exch, dtype in CASES:
        reason = None
        if backend != "dense":
            # Environment-derived, so every rank skips (or runs) the case together.
            reason = bd_kernels._varlen_backend_unavailable_reason(backend, dtype, device)
        if reason is not None:
            if rank == 0:
                print(f"[skip] {backend}/{exch}: {reason}")
            continue
        for name, seg_lens, pad in SCENARIOS:
            ok, detail = _run_case(cp_mesh, device, backend, exch, dtype, seg_lens, pad)
            flag = torch.tensor([1 if ok else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            passed = bool(flag.item())
            if rank == 0:
                print(f"[{'ok' if passed else 'FAIL'}] {backend:<5} {exch:<9} {name:<14} {detail}")
            if not passed:
                failures.append((backend, exch, name))
    dist.barrier()
    if rank == 0:
        print("[ALL CASES PASSED]" if not failures else f"[{len(failures)} CASES FAILED] {failures}")
    dist.destroy_process_group()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
