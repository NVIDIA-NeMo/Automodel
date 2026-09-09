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

"""Ulysses (all-to-all) context-parallel draft attention.

Two layers of coverage:

* CPU: the transport helpers' no-op (``uly == 1``) and divisibility guards, with
  ``dist.get_world_size`` monkeypatched (no process group needed).
* GPU: the mixed causal-block-0 + TTT-diagonal attention must equal a plain eager
  reference for the same joint softmax, in both forward AND backward. Because the
  block-0 backward is the subtle part (it must use the merged joint-softmax lse,
  not block-0's own), the gradient is checked by value: ``world_size == 1`` pins
  the merge math (all-to-all is a no-op) and ``world_size == 2`` pins the real
  all-to-all against the single-process reference restricted to each shard.
"""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

import nemo_automodel.components.speculative.eagle.ulysses_attention as ua

# --------------------------------------------------------------------------- #
# CPU: transport helper guards (no process group required)
# --------------------------------------------------------------------------- #
def test_gather_is_noop_at_world_size_one(monkeypatch):
    monkeypatch.setattr(ua.dist, "get_world_size", lambda group: 1)
    x = torch.randn(2, 4, 8, 5)
    assert ua._gather_seq_scatter_heads(x, object()) is x


def test_scatter_is_noop_at_world_size_one(monkeypatch):
    monkeypatch.setattr(ua.dist, "get_world_size", lambda group: 1)
    x = torch.randn(2, 8, 4, 5)
    assert ua._scatter_seq_gather_heads(x, object()) is x


def test_gather_rejects_heads_not_divisible_by_ulysses(monkeypatch):
    monkeypatch.setattr(ua.dist, "get_world_size", lambda group: 3)
    with pytest.raises(ValueError, match="divide the head count"):
        ua._gather_seq_scatter_heads(torch.randn(1, 6, 4, 5), object())  # 4 heads, uly 3


def test_scatter_rejects_seq_not_divisible_by_ulysses(monkeypatch):
    monkeypatch.setattr(ua.dist, "get_world_size", lambda group: 3)
    with pytest.raises(ValueError, match="divide the gathered sequence length"):
        ua._scatter_seq_gather_heads(torch.randn(1, 4, 2, 5), object())  # seq 4, uly 3


# --------------------------------------------------------------------------- #
# GPU: forward + gradient equivalence to an eager reference (CUDA + flash-attn)
# --------------------------------------------------------------------------- #
_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not ua.HAVE_FLASH_ATTN,
    reason="Ulysses attention numerics need CUDA + flash-attn",
)


def _eager_cached_reference(q_bthd, cache_k, cache_v, scale):
    """Eager EAGLE-3 mixed attention: block-0 causal + per-position TTT diagonals.

    Args:
        q_bthd: Query, ``[batch, sequence, heads, head_dim]``.
        cache_k: List of keys, each ``[batch, sequence, heads, head_dim]``; index 0
            is the step-0 sequence key, ``i>=1`` the per-position diagonal steps.
        cache_v: List of values, same layout as ``cache_k``.
        scale: Softmax scale (``head_dim ** -0.5``).

    Returns:
        Attention output ``[batch, sequence, heads, head_dim]``.
    """
    q_f = q_bthd.float()
    k0, v0 = cache_k[0].float(), cache_v[0].float()
    B, T, H, D = q_f.shape
    s0 = torch.einsum("bthd,bshd->bhts", q_f, k0) * scale  # [B, H, T, T]
    causal = torch.tril(torch.ones(T, T, device=q_f.device, dtype=torch.bool))
    s0 = s0.masked_fill(~causal, float("-inf"))
    diags = [(q_f * cache_k[i].float()).sum(-1).transpose(1, 2) * scale for i in range(1, len(cache_k))]  # [B, H, T]
    logits = torch.cat([s0] + [d.unsqueeze(-1) for d in diags], dim=-1)  # [B, H, T, T + NB-1]
    w = torch.softmax(logits, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", w[..., :T], v0)
    for i in range(1, len(cache_v)):
        wi = w[..., T + i - 1].transpose(1, 2).unsqueeze(-1)  # [B, T, H, 1]
        out = out + wi * cache_v[i].float()
    return out


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _ulysses_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        group = torch.distributed.group.WORLD
        dev = torch.device("cuda")
        dtype = torch.bfloat16

        # Identical full tensors on every rank (same seed): each rank slices its own
        # contiguous shard, and the gather reconstructs the full sequence in order.
        # Short block-0 + several diagonals amplify the joint-softmax coupling so the
        # gradient check is sensitive to a block-0-only (wrong) backward.
        torch.manual_seed(0)
        B, T_full, H, D, NB = 1, 16, 4, 32, 4
        scale = D**-0.5
        q_full = torch.randn(B, T_full, H, D, device=dev, dtype=dtype)
        ck_full = [torch.randn(B, T_full, H, D, device=dev, dtype=dtype) for _ in range(NB)]
        cv_full = [torch.randn(B, T_full, H, D, device=dev, dtype=dtype) for _ in range(NB)]

        # Reference: full joint softmax, differentiable in q.
        q_ref = q_full.clone().requires_grad_(True)
        ref = _eager_cached_reference(q_ref, ck_full, cv_full, scale)  # [B, T_full, H, D]

        from nemo_automodel.components.speculative.eagle.ulysses_attention import cached_ulysses_attention

        T_local = T_full // world_size
        sl = slice(rank * T_local, (rank + 1) * T_local)
        q_shard = q_full[:, sl].contiguous().requires_grad_(True)  # [B, T_local, H, D]
        ck_shard = [ck_full[i][:, sl].contiguous() for i in range(NB)]
        cv_shard = [cv_full[i][:, sl].contiguous() for i in range(NB)]

        out = cached_ulysses_attention(q_shard, ck_shard, cv_shard, group, scale)  # [B, T_local, H, D]
        ref_shard = ref[:, sl].detach()
        rel = (out.float() - ref_shard).abs().max() / ref_shard.abs().max().clamp_min(1e-6)
        assert rel < 3e-2, f"[rank {rank}] forward mismatch rel={rel.item():.3e}"

        # Gradient by value: q_t affects only output t (causal), so the reference's
        # gradient restricted to the shard is exactly this shard's dq. A block-0-only
        # backward (missing the merged-lse term) fails this check.
        grad_out = torch.randn_like(out)
        dq = torch.autograd.grad(out.float(), q_shard, grad_out.float(), retain_graph=True)[0]
        dq_ref = torch.autograd.grad(ref[:, sl], q_ref, grad_out.float())[0][:, sl]
        dq_rel = (dq - dq_ref).abs().max() / dq_ref.abs().max().clamp_min(1e-6)
        assert dq_rel < 3e-2, f"[rank {rank}] dq mismatch rel={dq_rel.item():.3e}"
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@_GPU
def test_ulysses_world1_matches_eager():
    mp.spawn(_ulysses_worker, args=(1, _free_port()), nprocs=1, join=True)


@_GPU
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs for the real all-to-all")
def test_ulysses_world2_matches_full_sequence_reference():
    mp.spawn(_ulysses_worker, args=(2, _free_port()), nprocs=2, join=True)
