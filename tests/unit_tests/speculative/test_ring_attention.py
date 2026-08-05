# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""World-size-1 numerics for the cached EAGLE-3 ring attention.

At ``world_size == 1`` the ring reduces to a single FlashAttention block plus the
per-position TTT diagonals, so it must match a plain eager reference for the same
mixed causal + diagonal softmax. This pins the forward and backward math (the ring
comms are exercised separately by the 2-GPU checks).
"""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from nemo_automodel.components.speculative.eagle.ring_attention import HAVE_FLASH_ATTN

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_FLASH_ATTN,
    reason="ring attention needs CUDA + flash-attn",
)


def _eager_cached_reference(q, cache_k, cache_v, scale):
    """Eager EAGLE-3 mixed attention: block-0 causal + per-position diagonals.

    All tensors are ``[B, T, H, D]``; block 0 is the causal sequence attention and
    blocks ``i>=1`` each contribute one diagonal key ``(q_t . k_i_t)`` with value
    ``v_i_t`` to the joint softmax. Returns ``[B, T, H, D]``.
    """
    q_f = q.float()
    k0, v0 = cache_k[0].float(), cache_v[0].float()
    B, T, H, D = q_f.shape
    s0 = torch.einsum("bthd,bshd->bhts", q_f, k0) * scale  # [B, H, T, T]
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
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


def _ring_worker(rank: int, port: int) -> None:
    # Run in a spawned child so the process group never collides with one another
    # test in the suite left initialized. NCCL (not gloo): the ring backward issues
    # a self send/recv even at world_size=1, which gloo rejects but NCCL handles.
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)

        from nemo_automodel.components.speculative.eagle.ring_attention import cached_ring_attention

        torch.manual_seed(0)
        B, T, H, D, NB = 1, 64, 4, 32, 3
        dev = torch.device("cuda")
        scale = D**-0.5
        q = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        ck = [torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(NB)]
        cv = [torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(NB)]

        out = cached_ring_attention(q, ck, cv, torch.distributed.group.WORLD, scale)
        ref = _eager_cached_reference(q, ck, cv, scale)
        assert out.shape == ref.shape
        rel = (out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
        assert rel < 2e-2, f"forward mismatch rel={rel.item():.3e}"

        grad_out = torch.randn_like(out)
        dq = torch.autograd.grad(out.float(), q, grad_out.float(), retain_graph=True)[0]
        dq_ref = torch.autograd.grad(ref, q, grad_out.float())[0]
        dq_rel = (dq - dq_ref).abs().max() / dq_ref.abs().max().clamp_min(1e-6)
        assert dq_rel < 5e-2, f"dq mismatch rel={dq_rel.item():.3e}"
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_cached_ring_attention_world1_matches_eager():
    mp.spawn(_ring_worker, args=(_free_port(),), nprocs=1, join=True)
