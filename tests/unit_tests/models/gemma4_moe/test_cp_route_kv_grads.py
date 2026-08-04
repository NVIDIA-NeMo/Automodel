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

"""Gradient routing for the Gemma4 CP ring attention backward.

``_route_kv_grads_to_owners`` must stay a reduce-scatter. Hand-rolling it from all-pairs p2p
makes NCCL allocate connection buffers for every CP peer at the backward memory peak, which
OOMs. These tests pin both the numerics and the absence of p2p.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.models.gemma4_moe.cp_attention import _route_kv_grads_to_owners

CP_SIZE = 4
# batch > 1 on purpose: an owner-stacked [cp_size, batch, ...] passes the collective's
# dim-0 check only when batch == 1, so batch 1 would hide that bug.
SHAPE = (2, 2, 8, 4)  # [batch, kv_heads, seq_local, head_dim]


def _per_owner_grads(rank: int) -> dict[int, torch.Tensor]:
    """Deterministic per-owner dK for ``rank``: entry o is filled with (rank + 1) * (o + 1)."""
    return {owner: torch.full(SHAPE, float((rank + 1) * (owner + 1))) for owner in range(CP_SIZE)}


def _expected(rank: int) -> torch.Tensor:
    """Reference: sum over all ranks of their contribution for owner ``rank``."""
    total = sum((r + 1) * (rank + 1) for r in range(CP_SIZE))
    return torch.full(SHAPE, float(total))


def _worker(rank: int, world_size: int, port: int, out_q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        group = dist.group.WORLD
        grad_key, grad_value = _route_kv_grads_to_owners(
            _per_owner_grads(rank),
            _per_owner_grads(rank),
            cp_group=group,
            cp_rank=rank,
            cp_size=world_size,
        )
        out_q.put((rank, grad_key.clone(), grad_value.clone()))
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(180)
def test_route_kv_grads_matches_reference_reduction():
    """Each rank must end with the sum of every rank's contribution for its own owner slot."""
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    port = 29623
    procs = [ctx.Process(target=_worker, args=(r, CP_SIZE, port, out_q)) for r in range(CP_SIZE)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(CP_SIZE):
        rank, grad_key, grad_value = out_q.get(timeout=120)
        results[rank] = (grad_key, grad_value)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    assert set(results) == set(range(CP_SIZE))
    for rank, (grad_key, grad_value) in results.items():
        expected = _expected(rank)
        assert grad_key.shape == SHAPE
        torch.testing.assert_close(grad_key, expected)
        torch.testing.assert_close(grad_value, expected)


def test_route_kv_grads_uses_no_point_to_point(monkeypatch):
    """Routing must not open per-peer p2p connections; that is what OOMs, not the arithmetic."""
    import nemo_automodel.components.models.gemma4_moe.cp_attention as cpa

    def _fail(*args, **kwargs):
        raise AssertionError("_route_kv_grads_to_owners must not use point-to-point exchange")

    monkeypatch.setattr(cpa, "_direct_exchange", _fail)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", _fail)

    grads = {0: torch.full(SHAPE, 3.0)}
    captured = {}

    def _fake_reduce_scatter(out, inp, group=None):
        captured["input_shape"] = tuple(inp.shape)
        out.copy_(inp)

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", _fake_reduce_scatter)
    grad_key, grad_value = _route_kv_grads_to_owners(grads, grads, cp_group=None, cp_rank=0, cp_size=1)

    assert captured["input_shape"] == SHAPE  # owner-major on dim 0, not a [cp_size, batch, ...] stack
    torch.testing.assert_close(grad_key, torch.full(SHAPE, 3.0))
    torch.testing.assert_close(grad_value, torch.full(SHAPE, 3.0))
