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

"""Four-rank CPU/Gloo parity of separate-mesh KD teacher logits assembled on the transport rank only.

The teacher mesh is ``(cp=2, tp=2)``. Every rank holds a load-balanced CP chunk
of its vocabulary shard (9 ids over two ranks: 5 and 4). The transport rank
(cp = tp = 0) must receive exactly what the previous all-gather implementation
produced on every rank, while the other ranks return None, never call
``DTensor.full_tensor``, and never all-gather logits.
"""

from __future__ import annotations

import json
import os
import socket
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_WORLD_SIZE = 4


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _check(rank: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor
    from torch.distributed.tensor.debug import CommDebugMode

    from nemo_automodel.components.distributed.context_parallel.utils import unshard_context_parallel_tensor
    from nemo_automodel.recipes import kd_utils

    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
    cp_rank = mesh["cp"].get_local_rank()
    # Each CP rank holds its own (batch=2, local_sequence=4, vocab=9) chunk, sharded over TP.
    chunk = torch.randn(2, 4, 9, generator=torch.Generator().manual_seed(100 + cp_rank))
    logits = distribute_tensor(chunk, mesh["tp"], [Shard(-1)])
    # The previous implementation, which materialized full logits on every rank.
    expected = unshard_context_parallel_tensor(mesh["cp"], logits.full_tensor(), seq_dim=1).narrow(1, 0, 7)

    original = DTensor.full_tensor

    def forbidden(*args, **kwargs):
        raise AssertionError("DTensor.full_tensor was called")

    DTensor.full_tensor = forbidden
    try:
        with CommDebugMode() as comm:
            actual = kd_utils.materialize_teacher_logits(logits, device_mesh=mesh, sequence_length=7)
    finally:
        DTensor.full_tensor = original

    counts = {str(op).split(".")[-1]: count for op, count in comm.get_comm_counts().items()}
    # One all-gather of the CP position codes ([1, sequence] int64); logits only move by gather.
    assert counts.get("all_gather_into_tensor") == 1, counts
    assert counts.get("_allgather_base_", 0) == 0 and counts.get("allgather_", 0) == 0, counts
    assert counts.get("gather_") == (2 if cp_rank == 0 else 1), counts
    if rank == 0:
        assert actual is not None and actual.is_contiguous()
        assert torch.equal(actual, expected)
    else:
        assert actual is None


def _worker(rank: int, port: int, result_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        try:
            _check(rank)
            outcome = "ok"
        except Exception:
            outcome = traceback.format_exc()
    finally:
        dist.destroy_process_group()
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(outcome))


@pytest.mark.runtime_budget(30, hard_timeout=90, reason="spawns four Gloo ranks once")
def test_teacher_logits_assembled_on_transport_rank_only(tmp_path: Path) -> None:
    mp.spawn(_worker, args=(_free_port(), str(tmp_path)), nprocs=_WORLD_SIZE, join=True)
    outcomes = [json.loads(Path(tmp_path, f"rank{rank}.json").read_text()) for rank in range(_WORLD_SIZE)]
    failures = [f"rank {rank}:\n{outcome}" for rank, outcome in enumerate(outcomes) if outcome != "ok"]
    assert not failures, "\n".join(failures)
