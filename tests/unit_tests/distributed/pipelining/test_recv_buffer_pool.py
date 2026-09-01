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

"""Unit and gloo-parity tests for the pooled pipeline recv buffers.

The parity test trains the same pp4 1F1B pipeline twice inside each spawned
rank — first with stock recv buffers, then with the ring pool installed —
using identical seeds and data, and requires bitwise-identical loss
trajectories and per-stage parameter sums. It catches ring-too-small
corruption (a prefetch overwriting a buffer still needed by a chunk's
backward): the linear weight grads read the saved stage input, which IS the
recv buffer object.
"""

import faulthandler
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from nemo_automodel.components.distributed.pipelining.recv_buffer_pool import (
    _ring_size,
    schedule_supports_recv_pool,
)

_PP = 4
_HIDDEN = 64
_MB = 16  # microbatches per step (>> ring K)
_MBS = 4  # rows per microbatch
_STEPS = 3


class _FakeStage:
    def __init__(self, num_stages: int, stage_index: int):
        self.num_stages = num_stages
        self.stage_index = stage_index


def test_ring_size_covers_inflight_plus_slack():
    # Stage 0 of 8: 8 in flight + 2 slack, capped by the microbatch count.
    assert _ring_size(_FakeStage(8, 0), num_microbatches=64, slack=2) == 10
    # Late stage: small in-flight depth, floor of 2.
    assert _ring_size(_FakeStage(8, 7), num_microbatches=64, slack=0) == 2
    # Never exceeds the microbatch count.
    assert _ring_size(_FakeStage(8, 0), num_microbatches=4, slack=2) == 4


def test_schedule_gate_only_allows_bounded_inflight_schedules():
    assert schedule_supports_recv_pool("1f1b")
    assert schedule_supports_recv_pool("1F1B")
    assert not schedule_supports_recv_pool("gpipe")
    assert not schedule_supports_recv_pool("interleaved_1f1b")
    assert not schedule_supports_recv_pool(None)


class _Block(nn.Module):
    """One pipeline stage: two linears with a residual.

    ``forward`` takes and returns a tensor of shape [rows, hidden].
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(_HIDDEN, _HIDDEN)
        self.l2 = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear-relu-linear with a residual.

        Args:
            x: Tensor of shape [rows, hidden].

        Returns:
            Tensor of shape [rows, hidden].
        """
        return self.l2(torch.relu(self.l1(x))) + x


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _train_once(rank: int) -> tuple[list[float], float]:
    """Build a fresh pp4 stage for this rank and train 1F1B for a few steps.

    Returns:
        (losses, param_sum): per-step mean losses (non-empty on the last rank
        only) and the double-precision sum of this stage's parameters.
    """
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import Schedule1F1B

    torch.manual_seed(1234)  # same init on all ranks; each keeps its stage
    full = nn.Sequential(*[_Block() for _ in range(_PP)])
    stage_mod = full[rank]
    stage = PipelineStage(stage_mod, rank, _PP, torch.device("cpu"))

    def loss_fn(out, tgt):
        return torch.nn.functional.mse_loss(out, tgt)

    sched = Schedule1F1B(stage, n_microbatches=_MB, loss_fn=loss_fn)
    opt = torch.optim.SGD(stage_mod.parameters(), lr=0.05)

    g = torch.Generator().manual_seed(42)
    losses: list[float] = []
    for _ in range(_STEPS):
        x = torch.randn(_MB * _MBS, _HIDDEN, generator=g)
        tgt = torch.randn(_MB * _MBS, _HIDDEN, generator=g)
        opt.zero_grad(set_to_none=True)
        if rank == 0:
            sched.step(x)
        elif rank == _PP - 1:
            out_losses: list[torch.Tensor] = []
            sched.step(target=tgt, losses=out_losses)
            losses.append(torch.stack(out_losses).mean().item())
        else:
            sched.step()
        opt.step()

    param_sum = sum(p.double().sum().item() for p in stage_mod.parameters())
    return losses, param_sum


def _parity_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    faulthandler.dump_traceback_later(240, exit=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.set_num_threads(1)

        stock_losses, stock_sum = _train_once(rank)

        from nemo_automodel.components.distributed.pipelining.recv_buffer_pool import (
            install_recv_buffer_pool,
        )

        assert install_recv_buffer_pool(slack=2)
        pooled_losses, pooled_sum = _train_once(rank)

        # Bitwise equality: pooling only changes which buffer object receives
        # each chunk, never the values flowing through the schedule.
        assert pooled_losses == stock_losses, (rank, stock_losses, pooled_losses)
        assert pooled_sum == stock_sum, (rank, stock_sum, pooled_sum)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.run_only_on("GPU")
def test_pooled_recv_buffers_match_stock_1f1b_exactly():
    mp.spawn(_parity_worker, args=(_PP, _free_port()), nprocs=_PP, join=True)
