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

"""Unit, layout-simulation and gloo-parity tests for the pooled pipeline recv buffers.

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

from nemo_automodel.components.distributed.pipelining import recv_buffer_pool
from nemo_automodel.components.distributed.pipelining.recv_buffer_pool import (
    _ring_size,
    install_recv_buffer_pool,
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


def test_negative_slack_is_rejected():
    from nemo_automodel.components.distributed.pipelining.config import PipelineConfig

    with pytest.raises(ValueError):
        install_recv_buffer_pool(slack=-1)
    with pytest.raises(ValueError):
        PipelineConfig(pp_recv_buffer_pool=True, pp_recv_buffer_pool_slack=-1)
    assert PipelineConfig(pp_recv_buffer_pool=True).pp_recv_buffer_pool_slack == 2


@pytest.fixture
def fresh_install(monkeypatch):
    """Reset the process-wide install flag around a test and restore it afterwards."""
    monkeypatch.setattr(recv_buffer_pool, "_INSTALLED", False)
    monkeypatch.setattr(recv_buffer_pool, "_INSTALLED_LAYOUT", None)
    yield


def _fake_stage_classes(layout: str):
    """Minimal stand-ins for torch's _PipelineStageBase / PipelineStage exposing one recv-info layout."""

    class FakeBase:
        def __init__(self, num_stages, stage_index):
            self.num_stages = num_stages
            self.stage_index = stage_index
            self.is_first = stage_index == 0
            self.is_last = stage_index == num_stages - 1
            self.args_recv_info = {}
            self.grad_recv_info = {}
            self.chunks = None

    class FakeManual(FakeBase):
        pass

    if layout == "per-direction-setup":

        def _setup_forward_recv_info(self, num_microbatches, has_backward):
            for c in range(num_microbatches):
                self.args_recv_info[c] = (object(),)

        def _setup_backward_recv_info(self, num_microbatches):
            self.chunks = num_microbatches
            for c in range(num_microbatches):
                self.grad_recv_info[c] = (object(),)

        FakeManual._setup_forward_recv_info = _setup_forward_recv_info
        FakeBase._setup_backward_recv_info = _setup_backward_recv_info
    elif layout == "prepare-infra":

        def _prepare_forward_infra(self, num_microbatches, args, kwargs=None):
            for c in range(num_microbatches):
                self.args_recv_info[c] = (object(),)
            return ("outputs",)

        def _prepare_backward_infra(self, num_microbatches):
            self.chunks = num_microbatches
            for c in range(num_microbatches):
                self.grad_recv_info[c] = (object(),)
            return None

        FakeManual._prepare_forward_infra = _prepare_forward_infra
        FakeBase._prepare_backward_infra = _prepare_backward_infra
    return FakeBase, FakeManual


def _install_on_fakes(monkeypatch, layout):
    from torch.distributed.pipelining import stage as stage_mod

    base, manual = _fake_stage_classes(layout)
    monkeypatch.setattr(stage_mod, "_PipelineStageBase", base)
    monkeypatch.setattr(stage_mod, "PipelineStage", manual)
    return base, manual


def _distinct(d):
    return len({id(v) for v in d.values()})


@pytest.mark.parametrize("layout", ["per-direction-setup", "prepare-infra"])
def test_both_layouts_pool_to_ring_and_restore_chunks(fresh_install, monkeypatch, layout):
    """Each known torch layout is bound: middle stage pools both directions to K sets, chunks stays true."""
    _base, manual = _install_on_fakes(monkeypatch, layout)
    assert install_recv_buffer_pool(slack=2) is True
    assert recv_buffer_pool._INSTALLED_LAYOUT == layout

    stage = manual(num_stages=4, stage_index=1)  # in-flight 3 + slack 2 = K 5
    if layout == "per-direction-setup":
        stage._setup_forward_recv_info(_MB, True)
        stage._setup_backward_recv_info(_MB)
    else:
        assert stage._prepare_forward_infra(_MB, (), None) == ("outputs",)
        stage._prepare_backward_infra(_MB)
    assert len(stage.args_recv_info) == _MB and _distinct(stage.args_recv_info) == 5
    assert len(stage.grad_recv_info) == _MB and _distinct(stage.grad_recv_info) == 5
    assert stage.chunks == _MB
    # the ring wraps: chunk k reuses chunk 0's buffer set
    assert stage.args_recv_info[5] is stage.args_recv_info[0]

    first = manual(num_stages=4, stage_index=0)
    last = manual(num_stages=4, stage_index=3)
    if layout == "per-direction-setup":
        first._setup_forward_recv_info(_MB, True)
        last._setup_backward_recv_info(_MB)
    else:
        first._prepare_forward_infra(_MB, (), None)
        last._prepare_backward_infra(_MB)
    # first stage has no forward recv, last stage no grad recv: both keep stock allocation
    assert _distinct(first.args_recv_info) == _MB
    assert _distinct(last.grad_recv_info) == _MB


def test_install_fails_open_on_unknown_layout(fresh_install, monkeypatch):
    from torch.distributed.pipelining import stage as stage_mod

    class Bare:
        pass

    monkeypatch.setattr(stage_mod, "_PipelineStageBase", Bare)
    monkeypatch.setattr(stage_mod, "PipelineStage", Bare)
    assert install_recv_buffer_pool(slack=2) is False
    assert recv_buffer_pool._INSTALLED is False and recv_buffer_pool._INSTALLED_LAYOUT is None


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


def _train_once(rank: int) -> tuple[list[float], float, object]:
    """Build a fresh pp4 stage for this rank and train 1F1B for a few steps.

    Returns:
        (losses, param_sum, stage): per-step mean losses (non-empty on the last
        rank only), the double-precision sum of this stage's parameters, and the
        PipelineStage (to inspect its recv-info maps).
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
    return losses, param_sum, stage


def _parity_worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    faulthandler.dump_traceback_later(240, exit=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.set_num_threads(1)

        stock_losses, stock_sum, _ = _train_once(rank)

        assert install_recv_buffer_pool(slack=2)
        pooled_losses, pooled_sum, stage = _train_once(rank)

        # The pool really aliased: only K distinct buffer sets remain per direction.
        k = _ring_size(stage, _MB, 2)
        if rank != 0:
            assert len(stage.args_recv_info) == _MB and _distinct(stage.args_recv_info) == k, rank
        if rank != world_size - 1:
            assert len(stage.grad_recv_info) == _MB and _distinct(stage.grad_recv_info) == k, rank

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


def _too_small_ring_worker(rank: int, world_size: int, port: int) -> None:
    """Negative control: a ring below the 1F1B in-flight depth must NOT reproduce the stock run."""
    import torch.distributed as dist

    faulthandler.dump_traceback_later(240, exit=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.set_num_threads(1)

        stock_losses, stock_sum, _ = _train_once(rank)

        # stage 1 has 3 chunks in flight; force a 2-set ring so a prefetch lands
        # on a buffer whose chunk is still waiting for its backward.
        recv_buffer_pool._ring_size = lambda stage, num_microbatches, slack: 2
        assert install_recv_buffer_pool(slack=0)
        diverged = False
        try:
            small_losses, small_sum, _ = _train_once(rank)
            diverged = (rank == world_size - 1 and small_losses != stock_losses) or small_sum != stock_sum
        except RuntimeError:
            diverged = True  # autograd may also refuse the overwritten saved input
        flags = [None] * world_size
        dist.all_gather_object(flags, diverged)
        assert any(flags), ("too-small ring reproduced the stock run bitwise", flags)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.run_only_on("GPU")
def test_ring_below_inflight_depth_is_detected_by_the_harness():
    mp.spawn(_too_small_ring_worker, args=(_PP, _free_port()), nprocs=_PP, join=True)
