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

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nemo_automodel.components.distributed.utils as du


class _FakeDist(SimpleNamespace):
    """
    Very small façade that satisfies the subset of the torch.distributed API
    that the utilities under test rely on.  It keeps the public surface narrow
    enough to be understandable at a glance.
    """

    def __init__(self, *, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self._rank = rank
        self._world_size = world_size
        # The code only needs ReduceOp.SUM
        self.ReduceOp = SimpleNamespace(SUM="sum")
        # Fabricate FSDP type hierarchy deep enough for isinstance checks
        fsdp_mod = SimpleNamespace(
            _fully_shard=SimpleNamespace(_fully_shard=SimpleNamespace(FSDPModule=type("DummyFSDP", (), {})))
        )
        self.fsdp = fsdp_mod
        self._initialised = True

    def is_initialized(self) -> bool:  # noqa: D401
        return self._initialised

    def get_rank(self) -> int:  # noqa: D401
        return self._rank

    def get_world_size(self) -> int:  # noqa: D401
        return self._world_size

    # All-reduce/barrier/abort/destroy are no-ops for the purpose of unit tests
    def all_reduce(self, *_, **__):  # noqa: D401
        pass

    def barrier(self, *_, **__):  # noqa: D401
        pass

    def abort(self, *_, **__):  # noqa: D401
        raise RuntimeError("abort called (simulated)")

    def destroy_process_group(self):  # noqa: D401
        self._initialised = False


@pytest.fixture()
def patch_dist(monkeypatch):
    """
    Replace `torch.distributed` **inside the utils module** with a lightweight
    fake implementation so tests do not need an actual back-end (NCCL / Gloo).
    """
    fake = _FakeDist()
    monkeypatch.setattr(du.torch, "distributed", fake, raising=False)
    # The module keeps a short alias ``dist``; patch it as well
    monkeypatch.setattr(du, "dist", fake, raising=False)
    yield fake



def test_first_rank_per_node_single_gpu(monkeypatch, patch_dist):
    """
    In the absence of a distributed init the context manager should behave like
    a regular `nullcontext()`, returning True for the guarded block.
    """
    # Pretend that dist is *not* initialised for this test
    patch_dist._initialised = False
    monkeypatch.setattr(du.dist, "is_initialized", lambda: False, raising=False)

    with du.FirstRankPerNode() as is_first:
        assert is_first is True



def test_reduce_loss_no_dp(monkeypatch):
    """
    With dp_group=None the routine must simply sum the supplied tensors and
    construct the correct denominator.
    """
    losses = [torch.tensor(1.0), torch.tensor(3.0)]
    tokens = torch.tensor(4)

    loss, denom = du.reduce_loss(losses, tokens, per_token_loss=True, dp_group=None)
    assert torch.isclose(loss, torch.tensor(4.0)), loss
    assert torch.equal(denom, torch.tensor(4)), denom


def test_get_sync_ctx(monkeypatch, patch_dist):
    """
    If the model is neither DDP nor FSDP the utility must return a
    `nullcontext`.
    """

    class Plain(torch.nn.Linear):
        pass

    ctx = du.get_sync_ctx(Plain(2, 2), is_optim_step=False)
    # entering/exiting the context must be a no-op
    with ctx:
        pass
