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

import signal
from typing import Any

import pytest
import torch

import nemo_automodel.utils.sig_utils as sutils


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

def test_get_device_nccl_cpu(monkeypatch):
    """
    get_device should return the (mocked) CUDA device when the backend is 'nccl'.
    We monkey-patch torch.distributed.get_backend -> 'nccl' and request rank 3.
    The returned torch.device must have type == 'cuda' and index == 3.
    """
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    dev = sutils.get_device(local_rank=3)
    assert isinstance(dev, torch.device)
    assert dev.type == "cuda"
    assert dev.index == 3


def test_get_device_gloo(monkeypatch):
    """
    get_device should return CPU when the backend is 'gloo'.
    """
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    dev = sutils.get_device()
    assert dev.type == "cpu"


def test_get_device_unknown_backend(monkeypatch):
    """
    An unsupported backend must raise RuntimeError.
    """
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "mpi")

    with pytest.raises(RuntimeError):
        _ = sutils.get_device()


# ---------------------------------------------------------------------------
# all_gather_item
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_distributed(monkeypatch):
    """
    Fixture that installs a minimal fake torch.distributed API so we can call
    sutils.all_gather_item without actually initialising a process-group.
    """
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")  # re-used by get_device

    # A no-op all_gather that simply copies the input tensor into every output slot
    def _fake_all_gather(out_list: list[torch.Tensor],
                         in_tensor: torch.Tensor,
                         *_args: Any, **_kw: Any) -> None:
        for out in out_list:
            out.copy_(in_tensor)

    monkeypatch.setattr(torch.distributed, "all_gather", _fake_all_gather)


def test_all_gather_item_collects_scalar(fake_distributed, monkeypatch):
    """
    all_gather_item should return a list that contains the scalar value from
    each rank.  We fake a two-rank world so the expected result is [42, 42].
    """
    # Ensure get_device inside all_gather_item picks CPU so CUDA is never used
    monkeypatch.setattr(sutils, "get_device", lambda _rank=None: torch.device("cpu"))

    gathered = sutils.all_gather_item(42, dtype=torch.int64)
    assert gathered == [42, 42]


def test_all_gather_item_falls_back_when_not_initialised(monkeypatch):
    """
    If torch.distributed is not initialised, the helper must simply return [item].
    """
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    result = sutils.all_gather_item("nothing-to-see", dtype=torch.int32)
    assert result == ["nothing-to-see"]


# ---------------------------------------------------------------------------
# DistributedSignalHandler
# ---------------------------------------------------------------------------

def test_signal_handler_installs_and_restores(monkeypatch):
    """
    Entering the context must install a new handler, set _signal_received=False,
    and on exit restore the original handler.
    """
    sig = signal.SIGUSR1  # safer than SIGTERM for unit tests
    original = signal.getsignal(sig)

    with sutils.DistributedSignalHandler(sig=sig) as h:
        # Handler must be different from the original
        current = signal.getsignal(sig)
        assert current is not original
        # Simulate a signal
        current(sig, None)
        assert h._signal_received is True
    # After context manager closes the original handler must be restored
    assert signal.getsignal(sig) is original


def test_signals_received_aggregates(monkeypatch):
    """
    signals_received() should delegate to all_gather_item; we patch that call
    so the method becomes deterministic.
    """
    expected = [False, True, False]

    def fake_all_gather_item(*_args: Any, **_kw: Any) -> list[bool]:
        return expected

    monkeypatch.setattr(sutils, "all_gather_item", fake_all_gather_item)

    handler = sutils.DistributedSignalHandler()
    assert handler.signals_received() == expected
