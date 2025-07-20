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

import importlib
import sys
import time
import types

import pytest
import torch

try:
    cuda_available = torch.cuda.is_available()
except:
    cuda_available = False


@pytest.fixture(autouse=True)
def patch_torch_distributed(monkeypatch):
    """
    Automatically patch torch.cuda and torch.distributed for every test.

    The real implementation requires GPUs and a multi-process environment.
    A minimal stub is sufficient for unit tests that only check the Python
    logic.
    """
    # CUDA stubs
    if not hasattr(torch, "cuda"):
        torch.cuda = types.SimpleNamespace()

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)

    # Distributed stubs
    dist_stub = types.ModuleType("torch.distributed")

    # Minimal API surface that NeMo timers touch
    dist_stub.get_world_size = lambda: 1
    dist_stub.get_rank = lambda: 0
    dist_stub.barrier = lambda group=None: None
    dist_stub.is_initialized = lambda: False  # helps _get_default_group check

    def _all_gather(dest: torch.Tensor, src: torch.Tensor):  # noqa: D401
        """
        Dummy all_gather / all_gather_into_tensor implementation for world_size=1.
        """
        dest.copy_(src)

    # Provide both APIs that the library may request.
    dist_stub.all_gather_into_tensor = _all_gather
    dist_stub._all_gather_base = _all_gather

    monkeypatch.setattr(torch, "distributed", dist_stub, raising=False)
    sys.modules["torch.distributed"] = dist_stub

    # Import the module *after* stubs are in place so it picks them up.
    global timers_mod
    timers_mod = importlib.import_module("nemo_automodel.components.training.timers")
    # Re-export so tests can use a short alias.
    globals().update(
        {
            "Timer": timers_mod.Timer,
            "DummyTimer": timers_mod.DummyTimer,
            "Timers": timers_mod.Timers,
        }
    )


# Individual unit tests
def test_dummy_timer_raises_on_elapsed():
    """DummyTimer.elapsed must raise to prevent accidental use."""
    dummy = DummyTimer()  # noqa: F821
    with pytest.raises(Exception):
        _ = dummy.elapsed()


def test_timer_basic_start_stop_elapsed():
    """
    Timer should accumulate elapsed time correctly between explicit
    start() and stop() calls.
    """
    t = Timer("unit-test")  # noqa: F821
    t.start()
    time.sleep(0.02)
    t.stop()

    measured = t.elapsed(reset=False)  # do not reset so we can re-query
    assert measured > 0.0
    # 20 ms sleep + small overhead. Expect < 100 ms on normal CI machines.
    assert measured < 0.1


def test_timer_double_start_fails():
    """Calling start() twice without an intermediate stop() must assert."""
    t = Timer("unit-test-double-start")  # noqa: F821
    t.start()
    with pytest.raises(AssertionError):
        t.start()


def test_timer_elapsed_resets_properly():
    """
    elapsed(reset=True) should zero the internal counter while leaving
    _active_time untouched.
    """
    t = Timer("reset-test")  # noqa: F821
    t.start()
    time.sleep(0.01)
    t.stop()

    first = t.elapsed(reset=True)
    second = t.elapsed(reset=False)  # should be 0 because of previous reset
    assert first > 0.0
    assert second == 0.0
    # active_time aggregates all usage and must still be >= first
    assert t.active_time() >= first


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_timers_collection_and_logging(monkeypatch, capsys):
    """
    End-to-end test of the Timers container:
      * creating timers via __call__
      * automatic DummyTimer routing based on log_level
      * string generation / stdout logging
    """
    timers = Timers(log_level=1, log_option="max")  # noqa: F821

    dummy = timers("foo", log_level=2)
    assert isinstance(dummy, DummyTimer)  # noqa: F821

    # log_level within threshold → real Timer
    real_timer = timers("bar", log_level=1)
    real_timer.start()
    time.sleep(0.015)
    real_timer.stop()

    # Ask Timers to print.  Rank==0 under the stub.
    timers.log(names=["bar"], normalizer=1.0, reset=True)

    captured = capsys.readouterr().out
    # Expect the name and the word "max" in the printed string.
    assert "bar" in captured
    assert "max" in captured.lower()
