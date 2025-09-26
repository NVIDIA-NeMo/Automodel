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
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)

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


def test_timer_context_manager_with_barrier_and_restart():
    """
    Using Timer as a context manager should start/stop automatically. When elapsed is called
    while the timer is running with barrier=True, it should stop, report, reset, and restart.
    """
    t = Timer("cm-barrier")  # noqa: F821
    t.set_barrier_group(object())
    # Use context manager with barrier
    with t.with_barrier(True):
        time.sleep(0.01)
        # Call elapsed while started; should handle stop/reset/restart internally
        e = t.elapsed(reset=True, barrier=True)
        assert e > 0.0
        time.sleep(0.005)
    # After exiting context, timer is stopped; elapsed without reset keeps value 0 due to prior reset
    assert t.elapsed(reset=False) >= 0.0


def test_dummy_timer_noops_and_active_time_raises():
    dummy = DummyTimer()  # noqa: F821
    # start/stop/reset should be no-ops
    dummy.start()
    dummy.stop()
    dummy.reset()
    # active_time should raise
    with pytest.raises(Exception):
        _ = dummy.active_time()


def test_timers_call_existing_mismatch_and_dummy_with_barrier():
    timers = Timers(log_level=1, log_option="max")  # noqa: F821
    t = timers("same", log_level=1)
    # Calling again with different log_level should assert
    with pytest.raises(AssertionError):
        _ = timers("same", log_level=0)
    # Requesting a timer above allowed level returns DummyTimer and should support barrier via context
    dt = timers("dummy", log_level=2, barrier=True)
    assert isinstance(dt, DummyTimer)  # noqa: F821
    # Ensure context manager path exercises __enter__/__exit__ on DummyTimer
    with dt:
        pass


def test_get_all_timers_string_variants_and_names_none():
    # max/minmax
    timers_max = Timers(log_level=2, log_option="max")  # noqa: F821
    a = timers_max("a", log_level=1)
    a.start()
    time.sleep(0.005)
    a.stop()
    out_max = timers_max.get_all_timers_string(names=None, normalizer=1.0, reset=True, barrier=False)
    assert out_max is not None and "max" in out_max.lower()

    timers_minmax = Timers(log_level=2, log_option="minmax")  # noqa: F821
    b = timers_minmax("b", log_level=1)
    b.start()
    time.sleep(0.003)
    b.stop()
    out_minmax = timers_minmax.get_all_timers_string(names=["b"], normalizer=1.0, reset=True, barrier=False)
    assert out_minmax is not None and "(" in out_minmax and ")" in out_minmax

    # all-ranks string: when no timers recorded for provided names, should return None
    timers_all = Timers(log_level=2, log_option="all")  # noqa: F821
    none_out = timers_all.get_all_timers_string(names=["missing"], normalizer=1.0, reset=True, barrier=False)
    assert none_out is None

    # all-ranks string with an existing timer
    c = timers_all("c", log_level=1)
    c.start()
    time.sleep(0.002)
    c.stop()
    out_all = timers_all.get_all_timers_string(names=["c"], normalizer=1.0, reset=True, barrier=False)
    assert out_all is not None and "times across ranks" in out_all


def test_timers_write_and_wandb():
    timers = Timers(log_level=2, log_option="max")  # noqa: F821
    t = timers("write", log_level=1)
    t.start()
    time.sleep(0.004)
    t.stop()

    added = []

    class TBWriter:
        def add_scalar(self, name, value, iteration):
            added.append((name, value, iteration))

    tb = TBWriter()
    timers.write(names=["write"], writer=tb, iteration=5)
    assert any(n == "write-time" and i == 5 for (n, _, i) in added)

    logged = []

    class WBWriter:
        def log(self, data, iteration):
            logged.append((data, iteration))

    wb = WBWriter()
    timers.write_to_wandb(names=["write"], writer=wb, iteration=7)


def test_reload_uses_legacy_all_gather_when_torch_version_old(monkeypatch):
    # Force version check to return False and reload module
    import nemo_automodel.shared.import_utils as iu

    monkeypatch.setattr(iu, "is_torch_min_version", lambda v, check_equality=True: False, raising=False)

    mod = importlib.reload(importlib.import_module("nemo_automodel.components.training.timers"))
    # Under our fixture, both APIs point to the same stub function but attribute must resolve to _all_gather_base
    assert getattr(torch.distributed, "_all_gather_base") is mod.dist_all_gather_func

    # Restore original module state for subsequent tests
    importlib.reload(mod)


def test_existing_timer_with_barrier_return_path():
    timers = Timers(log_level=2, log_option="max")  # noqa: F821
    t1 = timers("exist", log_level=1)
    # Re-call with barrier=True should return same instance (with context barrier configured)
    t2 = timers("exist", barrier=True)
    assert t1 is t2


def test_default_log_level_is_max_value_timer_created():
    timers = Timers(log_level=2, log_option="max")  # noqa: F821
    # No log_level provided should default to _max_log_level and create a real Timer
    t = timers("no_level")
    assert isinstance(t, Timer)  # noqa: F821


def test_barrier_true_path_in_get_elapsed_time_all_ranks():
    timers = Timers(log_level=2, log_option="all")  # noqa: F821
    t = timers("barrier_ranks", log_level=1)
    t.start()
    time.sleep(0.002)
    t.stop()
    out = timers.get_all_timers_string(names=["barrier_ranks"], barrier=True)
    assert out is not None


def test_minmax_returns_none_when_no_timers_present():
    timers = Timers(log_level=2, log_option="minmax")  # noqa: F821
    out = timers.get_all_timers_string(names=["does_not_exist"], normalizer=1.0, reset=True, barrier=False)
    assert out is None
