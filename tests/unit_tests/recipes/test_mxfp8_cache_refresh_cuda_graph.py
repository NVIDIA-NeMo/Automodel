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

from contextlib import contextmanager

import pytest
import torch

import nemo_automodel.recipes.llm.full_iteration_cuda_graph as full_graph
from nemo_automodel.recipes.llm.mxfp8_cache_refresh_cuda_graph import (
    MXFP8CacheRefreshCudaGraphError,
    MXFP8CacheRefreshCudaGraphManager,
)


class _FakeStream:
    def __init__(self):
        self.waited_for = []

    def wait_stream(self, stream):
        self.waited_for.append(stream)


class _FakeGraph:
    def __init__(self):
        self.replays = 0
        self.resets = 0

    def replay(self):
        self.replays += 1

    def reset(self):
        self.resets += 1


class _Target:
    def __init__(self):
        self.signature = ((100, 200),)
        self.events = []
        self._managed_owner_ids = frozenset({7, 11})

    @property
    def managed_owner_ids(self):
        return self._managed_owner_ids

    def graph_signature(self):
        return self.signature

    def eager_refresh(self):
        self.events.append("eager")
        return 2

    def capture_refresh(self):
        self.events.append("capture")

    def mark_replayed(self):
        self.events.append("mark")
        return 2


@contextmanager
def _fake_graph_context(_graph, **kwargs):
    assert kwargs["capture_error_mode"] == "thread_local"
    assert kwargs["stream"] is not None
    assert kwargs["pool"] is not None
    yield


@pytest.fixture
def mocked_cuda_graph_runtime(monkeypatch):
    current_stream = _FakeStream()
    capture_stream = _FakeStream()
    graphs = []

    def make_graph():
        graph = _FakeGraph()
        graphs.append(graph)
        return graph

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", make_graph)
    monkeypatch.setattr(torch.cuda, "graph", _fake_graph_context)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: object())
    monkeypatch.setattr(full_graph, "_SHARED_CAPTURE_STREAM", capture_stream)
    monkeypatch.setattr(full_graph, "_SHARED_GRAPH_POOL", object())
    return current_stream, capture_stream, graphs


def test_cache_refresh_manager_waits_for_full_graph_then_captures_and_replays(mocked_cuda_graph_runtime):
    current_stream, capture_stream, graphs = mocked_cuda_graph_runtime
    target = _Target()
    manager = MXFP8CacheRefreshCudaGraphManager(target)

    assert manager.managed_owner_ids == frozenset({7, 11})
    assert manager.run(capture_allowed=False) == 2
    assert manager.run(capture_allowed=True) == 2
    assert manager.run(capture_allowed=True) == 2

    assert target.events == ["eager", "capture", "mark", "mark"]
    assert manager.eager_refresh_count == 1
    assert manager.capture_count == 1
    assert manager.replay_count == 1
    assert graphs[0].replays == 2
    assert capture_stream.waited_for == [current_stream, current_stream]
    assert current_stream.waited_for == [capture_stream, capture_stream]

    manager.close()
    assert graphs[0].resets == 1
    manager.close()


def test_cache_refresh_manager_destroys_graph_before_returning_to_eager(mocked_cuda_graph_runtime):
    _current_stream, _capture_stream, graphs = mocked_cuda_graph_runtime
    target = _Target()
    manager = MXFP8CacheRefreshCudaGraphManager(target)

    manager.run(capture_allowed=True)
    manager.run(capture_allowed=False)

    assert graphs[0].resets == 1
    assert target.events == ["capture", "mark", "eager"]
    assert manager.is_captured is False


def test_cache_refresh_manager_fails_closed_before_replaying_changed_pointer(mocked_cuda_graph_runtime):
    _current_stream, _capture_stream, graphs = mocked_cuda_graph_runtime
    target = _Target()
    manager = MXFP8CacheRefreshCudaGraphManager(target)
    manager.run(capture_allowed=True)
    target.signature = ((101, 200),)

    with pytest.raises(MXFP8CacheRefreshCudaGraphError, match="owner or fixed-address destination changed"):
        manager.run(capture_allowed=True)
    assert graphs[0].replays == 1
    assert graphs[0].resets == 1

    with pytest.raises(MXFP8CacheRefreshCudaGraphError, match="failed closed"):
        manager.run(capture_allowed=True)


def test_cache_refresh_manager_retains_live_graph_when_reset_fails():
    class BrokenGraph(_FakeGraph):
        def reset(self):
            raise RuntimeError("reset failed")

    target = _Target()
    manager = MXFP8CacheRefreshCudaGraphManager(target)
    graph = BrokenGraph()
    manager._graph = graph

    with pytest.raises(RuntimeError, match="reset failed"):
        manager._destroy_graph()

    assert manager._graph is graph


def test_cache_refresh_manager_rejects_capture_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manager = MXFP8CacheRefreshCudaGraphManager(_Target())

    with pytest.raises(MXFP8CacheRefreshCudaGraphError, match="require CUDA"):
        manager.run(capture_allowed=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
def test_cache_refresh_graph_tracks_changed_owner_values_with_eager_parity():
    owner = torch.arange(32, dtype=torch.float32, device="cuda")
    cache = torch.empty_like(owner)

    class TensorTarget:
        managed_owner_ids = frozenset({id(owner)})

        def __init__(self):
            self.replays = 0

        def graph_signature(self):
            return ((owner.data_ptr(), cache.data_ptr(), tuple(owner.shape), owner.dtype),)

        def eager_refresh(self):
            cache.copy_(owner.square())
            self.replays += 1
            return 1

        def capture_refresh(self):
            cache.copy_(owner.square())

        def mark_replayed(self):
            self.replays += 1
            return 1

    target = TensorTarget()
    manager = MXFP8CacheRefreshCudaGraphManager(target)
    try:
        for step, capture_allowed in enumerate((False, True, True, True), start=1):
            owner.copy_(torch.arange(32, dtype=torch.float32, device="cuda") + step)
            expected = owner.square().clone()
            manager.run(capture_allowed=capture_allowed)
            torch.cuda.synchronize()
            torch.testing.assert_close(cache, expected, rtol=0, atol=0)
        assert target.replays == 4
    finally:
        manager.close()
