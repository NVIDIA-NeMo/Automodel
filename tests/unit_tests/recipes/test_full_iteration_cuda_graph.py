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

from contextlib import contextmanager, nullcontext

import pytest
import torch

import nemo_automodel.recipes.llm.full_iteration_cuda_graph as full_graph
from nemo_automodel.recipes.llm.full_iteration_cuda_graph import (
    FullIterationCudaGraphError,
    FullIterationCudaGraphManager,
    _StaticCudaBatchBuffer,
)


def _batch(tensor: torch.Tensor, *, alias: bool = True):
    return [{"nested": {"input": tensor}, "alias": tensor if alias else tensor.clone(), "control": 3}]


def test_static_batch_buffer_copies_nested_noncontiguous_aliases():
    source = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    buffer = _StaticCudaBatchBuffer(_batch(source), device=torch.device("cpu"))

    replacement = (torch.arange(6, dtype=torch.float32) + 10).reshape(3, 2).t()
    buffer.copy_from(_batch(replacement))

    static_input = buffer.batches[0]["nested"]["input"]
    assert static_input.stride() == source.stride()
    assert static_input is buffer.batches[0]["alias"]
    torch.testing.assert_close(static_input, replacement)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ([{"different": torch.ones(2, 3)}], "pytree or mapping keys changed"),
        (_batch(torch.ones(3, 2)), "tensor metadata changed"),
        (_batch(torch.ones(2, 3, dtype=torch.float64)), "tensor metadata changed"),
        (_batch(torch.ones(3, 2).t()), "tensor metadata changed"),
        (_batch(torch.ones(2, 3), alias=False), "tensor object aliasing changed"),
        (_batch(torch.ones(2, 3)) + _batch(torch.ones(2, 3)), "pytree or mapping keys changed"),
    ],
)
def test_static_batch_buffer_rejects_signature_changes(replacement, message):
    buffer = _StaticCudaBatchBuffer(_batch(torch.ones(2, 3)), device=torch.device("cpu"))
    with pytest.raises(FullIterationCudaGraphError, match=message):
        buffer.copy_from(replacement)


def test_static_batch_buffer_rejects_distinct_views_of_one_storage():
    storage = torch.arange(12)
    batches = [{"left": storage[:6], "right": storage[6:]}]
    with pytest.raises(FullIterationCudaGraphError, match="distinct tensor objects that share storage"):
        _StaticCudaBatchBuffer(batches, device=torch.device("cpu"))


class _FakeStream:
    def __init__(self):
        self.waited_for = []

    def wait_stream(self, stream):
        self.waited_for.append(stream)


class _FakeGraph:
    def __init__(self):
        self.replays = 0
        self.resets = 0
        self.registered_rng_states = []

    def replay(self):
        self.replays += 1

    def reset(self):
        self.resets += 1

    def register_generator_state(self, state):
        self.registered_rng_states.append(state)


@contextmanager
def _fake_graph_context(_graph, **kwargs):
    assert kwargs["capture_error_mode"] == "thread_local"
    assert kwargs["stream"] is not None
    assert kwargs["pool"] is not None
    yield


@pytest.fixture
def mocked_cuda_graph_runtime(monkeypatch):
    fake_current_stream = _FakeStream()
    fake_graphs = []
    fake_streams = []
    stale_override_values = []

    def make_graph():
        graph = _FakeGraph()
        fake_graphs.append(graph)
        return graph

    def make_stream():
        stream = _FakeStream()
        fake_streams.append(stream)
        return stream

    original_empty_strided = torch.empty_strided

    def cpu_empty_strided(size, stride, *, dtype, device):
        assert torch.device(device).type == "cuda"
        return original_empty_strided(size, stride, dtype=dtype, device="cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: fake_current_stream)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", make_graph)
    monkeypatch.setattr(torch.cuda, "graph", _fake_graph_context)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: object())
    monkeypatch.setattr(torch, "empty_strided", cpu_empty_strided)
    monkeypatch.setattr(
        torch.autograd.graph,
        "set_override_stale_capture_stream",
        lambda enabled: stale_override_values.append(enabled),
        raising=False,
    )
    monkeypatch.setattr(torch._C, "_override_stale_capture_stream", lambda: False, raising=False)
    monkeypatch.setattr(full_graph, "_SHARED_CAPTURE_STREAM", None)
    monkeypatch.setattr(full_graph, "_SHARED_GRAPH_POOL", None)
    return {
        "current_stream": fake_current_stream,
        "graphs": fake_graphs,
        "streams": fake_streams,
        "stale_override_values": stale_override_values,
    }


def test_manager_warms_up_captures_and_replays(mocked_cuda_graph_runtime):
    runtime = mocked_cuda_graph_runtime
    calls = []

    def forward_backward(batches):
        calls.append((batches[0]["x"].clone(), torch.autograd.is_multithreading_enabled()))
        return batches[0]["x"].sum()

    manager = FullIterationCudaGraphManager(forward_backward, warmup_iterations=1)
    warmup_result = manager([{"x": torch.tensor([1.0])}])
    capture_result = manager([{"x": torch.tensor([2.0])}])
    replay_result = manager([{"x": torch.tensor([3.0])}])

    assert warmup_result.item() == 1
    assert capture_result.item() == 2
    assert replay_result is capture_result
    assert [value.item() for value, _multithreading_enabled in calls] == [1, 2]
    assert calls[1][1] is False
    assert manager.completed_warmups == 1
    assert manager.capture_count == 1
    assert manager.replay_count == 1
    assert runtime["graphs"][0].replays == 1
    assert runtime["stale_override_values"] == [True, False]
    copy_stream, capture_stream = runtime["streams"]
    assert capture_stream.waited_for == [runtime["current_stream"]]
    assert copy_stream in runtime["current_stream"].waited_for
    assert capture_stream in runtime["current_stream"].waited_for

    manager.close()
    assert runtime["graphs"][0].resets == 1
    manager.close()


def test_manager_fails_closed_on_changed_input_and_reset_recovers(mocked_cuda_graph_runtime):
    manager = FullIterationCudaGraphManager(lambda batches: batches[0]["x"].sum(), warmup_iterations=0)
    manager([{"x": torch.ones(2)}])

    with pytest.raises(FullIterationCudaGraphError, match="tensor metadata changed"):
        manager([{"x": torch.ones(3)}])
    with pytest.raises(FullIterationCudaGraphError, match="failed closed"):
        manager([{"x": torch.ones(2)}])

    manager.reset()
    result = manager([{"x": torch.ones(3)}])
    assert result.item() == 3


def test_manager_rejects_cpu_only_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manager = FullIterationCudaGraphManager(lambda batches: batches, warmup_iterations=0)
    with pytest.raises(FullIterationCudaGraphError, match="require CUDA"):
        manager([{"x": torch.ones(1)}])


def test_graph_pool_selection(monkeypatch):
    pools = []

    def make_pool():
        pool = object()
        pools.append(pool)
        return pool

    monkeypatch.setattr(torch.cuda, "graph_pool_handle", make_pool)
    monkeypatch.setattr(full_graph, "_SHARED_GRAPH_POOL", None)

    first_local = full_graph._get_graph_pool(False)
    second_local = full_graph._get_graph_pool(False)
    first_shared = full_graph._get_graph_pool(True)
    second_shared = full_graph._get_graph_pool(True)

    assert first_local is not second_local
    assert first_shared is second_shared
    assert len(pools) == 3


def test_distributed_barriers_only_surround_capture(mocked_cuda_graph_runtime, monkeypatch):
    barriers = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append("barrier"))

    manager = FullIterationCudaGraphManager(lambda batches: batches[0]["x"].sum(), warmup_iterations=0)
    manager([{"x": torch.ones(1)}])
    assert barriers == ["barrier", "barrier"]

    manager([{"x": torch.ones(1)}])
    assert barriers == ["barrier", "barrier"]


def test_capture_registers_deduplicated_builtin_and_injected_rng_states(mocked_cuda_graph_runtime, monkeypatch):
    generator = torch.Generator()
    provider_calls = []
    monkeypatch.setattr(full_graph, "_get_transformer_engine_rng_states", lambda: {"te": generator})

    def extra_provider():
        provider_calls.append("extra")
        return (generator,)

    manager = FullIterationCudaGraphManager(
        lambda batches: batches[0]["x"].sum(),
        warmup_iterations=0,
        rng_state_providers=(extra_provider,),
    )
    manager([{"x": torch.ones(1)}])

    assert provider_calls == ["extra"]
    assert mocked_cuda_graph_runtime["graphs"][0].registered_rng_states == [generator]


def test_capture_rejects_legacy_tensor_rng_state(mocked_cuda_graph_runtime):
    manager = FullIterationCudaGraphManager(
        lambda batches: batches[0]["x"].sum(),
        warmup_iterations=0,
        rng_state_providers=(lambda: {"legacy": torch.zeros(1, dtype=torch.uint8)},),
    )

    with pytest.raises(FullIterationCudaGraphError, match="graph-safe torch.Generator"):
        manager([{"x": torch.ones(1)}])
