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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import nemo_automodel.components.moe.paged_stash as paged_stash
from nemo_automodel.components.moe.experts import GroupedExpertsTE, GroupedExpertsTeOps
from nemo_automodel.components.moe.paged_stash_ops import HAVE_TRITON


class _SaveForBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return tensor * 2

    @staticmethod
    def backward(ctx, grad_output):
        (tensor,) = ctx.saved_tensors
        return grad_output + tensor * 0


class _PrefixSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, num_tokens):
        ctx.save_for_backward(tensor, num_tokens)
        return tensor.square()

    @staticmethod
    def backward(ctx, grad_output):
        tensor, num_tokens = ctx.saved_tensors
        row_mask = torch.arange(tensor.shape[0], device=tensor.device) < num_tokens
        row_mask = row_mask.reshape(-1, *([1] * (tensor.dim() - 1)))
        grad = torch.where(row_mask, 2 * tensor * grad_output, 0)
        return grad, None


def _record_group(manager, tensor, num_tokens, *, max_num_tokens, name="test"):
    tensor.grouped_tensor_scale_inv = False
    group = manager.group(
        name=name,
        max_num_tokens=max_num_tokens,
        num_tokens_tensor=num_tokens,
    )
    tensor = group.start(tensor)
    with group:
        output = _SaveForBackward.apply(tensor)
    return group.commit(output)


def test_recording_profiles_peak_page_charges_and_releases_after_backward():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=4, buffer_size_factor=1.2)

    first = torch.arange(32.0).reshape(8, 4).detach().requires_grad_()
    second_input = _record_group(manager, first, torch.tensor(5), max_num_tokens=8, name="first")
    second_input.grouped_tensor_scale_inv = False
    second = _record_group(manager, second_input, torch.tensor(3), max_num_tokens=8, name="second")
    second.sum().backward()

    diagnostics = manager.diagnostics()
    # The two simultaneously live tensors use ceil(5/4) + ceil(3/4) pages.
    assert diagnostics["recorded_peak_tokens"] == {(torch.float32, 4): 12}
    assert diagnostics["live_groups"] == 0
    with pytest.raises(RuntimeError, match="CUDA warmup"):
        manager.prepare()
    manager.close()


def test_recording_understands_te_columnwise_scale_inverse_rows():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    tensor = torch.arange(6.0).reshape(2, 3).detach().requires_grad_()
    tensor.grouped_tensor_scale_inv = True
    group = manager.group(
        name="columnwise_scale",
        max_num_tokens=64,
        num_tokens_tensor=torch.tensor(32),
    )
    with group:
        output = _SaveForBackward.apply(tensor)
    output = group.commit(output)
    output.sum().backward()

    # One live scale row still owns one complete page in the active allocator.
    assert manager.diagnostics()["recorded_peak_tokens"] == {(torch.float32, 3): 64}
    manager.close()


def test_disabled_context_bypasses_unvalidated_nested_hook_regions():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    tensor = torch.ones(8, 4, requires_grad=True)

    with manager.disabled():
        output = _record_group(manager, tensor, torch.tensor(5), max_num_tokens=8)
        output.sum().backward()

    assert manager.diagnostics()["recorded_peak_tokens"] == {}
    manager.close()


def test_groups_fail_closed_when_nested():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    outer = manager.group(name="outer", max_num_tokens=8, num_tokens_tensor=torch.tensor(4))

    with pytest.raises(RuntimeError, match="cannot nest"):
        with outer:
            manager.group(name="inner", max_num_tokens=8, num_tokens_tensor=torch.tensor(4))

    with pytest.raises(RuntimeError, match="recording was invalid"):
        manager.prepare()
    manager.close()


def test_stream_overlap_no_marker_group_cleans_boundary_state_and_close_disables_eval():
    manager = paged_stash.PagedStashManager()
    manager._state = paged_stash._PagedStashState.ACTIVE
    manager._full_iteration_stream_overlap = True
    tensor = torch.randn(4, requires_grad=True)
    group = manager.group(name="no_markers", max_num_tokens=4, num_tokens_tensor=torch.tensor(4))
    tensor_for_compute = group.start(tensor)
    with group:
        output = tensor_for_compute.square()
    group.commit(output).sum().backward()

    assert manager._groups_without_stash == set()
    assert manager._backward_in_progress_group_id is None
    manager.close()

    with torch.no_grad():
        eval_group = manager.group(name="eval", max_num_tokens=4, num_tokens_tensor=torch.tensor(4))
        eval_input = eval_group.start(tensor.detach())
        with eval_group:
            eval_output = eval_input + 1
        assert eval_group.commit(eval_output) is eval_output


def test_stream_overlap_anchor_finishes_backward_with_frozen_expert_input():
    manager = paged_stash.PagedStashManager()
    manager._state = paged_stash._PagedStashState.RECORDING
    manager._full_iteration_stream_overlap = True
    frozen_input = torch.randn(4)
    expert_weight = torch.nn.Parameter(torch.tensor(2.0))

    recording_group = manager.group(
        name="frozen_input_recording",
        max_num_tokens=4,
        num_tokens_tensor=torch.tensor(4),
    )
    recording_input = recording_group.start(frozen_input)
    assert recording_input.requires_grad
    with recording_group:
        recording_output = recording_input * expert_weight
    recording_group.commit(recording_output).sum().backward()
    assert manager._backward_in_progress_group_id is None

    expert_weight.grad = None
    manager._state = paged_stash._PagedStashState.ACTIVE
    group = manager.group(name="frozen_input", max_num_tokens=4, num_tokens_tensor=torch.tensor(4))
    input_for_compute = group.start(frozen_input)
    assert input_for_compute.requires_grad
    with group:
        output = input_for_compute * expert_weight
    group.commit(output).sum().backward()

    torch.testing.assert_close(expert_weight.grad, frozen_input.sum())
    assert manager._backward_anchor is not None
    assert manager._backward_anchor.grad is None
    assert manager._groups_without_stash == set()
    assert manager._backward_in_progress_group_id is None
    manager.close()


def test_disabled_stash_composes_with_nonreentrant_checkpoint_recomputation():
    manager = paged_stash.PagedStashManager()
    calls = []

    def checkpointed(input_):
        calls.append("forward")
        group = manager.group(name="checkpoint", max_num_tokens=4, num_tokens_tensor=torch.tensor(4))
        input_ = group.start(input_)
        with group:
            output = input_.sin().square()
        return group.commit(output)

    tensor = torch.randn(4, requires_grad=True)
    checkpoint(
        checkpointed,
        tensor,
        use_reentrant=False,
        preserve_rng_state=True,
    ).sum().backward()

    assert calls == ["forward", "forward"]
    assert tensor.grad is not None
    assert manager.diagnostics()["state"] == "disabled"


def test_configuration_is_idempotent_and_can_restart_recording():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=32, buffer_size_factor=1.25)
    manager.configure(enabled=True, page_size=32, buffer_size_factor=1.25)
    with pytest.raises(RuntimeError, match="same page_size"):
        manager.configure(enabled=True, page_size=64, buffer_size_factor=1.25)

    manager.configure_full_iteration_stream_overlap(enabled=True)
    manager.restart_recording()
    diagnostics = manager.diagnostics()
    assert diagnostics["state"] == "recording"
    assert diagnostics["page_size"] == 32
    assert diagnostics["buffer_size_factor"] == 1.25
    assert diagnostics["full_iteration_stream_overlap"] is True
    manager.close()


class _FakeCudaStream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_stream(self, other):
        self.events.append((self.name, "wait", other.name))


class _FakePagedTensor:
    dtype = torch.float32
    hidden_size = 4

    def __init__(self, events, current_stream):
        self.events = events
        self.current_stream = current_stream
        self._tensor = object()
        self._original_tensor = None

    def offload(self, buffer):
        self.events.append(("pack", self.current_stream().name, buffer.name))
        self._original_tensor = self._tensor
        self._tensor = None

    def release_original(self):
        self.events.append(("release", self.current_stream().name))
        self._original_tensor = None

    def allocate_for_reload(self):
        assert self._tensor is None
        self.events.append(("allocate", self.current_stream().name))
        self._tensor = object()

    def reload(self, buffer):
        assert self._tensor is not None
        self.events.append(("reload", self.current_stream().name, buffer.name))


def _stream_overlap_manager(monkeypatch):
    events = []
    main_stream = _FakeCudaStream("main", events)
    transfer_stream = _FakeCudaStream("transfer", events)
    current_stream = [main_stream]

    @contextmanager
    def use_stream(stream):
        previous = current_stream[0]
        current_stream[0] = stream
        try:
            yield
        finally:
            current_stream[0] = previous

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream[0])
    monkeypatch.setattr(torch.cuda, "stream", use_stream)

    manager = paged_stash.PagedStashManager()
    manager._state = paged_stash._PagedStashState.ACTIVE
    manager._full_iteration_stream_overlap = True
    manager._transfer_stream = transfer_stream
    manager._buffers[torch.float32, 4] = SimpleNamespace(name="pages", total_tokens=8)
    tensor = _FakePagedTensor(events, lambda: current_stream[0])
    manager._group_tensors[7] = [tensor]
    return manager, tensor, events


def test_stream_overlap_pack_rejoins_at_next_expert_boundary(monkeypatch):
    manager, tensor, events = _stream_overlap_manager(monkeypatch)

    manager._stash_group(7)

    assert events == [
        ("transfer", "wait", "main"),
        ("pack", "transfer", "pages"),
    ]
    assert tensor._original_tensor is not None
    assert manager.diagnostics()["transfer_stream_status"] == "packing"

    manager._wait_for_stash_to_complete()

    assert events[-2:] == [
        ("main", "wait", "transfer"),
        ("release", "main"),
    ]
    assert tensor._original_tensor is None
    assert manager.diagnostics()["transfer_stream_status"] == "idle"


def test_stream_overlap_reload_chains_after_pack_and_rejoins_before_unpack(monkeypatch):
    manager, tensor, events = _stream_overlap_manager(monkeypatch)
    manager._stash_group(7)

    # No schedule lookahead is available, so reload starts at this group's
    # backward boundary. It still chains after pack on the same stream and the
    # main stream waits before saved-tensor unpack can observe the destination.
    manager._reload_group(7)

    assert events == [
        ("transfer", "wait", "main"),
        ("pack", "transfer", "pages"),
        ("allocate", "main"),
        ("transfer", "wait", "main"),
        ("reload", "transfer", "pages"),
        ("main", "wait", "transfer"),
        ("release", "main"),
    ]
    assert tensor._tensor is not None
    assert tensor._original_tensor is None
    assert manager.diagnostics()["transfer_stream_status"] == "idle"
    assert manager.diagnostics()["live_groups"] == 0


def test_stream_overlap_prefetches_next_lifo_group_after_expert_backward(monkeypatch):
    manager, tensor7, events = _stream_overlap_manager(monkeypatch)
    tensor6 = _FakePagedTensor(events, tensor7.current_stream)
    manager._group_tensors[6] = [tensor6]

    manager._stash_group(6)
    manager._wait_for_stash_to_complete()
    manager._stash_group(7)
    manager._begin_group_backward(7)
    assert manager._backward_in_progress_group_id == 7

    event_count = len(events)
    manager._finish_group_backward(7)

    # The next LIFO group's destination and reload are enqueued, but main does
    # not wait yet: reload overlaps intervening non-expert backward work.
    assert events[event_count:] == [
        ("allocate", "main"),
        ("transfer", "wait", "main"),
        ("reload", "transfer", "pages"),
    ]
    assert manager.diagnostics()["transfer_stream_status"] == "reloading"
    assert manager.diagnostics()["prefetched_group_id"] == 6

    manager._begin_group_backward(6)
    assert events[-1] == ("main", "wait", "transfer")
    assert manager._backward_in_progress_group_id == 6
    manager._finish_group_backward(6)

    diagnostics = manager.diagnostics()
    assert diagnostics["transfer_stream_status"] == "idle"
    assert diagnostics["prefetched_group_id"] is None
    assert diagnostics["backward_schedule_depth"] == 0
    manager.close()
    assert manager.diagnostics()["state"] == "disabled"


def test_stream_overlap_fails_closed_on_non_lifo_backward(monkeypatch):
    manager, _tensor, _events = _stream_overlap_manager(monkeypatch)
    manager._backward_group_stack[:] = [6, 7]

    with pytest.raises(RuntimeError, match="backward order is not LIFO"):
        manager._begin_group_backward(6)


def test_stream_overlap_lifecycle_rejects_reset_with_pack_in_flight(monkeypatch):
    manager, _tensor, _events = _stream_overlap_manager(monkeypatch)
    manager._stash_group(7)
    manager._group_tensors.clear()

    with pytest.raises(RuntimeError, match="transfer is pending"):
        manager.reset_after_overflow()
    with pytest.raises(RuntimeError, match="transfer is pending"):
        manager.close()


def test_stream_overlap_cannot_be_disabled_during_expert_backward(monkeypatch):
    manager, _tensor, _events = _stream_overlap_manager(monkeypatch)
    manager._stash_group(7)
    manager._begin_group_backward(7)

    with pytest.raises(RuntimeError, match="backward schedule is live"):
        manager.configure_full_iteration_stream_overlap(enabled=False)

    manager._finish_group_backward(7)
    manager.close()


def test_force_abort_after_error_synchronizes_and_clears_live_state(monkeypatch):
    manager, tensor, _events = _stream_overlap_manager(monkeypatch)
    manager._record_device = torch.device("cuda", 0)
    manager._stash_group(7)
    synchronizations = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synchronizations.append(device))

    manager.force_abort_after_error()

    diagnostics = manager.diagnostics()
    assert synchronizations == [torch.device("cuda", 0)]
    assert tensor._original_tensor is None
    assert diagnostics["state"] == "disabled"
    assert diagnostics["live_groups"] == 0
    assert diagnostics["transfer_stream_status"] == "idle"
    assert diagnostics["backward_schedule_depth"] == 0


def test_device_overflow_api_does_not_require_host_read():
    manager = paged_stash.PagedStashManager()
    assert manager.check_overflow() is None
    manager._overflow = torch.ones(1, dtype=torch.int64)
    assert manager.check_overflow() is manager._overflow
    with pytest.raises(paged_stash.PagedStashOverflowError):
        manager.finish_iteration()
    manager._overflow = None


def test_te_ops_reads_finalized_backend_paged_stash_names(monkeypatch):
    manager = MagicMock()
    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: manager)

    def fake_te_init(self, *args, **kwargs):
        self._te_ops_uses_padded_capacity = True

    monkeypatch.setattr(GroupedExpertsTE, "__init__", fake_te_init)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=None,
        moe_paged_stash=True,
        moe_paged_stash_page_size=128,
        moe_paged_stash_buffer_size_factor_cuda=1.3,
        moe_paged_stash_buffer_size_factor_cpu=0.0,
        partial_cuda_graph_experts=False,
    )

    experts = GroupedExpertsTeOps(MagicMock(), backend=backend)

    assert experts.moe_paged_stash is True
    manager.configure.assert_called_once_with(enabled=True, page_size=128, buffer_size_factor=1.3)


def test_te_ops_paged_stash_requires_full_mxfp8_fusion(monkeypatch):
    monkeypatch.setattr(GroupedExpertsTE, "__init__", lambda self, *args, **kwargs: None)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=SimpleNamespace(recipe="mxfp8"),
        moe_paged_stash=True,
        moe_paged_stash_page_size=64,
        moe_paged_stash_buffer_size_factor_cuda=1.1,
        moe_paged_stash_buffer_size_factor_cpu=0.0,
        partial_cuda_graph_experts=False,
    )

    with pytest.raises(RuntimeError, match="requires the TE CuTe DSL fused MXFP8"):
        GroupedExpertsTeOps(MagicMock(), backend=backend)


def test_recording_rejects_misaligned_mxfp8_expert_splits():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)

    with pytest.raises(RuntimeError, match="every expert split"):
        manager.group(
            name="misaligned",
            max_num_tokens=256,
            num_tokens_tensor=torch.tensor(256),
            tokens_per_expert=torch.tensor([128, 127, 1]),
        )

    manager.close()


def test_te_ops_paged_stash_fails_closed_on_unimplemented_host_spill(monkeypatch):
    monkeypatch.setattr(GroupedExpertsTE, "__init__", lambda self, *args, **kwargs: None)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=None,
        moe_paged_stash=True,
        moe_paged_stash_buffer_size_factor_cpu=0.5,
        partial_cuda_graph_experts=False,
    )

    with pytest.raises(ValueError, match="does not yet support pinned-host spill"):
        GroupedExpertsTeOps(MagicMock(), backend=backend)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA and Triton are required")
def test_cuda_stash_restores_dynamic_live_prefix_for_backward():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=2, buffer_size_factor=1.5)
    manager.configure_full_iteration_stream_overlap(enabled=True)

    warmup = torch.randn(8, 4, device="cuda", requires_grad=True)
    warmup_tokens = torch.tensor(5, device="cuda", dtype=torch.int64)
    warmup.grouped_tensor_scale_inv = False
    group = manager.group(name="warmup", max_num_tokens=8, num_tokens_tensor=warmup_tokens)
    warmup = group.start(warmup)
    with group:
        warmup_output = _PrefixSquare.apply(warmup, warmup_tokens)
    warmup_output = group.commit(warmup_output)
    warmup_output.sum().backward()
    manager.prepare()

    tensor = torch.randn(8, 4, device="cuda", requires_grad=True)
    num_tokens = torch.tensor(4, device="cuda", dtype=torch.int64)
    tensor.grouped_tensor_scale_inv = False
    group = manager.group(name="active", max_num_tokens=8, num_tokens_tensor=num_tokens)
    tensor_for_compute = group.start(tensor)
    with group:
        output = _PrefixSquare.apply(tensor_for_compute, num_tokens)
    output = group.commit(output)
    output.sum().backward()

    expected = torch.cat((2 * tensor.detach()[:4], torch.zeros_like(tensor.detach()[4:])))
    torch.testing.assert_close(tensor.grad, expected)
    assert manager.check_overflow() is not None
    assert manager.check_overflow().item() == 0
    manager.finish_iteration()
    manager.close()


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA and Triton are required")
def test_cuda_stream_overlap_survives_capture_dynamic_replay_and_allocator_pressure():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=2, buffer_size_factor=1.5)
    manager.configure_full_iteration_stream_overlap(enabled=True)

    def run_iteration(input_, token_counts):
        hidden = input_
        for layer, num_tokens in enumerate(token_counts):
            hidden.grouped_tensor_scale_inv = False
            group = manager.group(
                name=f"layer_{layer}",
                max_num_tokens=8,
                num_tokens_tensor=num_tokens,
            )
            hidden_for_compute = group.start(hidden)
            with group:
                hidden = _PrefixSquare.apply(hidden_for_compute, num_tokens)
            hidden = group.commit(hidden)
        hidden.sum().backward()
        return hidden

    warmup_input = torch.randn(8, 4, device="cuda", requires_grad=True)
    warmup_counts = [
        torch.tensor(5, device="cuda", dtype=torch.int64),
        torch.tensor(4, device="cuda", dtype=torch.int64),
    ]
    run_iteration(warmup_input, warmup_counts)
    manager.prepare()

    static_input = torch.randn(8, 4, device="cuda", requires_grad=True)
    static_counts = [
        torch.tensor(4, device="cuda", dtype=torch.int64),
        torch.tensor(3, device="cuda", dtype=torch.int64),
    ]
    # Compile Triton kernels, initialize the side-stream protocol, and retain a
    # persistent input-grad allocation before capture.
    run_iteration(static_input, static_counts)
    static_input.grad.zero_()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            pool=torch.cuda.graph_pool_handle(),
            capture_error_mode="thread_local",
        ):
            run_iteration(static_input, static_counts)

        diagnostics = manager.diagnostics()
        assert diagnostics["transfer_stream_status"] == "idle"
        assert diagnostics["backward_schedule_depth"] == 0
        assert diagnostics["prefetched_group_id"] is None

        for replay, (first_count, second_count) in enumerate(((4, 3), (2, 5))):
            replay_input = torch.randn_like(static_input)
            with torch.no_grad():
                static_input.copy_(replay_input)
                static_counts[0].fill_(first_count)
                static_counts[1].fill_(second_count)
                static_input.grad.zero_()
            graph.replay()
            torch.cuda.synchronize()

            live_rows = min(first_count, second_count)
            expected = torch.zeros_like(replay_input)
            expected[:live_rows] = 4 * replay_input[:live_rows].pow(3)
            torch.testing.assert_close(static_input.grad, expected)

            if replay == 0:
                pressure = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
                pressure.zero_()
                del pressure
                torch.cuda.empty_cache()

        graph.reset()
        manager.close()
    finally:
        # Keep cleanup best-effort so an assertion exposes its own failure.
        try:
            torch.cuda.synchronize()
            graph.reset()
        except Exception:
            pass
        if manager.is_enabled:
            manager.force_abort_after_error()
