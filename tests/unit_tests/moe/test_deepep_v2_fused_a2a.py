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

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.moe.megatron import fused_a2a


class _FakeGroup:
    def __init__(self, size: int = 1):
        self._size = size

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return 0


class _FakeEvent:
    def current_stream_wait(self) -> None:
        pass


class _FakeHandle:
    def __init__(self, topk_idx: torch.Tensor, num_max_tokens_per_rank: int | None = None):
        self.num_max_tokens_per_rank = (
            num_max_tokens_per_rank if num_max_tokens_per_rank is not None else topk_idx.size(0)
        )
        self.num_recv_tokens_per_expert_list = [2, 0]
        self.num_sms = 7
        self.topk_idx = topk_idx
        self.psum_num_recv_tokens_per_scaleup_rank = torch.tensor([topk_idx.size(0)], dtype=torch.int32)
        self.psum_num_recv_tokens_per_expert = torch.tensor([2, 2], dtype=torch.int32)
        self.recv_src_metadata = topk_idx.clone()
        self.dst_buffer_slot_idx = topk_idx.to(torch.int32).clone()
        self.token_metadata_at_forward = None
        self.channel_linked_list = None


class _FakeElasticBuffer:
    def __init__(self, _group, **kwargs):
        self.constructed_with_num_bytes = "num_bytes" in kwargs
        self.allow_hybrid_mode = kwargs.get("allow_hybrid_mode", True)
        self.allow_multiple_reduction = kwargs.get("allow_multiple_reduction", True)
        self.prefer_overlap_with_compute = kwargs.get("prefer_overlap_with_compute", True)
        self.num_gpu_timeout_secs = kwargs.get("num_gpu_timeout_secs")
        self.destroyed = False
        self.dispatch_handles = []
        self.combine_handles = []
        self.dispatch_kwargs = []
        self.combine_kwargs = []

    def dispatch(self, x, topk_idx=None, topk_weights=None, handle=None, **kwargs):
        self.dispatch_kwargs.append(kwargs)
        if handle is None:
            handle = _FakeHandle(topk_idx, kwargs["num_max_tokens_per_rank"])
            self.dispatch_handles.append(handle)
            return x + 1, topk_idx, topk_weights, handle, _FakeEvent()
        self.dispatch_handles.append(handle)
        return x + 2, None, None, handle, _FakeEvent()

    def combine(self, x, handle, topk_weights=None, **kwargs):
        self.combine_kwargs.append(kwargs)
        self.combine_handles.append(handle)
        return x + 3, topk_weights, _FakeEvent()

    def destroy(self) -> None:
        self.destroyed = True


def _reset_fake_state(monkeypatch) -> None:
    monkeypatch.setattr(fused_a2a, "ElasticBuffer", _FakeElasticBuffer)
    monkeypatch.setattr(fused_a2a, "_warmup_deepep_v2_group", lambda _group: None)
    monkeypatch.setattr(fused_a2a, "_deepep_v2_buffer", None)


def _init_test_deepep_v2_buffer(*args, **kwargs):
    fused_a2a.init_deepep_v2_buffer(*args, **kwargs)
    return fused_a2a._deepep_v2_buffer


def test_legacy_buffer_resolver_prefers_latest_legacy_module(monkeypatch):
    legacy_buffer = object()
    root_buffer = object()

    def import_module(module_name):
        if module_name == "deep_ep.legacy":
            raise ImportError(module_name)
        if module_name == "deep_ep.buffers.legacy":
            return SimpleNamespace(Buffer=legacy_buffer)
        if module_name == "deep_ep":
            return SimpleNamespace(Buffer=root_buffer)
        raise AssertionError(module_name)

    monkeypatch.setattr(fused_a2a.importlib, "import_module", import_module)

    available, buffer = fused_a2a._safe_import_first_symbol(
        ("deep_ep.legacy", "deep_ep.buffers.legacy", "deep_ep"),
        "Buffer",
    )

    assert available is True
    assert buffer is legacy_buffer


def test_legacy_buffer_resolver_falls_back_to_previous_root_buffer(monkeypatch):
    root_buffer = object()

    def import_module(module_name):
        if module_name == "deep_ep":
            return SimpleNamespace(Buffer=root_buffer)
        raise ImportError(module_name)

    monkeypatch.setattr(fused_a2a.importlib, "import_module", import_module)

    available, buffer = fused_a2a._safe_import_first_symbol(
        ("deep_ep.legacy", "deep_ep.buffers.legacy", "deep_ep"),
        "Buffer",
    )

    assert available is True
    assert buffer is root_buffer


def test_deepep_v2_num_sms_does_not_mutate_legacy_buffer(monkeypatch):
    legacy_calls = []
    monkeypatch.setattr(fused_a2a, "HAVE_DEEP_EP", True)
    monkeypatch.setattr(fused_a2a, "Buffer", SimpleNamespace(set_num_sms=legacy_calls.append))
    monkeypatch.setattr(fused_a2a, "_deepep_v2_num_sms", 0)

    fused_a2a.set_deepep_v2_num_sms(20)

    assert fused_a2a._deepep_v2_num_sms == 20
    assert legacy_calls == []

    fused_a2a.set_deepep_num_sms(17)

    assert legacy_calls == [17]
    assert fused_a2a._deepep_v2_num_sms == 20


def test_deepep_v2_num_qps_is_configurable(monkeypatch):
    monkeypatch.setattr(fused_a2a, "_deepep_v2_num_qps", 0)

    fused_a2a.set_deepep_v2_num_qps(65)

    assert fused_a2a._deepep_v2_num_qps == 65


def test_deepep_v2_dispatch_uses_legacy_return_contract(monkeypatch):
    _reset_fake_state(monkeypatch)

    x = torch.zeros(5, 256)
    token_indices = torch.tensor([[0], [1], [0], [1], [0]], dtype=torch.int64)
    token_probs = torch.ones(5, 1)

    recv_x, recv_indices, recv_probs, tokens_per_expert, handle = fused_a2a.DeepEPV2FusedDispatch.apply(
        x,
        token_indices,
        token_probs,
        2,
        _FakeGroup(),
        False,
        False,
    )

    assert torch.equal(recv_x, x + 1)
    assert torch.equal(recv_indices, token_indices)
    assert torch.equal(recv_probs, token_probs)
    assert torch.equal(tokens_per_expert, torch.tensor([2, 0], dtype=torch.int64))
    assert torch.equal(handle.topk_idx, token_indices)
    assert handle.num_max_tokens_per_rank == 5


def test_deepep_v2_handle_restore_uses_saved_tensor_fields():
    class _FakeCtx:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    original_handle = _FakeHandle(torch.tensor([[0]], dtype=torch.int64))
    recomputed_handle = _FakeHandle(torch.tensor([[1]], dtype=torch.int64))
    ctx = _FakeCtx()

    fused_a2a._save_deepep_v2_handle(ctx, original_handle)
    ctx.saved_tensors = tuple(getattr(recomputed_handle, field) for field in ctx.handle_tensor_fields)
    restored_handle = fused_a2a._restore_deepep_v2_handle(ctx)

    assert restored_handle is original_handle
    assert torch.equal(restored_handle.topk_idx, recomputed_handle.topk_idx)
    assert torch.equal(restored_handle.recv_src_metadata, recomputed_handle.recv_src_metadata)


def test_deepep_v2_handle_tensors_track_nonreentrant_recompute():
    handles = []
    backward_handles = []

    class _CheckpointedFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            handle = _FakeHandle(torch.tensor([[len(handles)]], dtype=torch.int64))
            handles.append(handle)
            fused_a2a._save_deepep_v2_handle(ctx, handle)
            return x * 2

        @staticmethod
        def backward(ctx, grad_output):
            backward_handles.append(fused_a2a._restore_deepep_v2_handle(ctx))
            return grad_output * 2

    x = torch.ones(1, requires_grad=True)
    checkpoint(_CheckpointedFunction.apply, x, use_reentrant=False).backward()

    assert len(handles) == 2
    assert backward_handles == [handles[0]]
    assert torch.equal(backward_handles[0].topk_idx, handles[-1].topk_idx)


def test_deepep_v2_buffer_validates_hidden_alignment(monkeypatch):
    _reset_fake_state(monkeypatch)
    group = _FakeGroup()

    hybrid_buffer = _init_test_deepep_v2_buffer(group, num_max_tokens_per_rank=4, hidden=2560, num_topk=2)

    assert hybrid_buffer.allow_hybrid_mode is True
    with pytest.raises(ValueError, match="hidden dimension divisible by 256"):
        _init_test_deepep_v2_buffer(group, num_max_tokens_per_rank=4, hidden=2880, num_topk=2)


def test_deepep_v2_buffer_initialization_is_idempotent(monkeypatch):
    _reset_fake_state(monkeypatch)
    group = _FakeGroup()

    buffer = _init_test_deepep_v2_buffer(group, num_max_tokens_per_rank=4096, hidden=256, num_topk=2)
    same_buffer = _init_test_deepep_v2_buffer(group, num_max_tokens_per_rank=4096, hidden=256, num_topk=2)

    assert same_buffer is buffer


def test_deepep_v2_dispatch_reuses_process_global_buffer(monkeypatch):
    _reset_fake_state(monkeypatch)
    group = _FakeGroup()
    token_indices = torch.tensor([[0], [1]], dtype=torch.int64)
    token_probs = torch.ones(2, 1)
    init_order = []

    class _OrderedFakeElasticBuffer(_FakeElasticBuffer):
        def __init__(self, buffer_group, **kwargs):
            init_order.append(("buffer", buffer_group))
            super().__init__(buffer_group, **kwargs)

    monkeypatch.setattr(fused_a2a, "ElasticBuffer", _OrderedFakeElasticBuffer)
    monkeypatch.setattr(
        fused_a2a,
        "_warmup_deepep_v2_group",
        lambda warmup_group: init_order.append(("warmup", warmup_group)),
    )

    fused_a2a.DeepEPV2FusedDispatch.apply(torch.zeros(2, 256), token_indices, token_probs, 2, group, False, False)
    buffer = fused_a2a._deepep_v2_buffer
    fused_a2a.DeepEPV2FusedDispatch.apply(torch.zeros(2, 256), token_indices, token_probs, 2, group, False, False)

    assert init_order == [("warmup", group), ("buffer", group)]
    assert fused_a2a._deepep_v2_buffer is buffer
    assert len(buffer.dispatch_handles) == 2
    assert buffer.constructed_with_num_bytes is False
    assert buffer.allow_multiple_reduction is True
    assert buffer.prefer_overlap_with_compute is True
    assert buffer.num_gpu_timeout_secs == 300


def test_deepep_v2_combine_uses_process_global_buffer(monkeypatch):
    _reset_fake_state(monkeypatch)
    group = _FakeGroup()
    token_indices = torch.tensor([[0], [1]], dtype=torch.int64)
    token_probs = torch.ones(2, 1)

    _, _, _, _, handle = fused_a2a.DeepEPV2FusedDispatch.apply(
        torch.zeros(2, 256),
        token_indices,
        token_probs,
        2,
        group,
        False,
        False,
    )
    buffer = fused_a2a._deepep_v2_buffer

    combined_x, _ = fused_a2a.DeepEPV2FusedCombine.apply(
        torch.zeros(2, 256),
        group,
        handle,
        False,
        False,
    )

    assert torch.equal(combined_x, torch.full((2, 256), 3.0))
    assert buffer.combine_handles == [handle]


def test_deepep_v2_uses_tuned_qp_count_in_forward_and_backward(monkeypatch):
    _reset_fake_state(monkeypatch)
    monkeypatch.setattr(fused_a2a, "_deepep_v2_num_qps", 65)
    group = _FakeGroup()
    x = torch.zeros(2, 256, requires_grad=True)
    token_indices = torch.tensor([[0], [1]], dtype=torch.int64)
    token_probs = torch.ones(2, 1)

    recv_x, _, _, _, handle = fused_a2a.DeepEPV2FusedDispatch.apply(
        x,
        token_indices,
        token_probs,
        2,
        group,
        False,
        False,
    )
    combined_x, _ = fused_a2a.DeepEPV2FusedCombine.apply(recv_x, group, handle, False, False)
    combined_x.sum().backward()

    buffer = fused_a2a._deepep_v2_buffer
    assert [kwargs["num_qps"] for kwargs in buffer.dispatch_kwargs] == [65, 65]
    assert [kwargs["num_qps"] for kwargs in buffer.combine_kwargs] == [65, 65]


def test_destroy_deepep_v2_buffer_clears_process_global_buffer(monkeypatch):
    _reset_fake_state(monkeypatch)
    group = _FakeGroup()

    buffer = _init_test_deepep_v2_buffer(group, num_max_tokens_per_rank=4, hidden=256, num_topk=2)

    fused_a2a.destroy_deepep_v2_buffer()

    assert buffer.destroyed is True
    assert fused_a2a._deepep_v2_buffer is None
