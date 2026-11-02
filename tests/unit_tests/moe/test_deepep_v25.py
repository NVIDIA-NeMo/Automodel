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

"""CPU autograd contracts; native transport is covered by the distributed test."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import nemo_automodel.shared.deepep_v25 as ep
from nemo_automodel.components.models.common.utils import BackendConfig


class _Event:
    def __init__(self, result):
        self.result = result

    def wait(self):
        return self.result


class _Buffer:
    def __init__(self, group, **kwargs):
        assert group.communicator_ready, "NCCL communicator must exist before EPBuffer borrows it"
        self.destroyed = False
        self.handles = []

    def capture(self):
        return object()

    def destroy(self):
        self.destroyed = True

    def dispatch(self, hidden, *, topk_idx=None, topk_weights=None, handle=None, **kwargs):
        """Emulate packed transport on CPU.

        Args:
            hidden: Tensor of shape [capacity, hidden].
            topk_idx: Integer tensor [capacity, top_k], with -1 for invalid routes.
            topk_weights: Optional tensor [capacity, top_k].
            handle: Previously recorded row/slot routing and counts.
            kwargs: Unused transport options.

        Returns:
            Packed tokens [received, hidden], None, optional weights [received],
            routing metadata, and None for the event.
        """
        if handle is None:
            rows, slots = (topk_idx >= 0).nonzero(as_tuple=True)
            # Alternate the receive order within experts, as real concurrent
            # senders can do across checkpoint forward and recomputation.
            if len(self.handles) % 2:
                rows, slots = rows.flip(0), slots.flip(0)
            order = topk_idx[rows, slots].argsort(stable=True)
            rows, slots = rows[order], slots[order]
            counts = torch.bincount(topk_idx[rows, slots], minlength=2)
            handle = SimpleNamespace(
                topk_idx=topk_idx,
                do_expand=True,
                expert_alignment=1,
                num_max_tokens_per_rank=hidden.shape[0],
                num_recv_tokens=rows.numel(),
                num_expanded_tokens=rows.numel(),
                num_recv_tokens_per_expert_list=counts.tolist(),
                psum_num_recv_tokens_per_scaleup_rank=torch.tensor([rows.numel()]),
                psum_num_recv_tokens_per_expert=counts.cumsum(0),
                num_unaligned_recv_tokens_per_expert=counts,
                recv_src_metadata=torch.stack((rows, slots), dim=1),
                dst_buffer_slot_idx=topk_idx.clone(),
                token_metadata_at_forward=None,
                channel_linked_list=None,
                num_experts=4,
                num_sms=20,
            )
            self.handles.append(handle)
        probs = (
            topk_weights[handle.recv_src_metadata[:, 0], handle.recv_src_metadata[:, 1]]
            if topk_weights is not None
            else None
        )
        result = hidden[handle.recv_src_metadata[:, 0]], None, probs, handle
        return _Event(result) if kwargs.get("defer_epilogue", False) else (*result, None)

    def combine(self, hidden, handle, *, topk_weights=None, **kwargs):
        """Invert the routing map for activations and optional weight gradients.

        Args:
            hidden: Tensor [received, hidden].
            handle: Row/slot routing metadata retained from dispatch.
            topk_weights: Optional tensor [received].
            kwargs: Unused transport options.

        Returns:
            Combined tokens [capacity, hidden], optional weights [capacity, top_k],
            and None for the event.
        """
        result = hidden.new_zeros(handle.topk_idx.shape[0], hidden.shape[1])
        result.index_add_(0, handle.recv_src_metadata[:, 0], hidden)
        probs = None
        if topk_weights is not None:
            probs = topk_weights.new_zeros(handle.topk_idx.shape)
            probs[handle.recv_src_metadata[:, 0], handle.recv_src_metadata[:, 1]] = topk_weights
        return _Event((result, probs)) if kwargs.get("defer_epilogue", False) else (result, probs, None)


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ep, "_runtime", ep._Runtime())
    monkeypatch.setattr(ep, "safe_import_from", lambda *args: (True, _Buffer))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(ep.dist, "get_backend", lambda group: "nccl")
    group = SimpleNamespace(size=lambda: 2, communicator_ready=False)

    def initialize_communicator(group, device_ids):
        assert device_ids == [0]
        group.communicator_ready = True

    monkeypatch.setattr(ep.dist, "barrier", initialize_communicator)
    config = ep.DeepEpV25Config(max_tokens_per_rank=5)
    result = config.build(group=group, num_experts=4, top_k=2, num_sms=20)
    with pytest.raises(RuntimeError, match="not initialized"):
        result.buffer
    buffer = config.init_buffer(group=group, hidden_dim=4, top_k=2)
    assert config.init_buffer(group=group, hidden_dim=4, top_k=2) is buffer
    with pytest.raises(ValueError, match="one EP group"):
        config.init_buffer(group=group, hidden_dim=8, top_k=2)
    yield result
    ep.finalize_deepep_v25()
    assert buffer.destroyed


def _experts(manager, hidden, weights, indices, async_finish):
    """Apply two scalar experts through production autograd.

    Args:
        manager: Production DeepEP manager with emulated communication.
        hidden: BF16 tensor [tokens, hidden].
        weights: FP32 tensor [tokens, top_k].
        indices: Integer tensor [tokens, top_k].
        async_finish: Exercise deferred epilogues on the emulated communication stream.

    Returns:
        Tensor [tokens, hidden] after weighted expert computation.
    """
    manager.token_indices, manager.token_probs = indices, weights
    received = manager.dispatch(hidden, async_finish=async_finish, allocate_on_comm_stream=async_finish)
    counts = manager.get_number_of_tokens_per_expert().tolist()
    first, second = received.split(counts)
    outputs = torch.cat((first * 2, second * 4))
    outputs = (outputs.float() * manager.dispatched_probs[:, None]).bfloat16()
    return manager.combine(outputs, async_finish=async_finish, allocate_on_comm_stream=async_finish)


@pytest.mark.parametrize("async_finish", [False, True])
@pytest.mark.parametrize("tokens", [0, 3, 5])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_forward_gradients_inflight_and_recompute(manager, tokens, checkpointed, masked, async_finish):
    torch.manual_seed(7)
    inputs, outputs = [], []
    for microbatch in range(2):
        hidden = torch.randn(tokens, 4, dtype=torch.bfloat16).requires_grad_()
        weights = torch.tensor([[0.25, 0.75]] * tokens).reshape(tokens, 2).requires_grad_()
        indices = torch.tensor([[1, 0]] * tokens, dtype=torch.int64).reshape(tokens, 2)
        if masked:
            indices.fill_(-1)
        output = (
            checkpoint(_experts, manager, hidden, weights, indices, async_finish, use_reentrant=False)
            if checkpointed
            else _experts(manager, hidden, weights, indices, async_finish)
        )
        expected = hidden * (0 if masked else 2.5)
        torch.testing.assert_close(output, expected, atol=0.03125, rtol=0.01)
        upstream = torch.randn_like(hidden)
        inputs.append((hidden, weights, upstream))
        outputs.append((output.float() * upstream).sum())
    sum(outputs).backward()
    for hidden, weights, upstream in inputs:
        torch.testing.assert_close(hidden.grad, upstream * (0 if masked else 2.5), atol=0.03125, rtol=0.02)
        base_grad = (hidden.float() * upstream.float()).sum(-1)
        expected_weight_grad = torch.stack((base_grad * 4, base_grad * 2), dim=1)
        if masked:
            expected_weight_grad.zero_()
        torch.testing.assert_close(weights.grad, expected_weight_grad, atol=0.03125, rtol=0.02)
    assert len({id(h) for h in manager.buffer.handles}) == len(manager.buffer.handles)
    ep.finalize_deepep_v25()
    ep.finalize_deepep_v25()


def test_backend_config_preserves_selection():
    backend = BackendConfig(dispatcher="deepep_v25", experts="torch_mm", deepep_v25={"max_tokens_per_rank": 4096})
    assert backend.dispatcher == "deepep_v25"
    assert isinstance(backend.deepep_v25, ep.DeepEpV25Config)
    assert backend.deepep_v25.max_tokens_per_rank == 4096


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="positive"):
        ep.DeepEpV25Config(max_tokens_per_rank=capacity)


def test_missing_capacity_and_async_selection():
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        BackendConfig(dispatcher="deepep_v25")
    backend = BackendConfig(
        dispatcher="deepep_v25",
        deepep_v25={"max_tokens_per_rank": 16, "use_fp8_dispatch": True},
        dispatcher_async_dispatch=True,
    )
    assert backend.dispatcher_async_dispatch
    assert backend.deepep_v25.use_fp8_dispatch


@pytest.mark.parametrize("boundary_codes", [(-352.0, -88.0, 176.0), (-320.0, -80.0, 160.0)])
def test_fp8_reference_accepts_rounding_boundaries_and_rejects_wrong_codes(boundary_codes):
    from tests.functional_tests.moe.run_deepep_v25_parity import _fp8_transport_reference

    hidden = torch.zeros(1, 128)
    hidden[0, :4] = torch.tensor([0.5390625, -0.404296875, -0.10107421875, 0.2021484375])
    hidden.requires_grad_()
    codes = torch.zeros_like(hidden)
    codes[0, :4] = torch.tensor([448.0, *boundary_codes])
    encoded = codes.to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5390625 / 448.0]])
    restored = (codes * scales).bfloat16()
    actual = _fp8_transport_reference(hidden, (encoded, scales), restored)
    torch.testing.assert_close(actual, restored.float(), rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    invalid = encoded.view(torch.uint8).clone()
    invalid[0, 1] = 0
    with pytest.raises(AssertionError, match="nearest E4M3"):
        _fp8_transport_reference(hidden, (invalid.view(torch.float8_e4m3fn), scales), restored)
