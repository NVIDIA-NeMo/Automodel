# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for DeepEP buffer teardown and HybridEP partitioned-combine ordering."""

from unittest import mock

import pytest
import torch

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


@pytest.fixture(autouse=True)
def _restore_buffer():
    """Save/restore the module-global ``_buffer`` so tests don't leak state."""
    saved = fused_a2a._buffer
    try:
        yield
    finally:
        fused_a2a._buffer = saved


def test_free_buffer_destroys_and_clears():
    sentinel = mock.MagicMock()
    fused_a2a._buffer = sentinel

    fused_a2a.free_buffer()

    sentinel.destroy.assert_called_once_with()
    assert fused_a2a._buffer is None


def test_free_buffer_is_noop_when_unset():
    fused_a2a._buffer = None

    fused_a2a.free_buffer()  # must not raise

    assert fused_a2a._buffer is None


def test_free_buffer_swallows_destroy_errors():
    # A buffer created without explicitly_destroy=True raises on destroy(); free_buffer must
    # still clear the reference and not propagate the error during shutdown.
    boom = mock.MagicMock()
    boom.destroy.side_effect = RuntimeError("`explicitly_destroy` flag must be set")
    fused_a2a._buffer = boom

    fused_a2a.free_buffer()  # must not raise

    boom.destroy.assert_called_once_with()
    assert fused_a2a._buffer is None


class _RecordingHybridEPBuffer:
    """HybridEP stand-in that records the order of communication rounds."""

    def __init__(self, events):
        self.group = object()
        self.events = events

    def combine_with_unpermute(self, *, hidden, handle, pad_multiple):
        self.events.append("combine")
        return hidden + 1, None

    def dispatch_with_permute(self, *, hidden, scaling_factor, handle, pad_multiple, num_permuted_tokens):
        self.events.append("dispatch")
        return hidden * 2, None, None, None, None


def test_partitioned_combine_fences_between_rounds(monkeypatch):
    events = []
    buffer = _RecordingHybridEPBuffer(events)
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", buffer)
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_round_fences", {})

    def _all_reduce(tensor, *, group):
        assert group is buffer.group
        events.append("fence")

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    x = torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4).requires_grad_()
    route_slots = torch.tensor([0, 1, 0])

    out = fused_a2a.HybridEPCombinePartitions.apply(x, route_slots, 2, object(), 3, None)

    # Four rounds (2 slots x 2 partitions) share one handle: a fence separates
    # every pair of consecutive rounds, none before the first or after the last.
    assert events == ["combine", "fence"] * 3 + ["combine"]
    for slot in range(2):
        selected = (route_slots == slot).unsqueeze(-1)
        for partition in range(2):
            expected = torch.where(selected, x[:, partition], 0) + 1
            torch.testing.assert_close(out[:, slot * 2 + partition], expected)

    events.clear()
    out.sum().backward()

    assert events == ["dispatch", "fence"] * 3 + ["dispatch"]
    # Each row belongs to exactly one slot; the stand-in dispatch doubles it.
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


def test_reset_hybrid_ep_buffer_releases_fences(monkeypatch):
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", object())
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_round_fences", {torch.device("cpu"): torch.zeros(1)})

    fused_a2a.reset_hybrid_ep_buffer()

    assert fused_a2a._hybrid_ep_buffer is None
    assert not fused_a2a._hybrid_ep_round_fences


class _LocalRoutingBuffer:
    """CPU sum/gather adjoints for repeated routes, without simulating collectives."""

    def __init__(self):
        self.tokens = torch.tensor([0, 1, 0, 2, 1])

    def combine_with_unpermute(self, *, hidden: torch.Tensor, handle, pad_multiple):
        """Sum repeated local routes.

        Args:
            hidden: Contiguous tensor of shape [routes, hidden].
            handle: Unused routing handle.
            pad_multiple: Unused padding alignment.

        Returns:
            Tensor of shape [tokens, hidden] and None for routing probabilities.
        """
        assert hidden.is_contiguous()
        return hidden.new_zeros(3, hidden.shape[-1]).index_add(0, self.tokens, hidden), None

    def dispatch_with_permute(self, *, hidden: torch.Tensor, **kwargs):
        """Gather gradients using the adjoint of the route sum.

        Args:
            hidden: Contiguous tensor of shape [tokens, hidden].
            **kwargs: Unused routing metadata.

        Returns:
            Tensor of shape [routes, hidden] and four unused metadata values.
        """
        assert hidden.is_contiguous()
        return hidden[self.tokens], None, None, None, None


@pytest.mark.parametrize("slotted", [False, True])
def test_partitioned_combine_matches_dense_autograd(monkeypatch, slotted):
    buffer = _LocalRoutingBuffer()
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", buffer)
    # This test checks the local autograd contract; distributed fence ordering
    # is exercised separately and by the CP32 end-to-end experiment.
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_round_fence", lambda device: None)
    x = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3).requires_grad_()
    reference = x.detach().clone().requires_grad_()
    slots = torch.tensor([0, 1, 1, -1, 2]) if slotted else None
    num_slots = 2 if slotted else 1
    out = fused_a2a.HybridEPCombinePartitions.apply(x, slots, num_slots, object(), 5, None)

    valid = torch.ones(5, 1, dtype=torch.bool) if slots is None else slots[:, None] == torch.arange(num_slots)
    dense = reference[:, None] * valid[:, :, None, None]
    expected = reference.new_zeros(3, num_slots, 2, 3).index_add(0, buffer.tokens, dense).flatten(1, 2)
    torch.testing.assert_close(out, expected)
    # Noncontiguous, nonuniform gradients exercise partition views in backward.
    gradient = torch.arange(out.numel(), dtype=out.dtype).reshape(3, 3, -1).transpose(1, 2)
    out.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, reference.grad)
