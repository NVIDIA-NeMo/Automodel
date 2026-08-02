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

"""Unit tests for DeepEP buffer teardown (``fused_a2a.free_buffer``)."""

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


def test_hybridep_compact_routing_preserves_dense_probs_gradients(monkeypatch):
    """The compact metadata path must keep the existing dense probability gradient contract."""

    class FakeHybridEPBuffer:
        def __init__(self):
            self.dispatch_kwargs = None

        def dispatch_with_permute(self, **kwargs):
            self.dispatch_kwargs = kwargs
            tokens_per_expert = torch.tensor([1, 1])
            return kwargs["hidden"], kwargs["probs"], None, tokens_per_expert, ("handle",)

        def combine_with_unpermute(self, *, hidden, probs, handle, pad_multiple, fuse_unpermute_combine=False):
            assert handle == ("handle",)
            assert pad_multiple is None
            assert not fuse_unpermute_combine
            return hidden * 2, probs * 3

    buffer = FakeHybridEPBuffer()
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", buffer)
    hidden = torch.randn(2, 4, requires_grad=True)
    topk_idx = torch.tensor([[0, 3], [1, 2]])
    dense_probs = torch.randn(2, 4, requires_grad=True)

    dispatched_hidden, dispatched_probs, _, _, _ = fused_a2a.HybridEPDispatch.apply(
        hidden,
        None,
        dense_probs,
        None,
        2,
        20,
        20,
        None,
        None,
        topk_idx,
        4,
    )
    (dispatched_hidden.sum() + dispatched_probs.sum()).backward()

    assert buffer.dispatch_kwargs["topk_idx"] is topk_idx
    assert buffer.dispatch_kwargs["routing_map"] is None
    assert buffer.dispatch_kwargs["probs"] is dense_probs
    assert buffer.dispatch_kwargs["num_of_experts"] == 4
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2))
    torch.testing.assert_close(dense_probs.grad, torch.full_like(dense_probs, 3))


def test_init_hybridep_buffer_forwards_constructor_tuning(monkeypatch):
    """AutoModel must pass HybridEP constructor knobs instead of relying on unused env vars."""
    buffer = mock.Mock()
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", buffer, raising=False)
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)

    fused_a2a.init_hybrid_ep_buffer(
        group=mock.Mock(),
        hidden_dim=4096,
        seq_len=4096,
        num_local_experts=1,
        num_sms_dispatch_api=20,
        num_sms_combine_api=20,
        fp8_dispatch=False,
        num_sms_preprocessing_api=132,
        num_blocks_permute=112,
        num_blocks_unpermute=111,
    )

    assert buffer.call_args.kwargs["num_sms_preprocessing_api"] == 132
    assert buffer.call_args.kwargs["num_blocks_permute"] == 112
    assert buffer.call_args.kwargs["num_blocks_unpermute"] == 111


def test_hybridep_dispatch_preserves_legacy_positional_order(monkeypatch):
    """Compact-routing inputs must remain optional after the legacy positional inputs."""

    class FakeHybridEPBuffer:
        def dispatch_with_permute(self, **kwargs):
            assert kwargs["topk_idx"] is None
            assert kwargs["num_of_experts"] is None
            return kwargs["hidden"], kwargs["probs"], None, torch.tensor([1, 1]), ("handle",)

        def combine_with_unpermute(self, *, hidden, probs, handle, pad_multiple, fuse_unpermute_combine=False):
            assert not fuse_unpermute_combine
            return hidden, probs

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", FakeHybridEPBuffer())
    hidden = torch.randn(2, 4, requires_grad=True)
    routing_map = torch.tensor([[True, False], [False, True]])
    probs = torch.randn(2, 2, requires_grad=True)

    dispatched_hidden, dispatched_probs, _, _, _ = fused_a2a.HybridEPDispatch.apply(
        hidden, routing_map, probs, None, 2, 20, 21, None, None
    )
    (dispatched_hidden.sum() + dispatched_probs.sum()).backward()

    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))


def test_hybridep_permute_fusion_is_used_in_forward_and_backward(monkeypatch):
    """The opt-in flag must select both fused DeepEP permutation directions."""

    class FakeHybridEPBuffer:
        def dispatch_with_permute(self, **kwargs):
            assert kwargs["fuse_permute_dispatch"]
            return kwargs["hidden"], kwargs["probs"], None, torch.tensor([1, 1]), ("handle",)

        def combine_with_unpermute(self, *, hidden, probs, handle, pad_multiple, fuse_unpermute_combine=False):
            assert handle == ("handle",)
            assert fuse_unpermute_combine
            return hidden, probs

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", FakeHybridEPBuffer())
    hidden = torch.randn(2, 4, requires_grad=True)
    topk_idx = torch.tensor([[0, 3], [1, 2]])
    probs = torch.randn(2, 4, requires_grad=True)

    dispatched_hidden, dispatched_probs, _, _, _ = fused_a2a.HybridEPDispatch.apply(
        hidden, None, probs, None, 2, 20, 21, None, None, topk_idx, 4, True
    )
    (dispatched_hidden.sum() + dispatched_probs.sum()).backward()

    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))


def test_hybridep_combine_permute_fusion_is_used_in_forward_and_backward(monkeypatch):
    """Combine autograd must pair fused unpermute forward with fused permute backward."""

    class FakeHybridEPBuffer:
        def combine_with_unpermute(self, *, hidden, handle, pad_multiple, fuse_unpermute_combine=False):
            assert handle == ("handle",)
            assert fuse_unpermute_combine
            return hidden, None

        def dispatch_with_permute(self, **kwargs):
            assert kwargs["handle"] == ("handle",)
            assert kwargs["fuse_permute_dispatch"]
            return kwargs["hidden"], None, None, torch.tensor([1, 1]), ("unused",)

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", FakeHybridEPBuffer())
    hidden = torch.randn(2, 4, requires_grad=True)

    fused_a2a.HybridEPCombine.apply(hidden, ("handle",), None, None, True).sum().backward()

    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
