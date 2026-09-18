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

"""Unit tests for fused DeepEP and HybridEP dispatch helpers."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.utils.checkpoint import CheckpointError, CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


@pytest.fixture(autouse=True)
def _restore_buffer():
    """Save/restore module-global dispatch state so tests don't leak state."""
    saved = fused_a2a._buffer
    saved_spec = fused_a2a._buffer_spec
    saved_hybridep = fused_a2a._hybrid_ep_buffer
    saved_recorder = fused_a2a._hybridep_dispatch_replay_state.recorder
    saved_mode = fused_a2a._hybridep_dispatch_replay_state.mode
    try:
        yield
    finally:
        fused_a2a._buffer = saved
        fused_a2a._buffer_spec = saved_spec
        fused_a2a._hybrid_ep_buffer = saved_hybridep
        fused_a2a._hybridep_dispatch_replay_state.recorder = saved_recorder
        fused_a2a._hybridep_dispatch_replay_state.mode = saved_mode


def test_free_buffer_destroys_and_clears():
    sentinel = mock.MagicMock()
    fused_a2a._buffer = sentinel

    fused_a2a.free_buffer()

    sentinel.destroy.assert_called_once_with()
    assert fused_a2a._buffer is None
    assert fused_a2a._buffer_spec is None


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


class _CompletedEvent:
    def current_stream_wait(self):
        pass


class _FakeElasticBuffer:
    def __init__(self):
        self.dispatch_kwargs = None
        self.combine_kwargs = None

    def dispatch(self, x, **kwargs):
        self.dispatch_kwargs = kwargs
        handle = SimpleNamespace(
            num_unaligned_recv_tokens_per_expert=torch.tensor([2, 1], device=x.device),
            num_recv_tokens_per_expert_list=[2, 1],
        )
        recv_weights = kwargs["topk_weights"].reshape(-1)[: x.shape[0]]
        return x, None, recv_weights, handle, _CompletedEvent()

    def combine(self, x, handle, **kwargs):
        self.combine_kwargs = kwargs
        combined_weights = kwargs.get("topk_weights")
        if combined_weights is not None:
            combined_weights = combined_weights.reshape(x.shape[0], -1)
        return x, combined_weights, _CompletedEvent()


def test_sync_free_dispatch_uses_elastic_expanded_layout(monkeypatch):
    buffer = _FakeElasticBuffer()
    monkeypatch.setattr(fused_a2a, "get_buffer", lambda *args, **kwargs: buffer)
    x = torch.randn(3, 4, requires_grad=True)
    indices = torch.tensor([[0], [1], [0]])
    probs = torch.full((3, 1), 0.5, requires_grad=True)

    recv_x, recv_indices, recv_probs, tokens_per_expert, _ = fused_a2a.FusedDispatch.apply(
        x,
        indices,
        probs,
        2,
        object(),
        False,
        False,
        True,
    )

    assert recv_indices is None
    assert torch.equal(tokens_per_expert, torch.tensor([2, 1]))
    assert buffer.dispatch_kwargs["do_cpu_sync"] is False
    assert buffer.dispatch_kwargs["do_expand"] is True
    assert buffer.dispatch_kwargs["do_zero_padding"] is True
    assert buffer.dispatch_kwargs["async_with_compute_stream"] is False

    (recv_x.sum() + recv_probs.sum()).backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert torch.equal(probs.grad, torch.ones_like(probs))
    assert buffer.combine_kwargs["async_with_compute_stream"] is False


def test_combine_backward_preserves_expanded_dispatch_shape(monkeypatch):
    class FakeBuffer:
        def combine(self, x, handle, **kwargs):
            return x[:3], None, _CompletedEvent()

        def dispatch(self, x, **kwargs):
            self.dispatch_kwargs = kwargs
            expanded = torch.ones(6, x.shape[1], dtype=x.dtype)
            return expanded, None, None, kwargs["handle"], _CompletedEvent()

    buffer = FakeBuffer()
    monkeypatch.setattr(fused_a2a, "get_buffer", lambda *args, **kwargs: buffer)
    handle = SimpleNamespace(
        num_max_tokens_per_rank=3,
        topk_idx=torch.tensor([[0], [1], [0]]),
        do_expand=True,
    )
    x = torch.randn(6, 4, requires_grad=True)

    combined, _ = fused_a2a.FusedCombine.apply(x, object(), handle, False, False)
    combined.sum().backward()

    assert x.grad.shape == x.shape
    assert buffer.dispatch_kwargs["do_cpu_sync"] is False
    assert buffer.dispatch_kwargs["do_expand"] is True
    assert buffer.dispatch_kwargs["do_zero_padding"] is True


class _DriftingHybridEPBuffer:
    """Fake a full-layout replay that returns a different receive-token count."""

    def __init__(self):
        self.full_dispatches = 0
        self.cached_dispatches = 0
        self.input_shape = None
        self.replayed_num_permuted_tokens = None

    def dispatch_with_permute(self, *, hidden, routing_map=None, probs=None, handle=None, **kwargs):
        if handle is not None:
            self.cached_dispatches += 1
            self.replayed_num_permuted_tokens = kwargs["num_permuted_tokens"]
            # Model the scalar conversion performed by HybridEP when its
            # sync-free token extent is accidentally passed as a tensor.
            if isinstance(self.replayed_num_permuted_tokens, torch.Tensor):
                int(self.replayed_num_permuted_tokens.item())
            dispatched_hidden = torch.cat((hidden, hidden[:1]), dim=0)
            dispatched_probs = torch.cat((probs[:, :1], probs[:1, :1]), dim=0)
            return dispatched_hidden, dispatched_probs, None, None, None

        self.full_dispatches += 1
        self.input_shape = hidden.shape
        # The first call models checkpoint forward. A second full-layout call
        # models HybridEP's nondeterministic recompute and deliberately drifts.
        if self.full_dispatches == 1:
            dispatched_hidden = torch.cat((hidden, hidden[:1]), dim=0)
            dispatched_probs = torch.cat((probs[:, :1], probs[:1, :1]), dim=0)
            tokens_per_expert = torch.tensor([2, 3])
        else:
            dispatched_hidden = hidden[:-1]
            dispatched_probs = probs[:-1, :1]
            tokens_per_expert = torch.tensor([1, 2])
        return dispatched_hidden, dispatched_probs, None, tokens_per_expert, "forward-layout"

    def combine_with_unpermute(self, *, hidden, probs=None, **kwargs):
        combined_hidden = hidden[: self.input_shape[0]]
        combined_probs = None if probs is None else torch.zeros(self.input_shape[0], 2, dtype=probs.dtype)
        return combined_hidden, combined_probs


def _run_checkpointed_hybridep(context_fn):
    x = torch.randn(4, 3, requires_grad=True)
    routing_map = torch.ones(4, 2, dtype=torch.bool)
    probs = torch.full((4, 2), 0.5, requires_grad=True)

    def block(hidden, token_probs):
        dispatched_hidden, dispatched_probs, _, _, _ = fused_a2a.HybridEPDispatch.apply(
            hidden,
            routing_map,
            token_probs,
            object(),
            1,
            24,
            24,
            None,
            None,
        )
        return dispatched_hidden.sin().sum() + dispatched_probs.square().sum()

    loss = checkpoint(block, x, probs, use_reentrant=False, context_fn=context_fn)
    loss.backward()


def test_hybridep_checkpoint_without_layout_replay_detects_shape_drift():
    buffer = _DriftingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer

    with pytest.raises(CheckpointError, match="different metadata"):
        _run_checkpointed_hybridep(lambda: (nullcontext(), nullcontext()))

    assert buffer.full_dispatches == 2
    assert buffer.cached_dispatches == 0


def test_hybridep_checkpoint_reuses_forward_layout_on_recompute():
    from nemo_automodel.components.moe.parallelizer import _replay_hybridep_dispatch_on_recompute

    buffer = _DriftingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer
    context_fn = _replay_hybridep_dispatch_on_recompute(lambda: (nullcontext(), nullcontext()))

    _run_checkpointed_hybridep(context_fn)

    assert buffer.full_dispatches == 1
    assert buffer.cached_dispatches == 1


def test_hybridep_checkpoint_replay_preserves_selective_op_trace():
    from nemo_automodel.components.moe.parallelizer import _replay_hybridep_dispatch_on_recompute

    buffer = _DriftingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer
    aten_sum = torch.ops.aten.sum.default
    aten_local_scalar_dense = torch.ops.aten._local_scalar_dense.default

    def save_replay_sensitive_ops(ctx, func, *args, **kwargs):
        replay_sensitive_ops = (aten_sum, aten_local_scalar_dense)
        return CheckpointPolicy.MUST_SAVE if func in replay_sensitive_ops else CheckpointPolicy.PREFER_RECOMPUTE

    context_fn = _replay_hybridep_dispatch_on_recompute(
        lambda: create_selective_checkpoint_contexts(save_replay_sensitive_ops)
    )

    _run_checkpointed_hybridep(context_fn)

    assert buffer.full_dispatches == 1
    assert buffer.cached_dispatches == 1
    assert buffer.replayed_num_permuted_tokens == 5
    assert isinstance(buffer.replayed_num_permuted_tokens, int)
