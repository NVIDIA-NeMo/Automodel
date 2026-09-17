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
from unittest import mock

import pytest
import torch
from torch.utils.checkpoint import CheckpointError, CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


@pytest.fixture(autouse=True)
def _restore_buffer():
    """Save/restore module-global dispatch state so tests don't leak state."""
    saved = fused_a2a._buffer
    saved_hybridep = fused_a2a._hybrid_ep_buffer
    saved_hybridep_signature = fused_a2a._hybrid_ep_runtime_signature
    saved_hybridep_capacity = fused_a2a._hybrid_ep_initialized_capacity
    saved_recorder = fused_a2a._hybridep_dispatch_replay_state.recorder
    saved_mode = fused_a2a._hybridep_dispatch_replay_state.mode
    try:
        yield
    finally:
        fused_a2a._buffer = saved
        fused_a2a._hybrid_ep_buffer = saved_hybridep
        fused_a2a._hybrid_ep_runtime_signature = saved_hybridep_signature
        fused_a2a._hybrid_ep_initialized_capacity = saved_hybridep_capacity
        fused_a2a._hybridep_dispatch_replay_state.recorder = saved_recorder
        fused_a2a._hybridep_dispatch_replay_state.mode = saved_mode


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
    assert "dense_routing" not in buffer.dispatch_kwargs
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


class _DriftingHybridEPBuffer:
    """Fake a full-layout replay that returns a different receive-token count."""

    def __init__(self):
        self.full_dispatches = 0
        self.cached_dispatches = 0
        self.input_shape = None
        self.replayed_num_permuted_tokens = None
        self.replayed_fuse_permute = None
        self.combined_fuse_permute = None

    def dispatch_with_permute(self, *, hidden, routing_map=None, probs=None, handle=None, **kwargs):
        if handle is not None:
            self.cached_dispatches += 1
            self.replayed_num_permuted_tokens = kwargs["num_permuted_tokens"]
            self.replayed_fuse_permute = kwargs.get("fuse_permute_dispatch", False)
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
        self.combined_fuse_permute = kwargs.get("fuse_unpermute_combine", False)
        combined_hidden = hidden[: self.input_shape[0]]
        combined_probs = None if probs is None else torch.zeros(self.input_shape[0], 2, dtype=probs.dtype)
        return combined_hidden, combined_probs


def _run_checkpointed_hybridep(context_fn, num_permuted_tokens=None, *, fuse_permute=False):
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
            num_permuted_tokens,
            None,
            None,
            None,
            fuse_permute,
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


def test_hybridep_recorder_keeps_a_host_extent_without_reducing_tokens_per_expert():
    recorder = fused_a2a.HybridEPDispatchReplayRecorder()
    tokens_per_expert = mock.MagicMock(spec=torch.Tensor)
    # capacity mode / static pin: the forward already ran with a host-side extent
    recorder.record("layout", tokens_per_expert, 24)
    # blocking dispatch: the extent comes from the reduction, after the checkpoint context exits
    recorder.record("layout", torch.tensor([2, 3]))
    recorder.finalize()

    assert recorder.take() == ["layout", tokens_per_expert, 24]
    assert recorder.take()[2] == 5
    tokens_per_expert.sum.assert_not_called()


def test_hybridep_checkpoint_replay_preserves_permute_fusion():
    from nemo_automodel.components.moe.parallelizer import _replay_hybridep_dispatch_on_recompute

    buffer = _DriftingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer
    context_fn = _replay_hybridep_dispatch_on_recompute(lambda: (nullcontext(), nullcontext()))

    _run_checkpointed_hybridep(context_fn, fuse_permute=True)

    assert buffer.cached_dispatches == 1
    assert buffer.replayed_fuse_permute is True
    assert buffer.combined_fuse_permute is True


def test_hybridep_checkpoint_replay_reuses_the_forward_capacity_extent():
    from nemo_automodel.components.moe.parallelizer import _replay_hybridep_dispatch_on_recompute

    buffer = _DriftingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer
    context_fn = _replay_hybridep_dispatch_on_recompute(lambda: (nullcontext(), nullcontext()))

    _run_checkpointed_hybridep(context_fn, num_permuted_tokens=24)

    assert buffer.full_dispatches == 1
    assert buffer.cached_dispatches == 1
    # the recompute output must be sized like the forward's (capacity rows), not to this dispatch's token count
    assert buffer.replayed_num_permuted_tokens == 24
