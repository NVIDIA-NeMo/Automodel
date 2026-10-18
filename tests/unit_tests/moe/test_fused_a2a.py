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
    try:
        yield
    finally:
        fused_a2a._buffer = saved
        fused_a2a._hybrid_ep_buffer = saved_hybridep
        fused_a2a._hybrid_ep_runtime_signature = saved_hybridep_signature
        fused_a2a._hybrid_ep_initialized_capacity = saved_hybridep_capacity


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


@pytest.mark.parametrize(
    ("extent", "expect_non_blocking"),
    [(24, True), (torch.tensor(24), False), (None, False)],
    ids=["capacity-int", "static-pin-tensor", "blocking"],
)
def test_hybridep_combine_backward_dispatch_is_non_blocking_only_for_a_host_int_extent(
    monkeypatch, extent, expect_non_blocking
):
    """Capacity mode's host int keeps the backward dispatch off the stream drain; a tensor pin or None does not."""
    seen = {}

    class FakeHybridEPBuffer:
        def combine_with_unpermute(self, *, hidden, handle, pad_multiple, fuse_unpermute_combine=False):
            return hidden, None

        def dispatch_with_permute(self, **kwargs):
            seen.update(kwargs)
            return kwargs["hidden"], None, None, torch.tensor([1, 1]), ("unused",)

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", FakeHybridEPBuffer())
    hidden = torch.randn(2, 4, requires_grad=True)

    fused_a2a.HybridEPCombine.apply(hidden, ("handle",), extent, None, False).sum().backward()

    assert seen["non_blocking"] is expect_non_blocking
    assert seen["num_permuted_tokens"] is extent


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


def test_hybridep_replay_recorder_reads_extents_lazily_and_once():
    """finalize() only starts the extent read; take()/resolve() turn it into an int, idempotently."""
    recorder = fused_a2a.HybridEPDispatchReplayRecorder()
    recorder.record(["layout-a", torch.tensor([2, 3]), None])
    recorder.record(["layout-b", torch.tensor([1, 2]), None])
    recorder.finalize()
    recorder.resolve()
    recorder.resolve()  # a second resolve has nothing left to do
    first = recorder.take()
    assert first[0] == "layout-a" and first[2] == 5 and isinstance(first[2], int)
    second = recorder.take()
    assert second[0] == "layout-b" and second[2] == 3 and isinstance(second[2], int)
    assert recorder.take() is None and recorder.replay_misses == 1
    # take() without an explicit resolve() still yields a host integer
    lazy = fused_a2a.HybridEPDispatchReplayRecorder()
    lazy.record(["layout-c", torch.tensor([4]), None])
    lazy.finalize()
    assert lazy.take()[2] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.runtime_budget(15, reason="may create the process's CUDA context")
def test_hybridep_replay_recorder_copies_cuda_extents_asynchronously():
    """On CUDA, finalize() leaves a pinned scalar and an event in flight; resolve()/take() yield the int."""
    recorder = fused_a2a.HybridEPDispatchReplayRecorder()
    recorder.record(["layout-a", torch.tensor([2, 3], device="cuda"), None])
    recorder.finalize()
    entry = recorder._records[0]
    assert entry[2] is None
    host, ready = recorder._copies[0]
    assert host.device.type == "cpu" and host.is_pinned() and isinstance(ready, torch.cuda.Event)
    recorder.resolve()
    assert entry[2] == 5 and isinstance(entry[2], int)
    assert recorder._copies[0] is None
    # take() without an explicit resolve() still yields a host integer
    lazy = fused_a2a.HybridEPDispatchReplayRecorder()
    lazy.record(["layout-b", torch.tensor([4], device="cuda"), None])
    lazy.finalize()
    assert lazy.take()[2] == 4


def test_hybridep_recorder_keeps_a_host_extent_without_reducing_tokens_per_expert():
    recorder = fused_a2a.HybridEPDispatchReplayRecorder()
    tokens_per_expert = mock.MagicMock(spec=torch.Tensor)
    # capacity mode: the forward already ran with a host-side extent
    recorder.record(["layout", tokens_per_expert, 24])
    # blocking dispatch: the extent comes from the reduction, after the checkpoint context exits
    recorder.record(["layout", torch.tensor([2, 3]), None])
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


# ---------------------------------------------------------------------------
# BackendConfig.dispatcher_recompute_cache: the combine output of the checkpoint
# forward is served in the recompute
# ---------------------------------------------------------------------------


class _CountingHybridEPBuffer(_DriftingHybridEPBuffer):
    """Non-drifting buffer (the same layout on every full dispatch) that counts combines and makes each forward
    combine's result depend on the call count, so a served cache is distinguishable from a recomputed combine."""

    def __init__(self):
        super().__init__()
        self.combines = 0
        self.grad_combines = 0
        self.grad_dispatches = 0

    def dispatch_with_permute(self, *, hidden, routing_map=None, probs=None, handle=None, **kwargs):
        if handle is not None and probs is None:
            # HybridEPCombine.backward: the gradient all-to-all through the saved handle (no probabilities).
            self.grad_dispatches += 1
            return torch.cat((hidden, hidden[:1]), dim=0), None, None, None, None
        if handle is None:
            # Full dispatch (forward and, without layout replay, the recompute): always the forward layout.
            self.full_dispatches += 1
            self.input_shape = hidden.shape
            dispatched_hidden = torch.cat((hidden, hidden[:1]), dim=0)
            dispatched_probs = torch.cat((probs[:, :1], probs[:1, :1]), dim=0)
            return dispatched_hidden, dispatched_probs, None, torch.tensor([2, 3]), "forward-layout"
        return super().dispatch_with_permute(
            hidden=hidden, routing_map=routing_map, probs=probs, handle=handle, **kwargs
        )

    def combine_with_unpermute(self, *, hidden, probs=None, **kwargs):
        combined_hidden, combined_probs = super().combine_with_unpermute(hidden=hidden, probs=probs, **kwargs)
        if probs is not None:
            # HybridEPDispatch.backward: the gradient combine through the saved handle, left untouched.
            self.grad_combines += 1
            return combined_hidden, combined_probs
        self.combines += 1
        # A recomputed forward combine sees different values; the cache must hide that.
        return combined_hidden * float(self.combines), combined_probs


def _run_checkpointed_hybridep_with_combine(context_fn, *, cache_output):
    torch.manual_seed(0)
    x = torch.randn(4, 3, requires_grad=True)
    routing_map = torch.ones(4, 2, dtype=torch.bool)
    probs = torch.full((4, 2), 0.5, requires_grad=True)

    def block(hidden, token_probs):
        dispatched_hidden, dispatched_probs, _, _, handle = fused_a2a.HybridEPDispatch.apply(
            hidden, routing_map, token_probs, object(), 1, 24, 24, None, None, None, None, False
        )
        expert_out = dispatched_hidden.sin()
        combined = fused_a2a.HybridEPCombine.apply(expert_out, handle, None, None, False, cache_output)
        # A consumer that saves its input, like the decoder block's ops after the MoE: without one the
        # non-reentrant checkpoint stops the recompute before the combine (early stop).
        return combined.square().sum() + dispatched_probs.square().sum()

    loss = checkpoint(block, x, probs, use_reentrant=False, context_fn=context_fn)
    loss.backward()
    return loss.detach(), x.grad.clone(), probs.grad.clone()


def test_moe_recompute_cache_round_trip():
    cache = fused_a2a.MoERecomputeCache()
    assert not cache.has_next()
    first, second = torch.ones(2), torch.zeros(2)
    cache.record(first)
    cache.record(second)
    first.fill_(5.0)  # the cache keeps a clone, not an alias
    assert cache.has_next()
    assert torch.equal(cache.take(), torch.ones(2))
    assert torch.equal(cache.take(), torch.zeros(2))
    assert not cache.has_next()
    assert cache.take() is None
    assert cache.replay_misses == 1
    cache.rewind()
    assert cache.has_next()  # cursor rewound; entries already handed out are None
    assert cache.take() is None


def test_combine_cache_serves_the_forward_output_in_the_recompute_without_layout_replay():
    """With the router recomputed (ignore_router_for_ac=False) there is no HybridEP layout replay and the
    recompute runs a full dispatch -- the combine cache must still work on its own scope."""
    from nemo_automodel.components.moe.parallelizer import _cache_moe_combine_on_recompute

    results = {}
    for cache_output in (False, True):
        buffer = _CountingHybridEPBuffer()
        fused_a2a._hybrid_ep_buffer = buffer
        context_fn = _cache_moe_combine_on_recompute(lambda: (nullcontext(), nullcontext()))
        results[cache_output] = (_run_checkpointed_hybridep_with_combine(context_fn, cache_output=cache_output), buffer)

    plain_buffer, cached_buffer = results[False][1], results[True][1]
    assert plain_buffer.full_dispatches == cached_buffer.full_dispatches == 2  # forward + recompute, no replay
    assert plain_buffer.combines == 2  # forward + recompute
    assert cached_buffer.combines == 1  # the recompute returned the cached forward output
    assert plain_buffer.grad_combines == cached_buffer.grad_combines == 1  # dispatch backward unchanged
    assert plain_buffer.grad_dispatches == cached_buffer.grad_dispatches == 1  # combine backward unchanged

    loss_plain, dx_plain, dp_plain = results[False][0]
    loss_cached, dx_cached, dp_cached = results[True][0]
    assert torch.equal(loss_plain, loss_cached)
    assert torch.equal(dp_plain, dp_cached)
    # The non-checkpointed run's input gradient: the cached run reproduces it (the recompute saw the forward's
    # combine values), the plain checkpointed run does not (its recompute saw the counting buffer's x2 values).
    buffer = _CountingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer
    torch.manual_seed(0)
    x = torch.randn(4, 3, requires_grad=True)
    routing_map = torch.ones(4, 2, dtype=torch.bool)
    probs = torch.full((4, 2), 0.5, requires_grad=True)
    dispatched_hidden, dispatched_probs, _, _, handle = fused_a2a.HybridEPDispatch.apply(
        x, routing_map, probs, object(), 1, 24, 24, None, None, None, None, False
    )
    combined = fused_a2a.HybridEPCombine.apply(dispatched_hidden.sin(), handle, None, None, False, False)
    (combined.square().sum() + dispatched_probs.square().sum()).backward()
    assert torch.equal(x.grad, dx_cached)
    assert not torch.equal(x.grad, dx_plain)


def test_combine_cache_composes_with_the_hybridep_layout_replay():
    """ignore_router_for_ac=True: the layout replay and the cache stack; one full dispatch, one forward combine."""
    from nemo_automodel.components.moe.parallelizer import (
        _cache_moe_combine_on_recompute,
        _replay_hybridep_dispatch_on_recompute,
    )

    buffer = _CountingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer
    context_fn = _cache_moe_combine_on_recompute(
        _replay_hybridep_dispatch_on_recompute(lambda: (nullcontext(), nullcontext()))
    )
    _run_checkpointed_hybridep_with_combine(context_fn, cache_output=True)
    assert buffer.full_dispatches == 1
    assert buffer.cached_dispatches == 1
    assert buffer.combines == 1


def test_moe_recompute_phase_and_combine_cache_flag():
    cache = fused_a2a.MoERecomputeCache()
    assert fused_a2a.moe_recompute_phase() is None
    assert fused_a2a.hybridep_combine_output_cached() is False
    with fused_a2a.moe_combine_replay.scope(cache, "record"):
        assert fused_a2a.moe_recompute_phase() == "record"
        assert fused_a2a.hybridep_combine_output_cached() is False
        cache.record(torch.ones(1))
    with fused_a2a.moe_combine_replay.scope(cache, "replay"):
        assert fused_a2a.moe_recompute_phase() == "replay"
        assert fused_a2a.hybridep_combine_output_cached() is True
        cache.take()
        assert fused_a2a.hybridep_combine_output_cached() is False
    assert fused_a2a.moe_recompute_phase() is None
    with pytest.raises(ValueError, match="Unsupported MoE combine replay mode"):
        with fused_a2a.moe_combine_replay.scope(cache, "forward"):
            pass


def test_hybridep_combine_without_cache_flag_ignores_the_bound_cache():
    buffer = _CountingHybridEPBuffer()
    fused_a2a._hybrid_ep_buffer = buffer
    cache = fused_a2a.MoERecomputeCache()
    x = torch.randn(3, 2)
    with fused_a2a.moe_combine_replay.scope(cache, "record"):
        buffer.input_shape = (3, 2)
        fused_a2a.HybridEPCombine.apply(x, "handle", None, None, False, False)
    assert not cache.has_next()
    assert buffer.combines == 1
