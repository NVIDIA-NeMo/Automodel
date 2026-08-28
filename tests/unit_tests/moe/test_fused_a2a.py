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
from torch.utils.checkpoint import CheckpointError, checkpoint

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


@pytest.fixture(autouse=True)
def _restore_buffer():
    """Save/restore module-global dispatch state so tests don't leak state."""
    saved = fused_a2a._buffer
    saved_hybridep = fused_a2a._hybrid_ep_buffer
    saved_recorder = fused_a2a._hybridep_dispatch_replay_state.recorder
    saved_mode = fused_a2a._hybridep_dispatch_replay_state.mode
    try:
        yield
    finally:
        fused_a2a._buffer = saved
        fused_a2a._hybrid_ep_buffer = saved_hybridep
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


class _DriftingHybridEPBuffer:
    """Fake a full-layout replay that returns a different receive-token count."""

    def __init__(self):
        self.full_dispatches = 0
        self.cached_dispatches = 0
        self.input_shape = None

    def dispatch_with_permute(self, *, hidden, routing_map=None, probs=None, handle=None, **kwargs):
        if handle is not None:
            self.cached_dispatches += 1
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
