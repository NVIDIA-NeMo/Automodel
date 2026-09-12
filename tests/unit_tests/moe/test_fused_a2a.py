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

"""Unit tests for DeepEP and HybridEP autograd wiring."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


@pytest.fixture(autouse=True)
def _restore_buffer():
    """Save/restore module-global buffers so tests don't leak state."""
    saved = fused_a2a._buffer
    saved_hybrid_ep = fused_a2a._hybrid_ep_buffer
    try:
        yield
    finally:
        fused_a2a._buffer = saved
        fused_a2a._hybrid_ep_buffer = saved_hybrid_ep


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


def test_hybridep_autograd_forwards_permute_fusion_in_both_directions():
    buffer = mock.MagicMock()
    hidden = torch.randn(4, 8)
    probs = torch.randn(4, 2)
    routing_map = torch.ones(4, 2, dtype=torch.bool)
    tokens_per_expert = torch.tensor([2, 2])
    handle = object()
    buffer.dispatch_with_permute.return_value = (hidden, probs, None, tokens_per_expert, handle)
    buffer.combine_with_unpermute.return_value = (hidden, probs)
    fused_a2a._hybrid_ep_buffer = buffer

    dispatch_ctx = SimpleNamespace()
    fused_a2a.HybridEPDispatch.forward(
        dispatch_ctx,
        hidden,
        routing_map,
        probs,
        None,
        2,
        permute_fusion=True,
    )
    dispatch_grads = fused_a2a.HybridEPDispatch.backward(dispatch_ctx, hidden, probs, None, None, None)

    combine_ctx = SimpleNamespace()
    fused_a2a.HybridEPCombine.forward(combine_ctx, hidden, handle, permute_fusion=True)
    combine_grads = fused_a2a.HybridEPCombine.backward(combine_ctx, hidden)

    assert buffer.dispatch_with_permute.call_args_list[0].kwargs["fuse_permute_dispatch"] is True
    assert buffer.dispatch_with_permute.call_args_list[1].kwargs["fuse_permute_dispatch"] is True
    assert buffer.combine_with_unpermute.call_args_list[0].kwargs["fuse_unpermute_combine"] is True
    assert buffer.combine_with_unpermute.call_args_list[1].kwargs["fuse_unpermute_combine"] is True
    assert len(dispatch_grads) == 10
    assert len(combine_grads) == 5
