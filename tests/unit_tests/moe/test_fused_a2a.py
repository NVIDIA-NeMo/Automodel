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

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


@pytest.fixture(autouse=True)
def _restore_buffer():
    """Save/restore module-global communication buffers so tests don't leak state."""
    saved = fused_a2a._buffer
    saved_hybrid = fused_a2a._hybrid_ep_buffer
    saved_hybrid_identity = fused_a2a._hybrid_ep_buffer_identity
    try:
        yield
    finally:
        fused_a2a._buffer = saved
        fused_a2a._hybrid_ep_buffer = saved_hybrid
        fused_a2a._hybrid_ep_buffer_identity = saved_hybrid_identity


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


@pytest.mark.parametrize("preprocessing_sms", [None, 32])
def test_init_hybrid_ep_buffer_preserves_default_or_forwards_preprocessing_sms(preprocessing_sms):
    """The opt-in knob maps exactly to HybridEPBuffer without overriding its default."""
    group = object()
    with mock.patch.object(fused_a2a, "HybridEPBuffer", create=True) as buffer_cls:
        fused_a2a.init_hybrid_ep_buffer(
            group=group,
            hidden_dim=128,
            seq_len=64,
            num_local_experts=2,
            num_sms_dispatch_api=24,
            num_sms_combine_api=24,
            fp8_dispatch=False,
            num_sms_preprocessing_api=preprocessing_sms,
        )

    buffer_kwargs = buffer_cls.call_args.kwargs
    if preprocessing_sms is None:
        assert "num_sms_preprocessing_api" not in buffer_kwargs
    else:
        assert buffer_kwargs["num_sms_preprocessing_api"] == preprocessing_sms
    assert fused_a2a._hybrid_ep_buffer_identity[-1] == preprocessing_sms


def test_hybrid_ep_buffer_identity_rejects_preprocessing_sms_changes():
    """A process-global buffer cannot silently serve managers with incompatible SM tuning."""
    group = object()
    fused_a2a._hybrid_ep_buffer = mock.MagicMock()
    fused_a2a._hybrid_ep_buffer_identity = fused_a2a._get_hybrid_ep_buffer_identity(
        group,
        hidden_dim=128,
        num_local_experts=2,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        fp8_dispatch=False,
        num_sms_preprocessing_api=None,
    )

    with pytest.raises(RuntimeError, match="different communication or geometry settings"):
        fused_a2a.HybridEPDispatch.forward(
            mock.Mock(),
            mock.Mock(shape=(64, 128)),
            mock.Mock(),
            mock.Mock(),
            group,
            2,
            24,
            24,
            None,
            None,
            32,
        )


def test_hybrid_ep_buffer_identity_allows_dynamic_sequence_length_changes():
    """Token-count growth reuses HybridEP's runtime-reallocatable buffer geometry."""
    group = object()
    buffer = mock.MagicMock()
    buffer.dispatch_with_permute.return_value = (mock.Mock(), mock.Mock(), None, mock.Mock(), mock.Mock())
    fused_a2a._hybrid_ep_buffer = buffer
    fused_a2a._hybrid_ep_buffer_identity = fused_a2a._get_hybrid_ep_buffer_identity(
        group,
        hidden_dim=128,
        num_local_experts=2,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        fp8_dispatch=False,
        num_sms_preprocessing_api=32,
    )

    fused_a2a.HybridEPDispatch.forward(
        mock.Mock(),
        mock.Mock(shape=(128, 128)),
        mock.Mock(),
        mock.Mock(),
        group,
        2,
        24,
        24,
        None,
        None,
        32,
    )

    buffer.dispatch_with_permute.assert_called_once()


@pytest.mark.parametrize(
    ("hidden_dim", "num_local_experts"),
    [
        (256, 2),
        (128, 4),
    ],
)
def test_hybrid_ep_buffer_identity_rejects_geometry_changes(hidden_dim, num_local_experts):
    """Hidden size and local expert count are immutable HybridEP buffer geometry."""
    group = object()
    fused_a2a._hybrid_ep_buffer = mock.MagicMock()
    fused_a2a._hybrid_ep_buffer_identity = fused_a2a._get_hybrid_ep_buffer_identity(
        group,
        hidden_dim=128,
        num_local_experts=2,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        fp8_dispatch=False,
        num_sms_preprocessing_api=32,
    )

    with pytest.raises(RuntimeError, match="different communication or geometry settings"):
        fused_a2a.HybridEPDispatch.forward(
            mock.Mock(),
            mock.Mock(shape=(64, hidden_dim)),
            mock.Mock(),
            mock.Mock(),
            group,
            num_local_experts,
            24,
            24,
            None,
            None,
            32,
        )


def test_reset_hybrid_ep_buffer_clears_identity():
    fused_a2a._hybrid_ep_buffer = mock.MagicMock()
    fused_a2a._hybrid_ep_buffer_identity = ("configured",)

    fused_a2a.reset_hybrid_ep_buffer()

    assert fused_a2a._hybrid_ep_buffer is None
    assert fused_a2a._hybrid_ep_buffer_identity is None
