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
