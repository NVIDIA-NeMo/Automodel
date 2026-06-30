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

"""Unit tests for fused all-to-all teardown and autograd wrappers."""

import os
from unittest import mock

import pytest
import torch

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


@pytest.fixture(autouse=True)
def _restore_buffers():
    """Save and restore process-global communication buffers."""
    saved_buffer = fused_a2a._buffer
    saved_hybrid_ep_buffer = fused_a2a._hybrid_ep_buffer
    try:
        yield
    finally:
        fused_a2a._buffer = saved_buffer
        fused_a2a._hybrid_ep_buffer = saved_hybrid_ep_buffer


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


def test_hybridep_combine_backward_returns_one_gradient_per_input(monkeypatch):
    class _FakeHybridEPBuffer:
        @staticmethod
        def combine_with_unpermute(*, hidden, **_kwargs):
            return hidden.clone(), None

        @staticmethod
        def dispatch_with_permute(*, hidden, **_kwargs):
            return hidden.clone(), None, None, None, None

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", _FakeHybridEPBuffer())
    x = torch.ones(2, 3, requires_grad=True)

    fused_a2a.HybridEPCombine.apply(x, object(), None, None).sum().backward()

    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_hybridep_jit_cache_reuses_and_persists_kernels(monkeypatch, tmp_path):
    monkeypatch.setenv("HYBRID_EP_CACHE_DIR", str(tmp_path))
    jit_dir = tmp_path / ".deepep" / "hybrid_ep" / "jit"
    stable_dir, process_dir = jit_dir / "kernel-cache", jit_dir / f"proc-{os.getpid()}"
    os.makedirs(stable_dir)
    cached_kernel = os.path.join(stable_dir, "cached.so")
    with open(cached_kernel, "wb") as kernel_file:
        kernel_file.write(b"cached")

    fused_a2a._sync_hybridep_jit_cache(persist=False)
    assert os.path.samefile(cached_kernel, os.path.join(process_dir, "cached.so"))

    compiled_kernel = os.path.join(process_dir, "compiled.so")
    with open(compiled_kernel, "wb") as kernel_file:
        kernel_file.write(b"compiled")
    fused_a2a._sync_hybridep_jit_cache(persist=True)

    assert os.path.isfile(os.path.join(stable_dir, "compiled.so"))
    assert not os.path.exists(process_dir)


def test_init_hybridep_buffer_always_loads_cached_kernels(monkeypatch):
    buffer = mock.MagicMock()
    hybrid_ep_buffer = mock.MagicMock(return_value=buffer)
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", hybrid_ep_buffer, raising=False)
    monkeypatch.delenv("HYBRID_EP_CACHE_DIR", raising=False)

    fused_a2a.init_hybrid_ep_buffer(
        group=mock.MagicMock(),
        hidden_dim=4096,
        seq_len=4096,
        num_local_experts=4,
        num_sms_dispatch_api=20,
        num_sms_combine_api=20,
        fp8_dispatch=False,
    )

    hybrid_ep_buffer.assert_called_once_with(
        group=mock.ANY,
        hidden_dim=4096,
        max_num_of_tokens_per_rank=4096,
        num_local_experts=4,
        use_fp8=False,
        num_sms_dispatch_api=20,
        num_sms_combine_api=20,
        load_cached_kernels=True,
    )
    assert fused_a2a._hybrid_ep_buffer is buffer
