# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

import nemo_automodel._diffusers.diffusers_patches as patches


@pytest.fixture(autouse=True)
def _reset_patch_state(monkeypatch):
    """Isolate the module-level applied flag and restore diffusers between tests."""
    from diffusers.models import attention_dispatch

    original = attention_dispatch._native_flash_attention_backward_op
    monkeypatch.setattr(patches, "_PATCH_APPLIED", False)
    yield
    attention_dispatch._native_flash_attention_backward_op = original


def test_patch_applies_on_buggy_diffusers():
    from diffusers.models import attention_dispatch

    original = attention_dispatch._native_flash_attention_backward_op
    applied = patches.apply_native_flash_backward_patch()

    if patches._BUGGY_SOURCE_MARKER in __import__("inspect").getsource(original):
        assert applied is True
        assert (
            attention_dispatch._native_flash_attention_backward_op is patches._fixed_native_flash_attention_backward_op
        )
    else:
        # Upstream already fixed: the patch must be a no-op.
        assert applied is False
        assert attention_dispatch._native_flash_attention_backward_op is original


def test_patch_is_idempotent():
    first = patches.apply_native_flash_backward_patch()
    second = patches.apply_native_flash_backward_patch()
    assert first == second


def test_patch_skips_fixed_upstream(monkeypatch):
    from diffusers.models import attention_dispatch

    def already_fixed_backward_op(ctx, grad_out, *args, **kwargs):
        return None

    monkeypatch.setattr(attention_dispatch, "_native_flash_attention_backward_op", already_fixed_backward_op)

    assert patches.apply_native_flash_backward_patch() is False
    assert attention_dispatch._native_flash_attention_backward_op is already_fixed_backward_op
