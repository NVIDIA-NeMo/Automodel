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

import inspect

import pytest

import nemo_automodel._diffusers.diffusers_patches as patches

# (apply function, {op name -> (fixed op, buggy source marker)})
PATCH_CASES = [
    (
        patches.apply_native_flash_backward_patch,
        {
            "_native_flash_attention_backward_op": (
                patches._fixed_native_flash_attention_backward_op,
                patches._BUGGY_KV_TRANSPOSE_MARKER,
            ),
        },
    ),
    (
        patches.apply_cudnn_attention_patch,
        {
            "_cudnn_attention_forward_op": (
                patches._fixed_cudnn_attention_forward_op,
                patches._BUGGY_CUDNN_LSE_MARKER,
            ),
            "_cudnn_attention_backward_op": (
                patches._fixed_cudnn_attention_backward_op,
                patches._BUGGY_KV_TRANSPOSE_MARKER,
            ),
        },
    ),
]

ALL_OP_NAMES = [name for _, ops in PATCH_CASES for name in ops]


@pytest.fixture(autouse=True)
def _reset_patch_state(monkeypatch):
    """Isolate the module-level applied set and restore diffusers between tests."""
    from diffusers.models import attention_dispatch

    originals = {name: getattr(attention_dispatch, name) for name in ALL_OP_NAMES}
    monkeypatch.setattr(patches, "_APPLIED_PATCHES", set())
    yield
    for name, fn in originals.items():
        setattr(attention_dispatch, name, fn)


@pytest.mark.parametrize("apply_patch,ops", PATCH_CASES)
def test_patch_applies_on_buggy_diffusers(apply_patch, ops):
    from diffusers.models import attention_dispatch

    originals = {name: getattr(attention_dispatch, name) for name in ops}
    applied = apply_patch()

    any_buggy = False
    for name, (fixed_op, marker) in ops.items():
        if marker in inspect.getsource(originals[name]):
            any_buggy = True
            assert getattr(attention_dispatch, name) is fixed_op
        else:
            # Upstream already fixed this op: it must be left untouched.
            assert getattr(attention_dispatch, name) is originals[name]
    assert applied is any_buggy


@pytest.mark.parametrize("apply_patch,ops", PATCH_CASES)
def test_patch_is_idempotent(apply_patch, ops):
    first = apply_patch()
    second = apply_patch()
    assert first == second


@pytest.mark.parametrize("apply_patch,ops", PATCH_CASES)
def test_patch_skips_fixed_upstream(monkeypatch, apply_patch, ops):
    from diffusers.models import attention_dispatch

    def already_fixed_op(ctx, *args, **kwargs):
        return None

    for name in ops:
        monkeypatch.setattr(attention_dispatch, name, already_fixed_op)

    assert apply_patch() is False
    for name in ops:
        assert getattr(attention_dispatch, name) is already_fixed_op


def test_patches_are_independent():
    """Patching one backend must not touch the other backend's ops."""
    from diffusers.models import attention_dispatch

    original_cudnn_fwd = attention_dispatch._cudnn_attention_forward_op
    original_cudnn_bwd = attention_dispatch._cudnn_attention_backward_op
    patches.apply_native_flash_backward_patch()
    assert attention_dispatch._cudnn_attention_forward_op is original_cudnn_fwd
    assert attention_dispatch._cudnn_attention_backward_op is original_cudnn_bwd
