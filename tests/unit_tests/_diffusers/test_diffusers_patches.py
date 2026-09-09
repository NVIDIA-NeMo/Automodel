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
import torch

import nemo_automodel._diffusers.diffusers_patches as patches

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

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


def test_apply_op_patch_returns_false_when_op_is_missing():
    """A diffusers install without the op (renamed/removed upstream) is a no-op."""

    def unused_fixed_op(ctx, *args, **kwargs):
        return None

    assert patches._apply_op_patch("_nonexistent_attention_op", unused_fixed_op, "marker") is False
    assert "_nonexistent_attention_op" not in patches._APPLIED_PATCHES


def test_apply_op_patch_returns_false_when_source_is_uninspectable(monkeypatch):
    """If the installed op's source cannot be read, feature detection must bail out."""
    from diffusers.models import attention_dispatch

    # ``len`` is a builtin: inspect.getsource raises TypeError on it.
    monkeypatch.setattr(attention_dispatch, "_native_flash_attention_backward_op", len)

    assert patches.apply_native_flash_backward_patch() is False
    assert attention_dispatch._native_flash_attention_backward_op is len
    assert "_native_flash_attention_backward_op" not in patches._APPLIED_PATCHES


def test_fixed_cudnn_forward_rejects_enable_gqa():
    ctx = _FakeFunctionCtx()
    query = torch.randn(1, 4, 2, 8)
    with pytest.raises(ValueError, match="enable_gqa"):
        patches._fixed_cudnn_attention_forward_op(ctx, query, query, query, enable_gqa=True)


# =============================================================================
# GPU numerical parity of the fixed ops against the SDPA autograd reference
# =============================================================================


class _FakeFunctionCtx:
    """Minimal FunctionCtx stand-in: records save_for_backward tensors and attrs.

    The diffusers attention ops only touch ``ctx.saved_tensors`` plus plain
    attributes (dropout_p, is_causal, scale, ...), so a bare object suffices.
    """

    def save_for_backward(self, *tensors):
        """Record tensors for the paired backward op.

        Args:
            *tensors: Tensors in the layouts the forward op saved them
                (query/key/value/out/lse in kernel layout [batch, heads, seq,
                head_dim], plus scalar bookkeeping tensors).
        """
        self.saved_tensors = tensors


def _model_layout_qkv(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build random CUDA query/key/value in diffusers' model layout.

    Args:
        seed: RNG seed for reproducible tensors.

    Returns:
        Tuple of (query, key, value), each of shape
        [batch=2, seq=32, heads=4, head_dim=64], bfloat16, on CUDA.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def make() -> torch.Tensor:
        return torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.bfloat16, generator=generator)

    return make(), make(), make()


def _sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute attention output and input gradients via plain SDPA autograd.

    Args:
        query: Tensor of shape [batch, seq, heads, head_dim] (model layout).
        key: Tensor of shape [batch, seq, heads, head_dim] (model layout).
        value: Tensor of shape [batch, seq, heads, head_dim] (model layout).
        grad_out: Gradient w.r.t. the attention output, of shape
            [batch, seq, heads, head_dim] (model layout).

    Returns:
        Tuple of (out, grad_query, grad_key, grad_value), each of shape
        [batch, seq, heads, head_dim] (model layout).
    """
    leaves = [t.detach().clone().requires_grad_(True) for t in (query, key, value)]
    out = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in leaves)).transpose(1, 2)
    out.backward(grad_out)
    return (out.detach(), *(t.grad for t in leaves))


@requires_cuda
def test_fixed_native_flash_backward_matches_sdpa_grads():
    """The patched flash backward must reproduce SDPA autograd gradients.

    The unpatched diffusers op re-transposes the already-transposed key/value
    saved by the forward, so it either raises or produces garbage gradients.
    """
    from diffusers.models import attention_dispatch

    query, key, value = _model_layout_qkv()
    grad_out = torch.randn_like(query)

    ctx = _FakeFunctionCtx()
    out = attention_dispatch._native_flash_attention_forward_op(ctx, query, key, value)
    grad_query, grad_key, grad_value = patches._fixed_native_flash_attention_backward_op(ctx, grad_out)

    ref_out, ref_gq, ref_gk, ref_gv = _sdpa_reference(query, key, value, grad_out)
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(grad_query, ref_gq, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(grad_key, ref_gk, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(grad_value, ref_gv, atol=3e-2, rtol=3e-2)


@requires_cuda
def test_fixed_cudnn_forward_backward_match_sdpa_grads():
    """The patched cuDNN forward+backward must reproduce SDPA autograd results.

    Covers both cuDNN bugs at once: the forward must save a valid lse for the
    backward, and the backward must not re-transpose key/value.
    """
    query, key, value = _model_layout_qkv(seed=1)
    grad_out = torch.randn_like(query)

    ctx = _FakeFunctionCtx()
    out = patches._fixed_cudnn_attention_forward_op(ctx, query, key, value)
    grad_query, grad_key, grad_value = patches._fixed_cudnn_attention_backward_op(ctx, grad_out)

    ref_out, ref_gq, ref_gk, ref_gv = _sdpa_reference(query, key, value, grad_out)
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(grad_query, ref_gq, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(grad_key, ref_gk, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(grad_value, ref_gv, atol=3e-2, rtol=3e-2)


@requires_cuda
def test_fixed_cudnn_forward_saves_valid_lse_without_return_lse():
    """Training path: return_lse=False must still save a real lse for backward."""
    query, key, value = _model_layout_qkv(seed=2)

    ctx = _FakeFunctionCtx()
    patches._fixed_cudnn_attention_forward_op(ctx, query, key, value, return_lse=False, _save_ctx=True)

    lse = ctx.saved_tensors[4]
    batch, seq, heads, _ = query.shape
    # cuDNN may append a trailing singleton dim to the log-sum-exp.
    assert lse.shape[:3] == (batch, heads, seq)
    assert torch.isfinite(lse).all()


@requires_cuda
def test_fixed_cudnn_forward_returns_lse_in_model_layout():
    query, key, value = _model_layout_qkv(seed=3)

    out, lse = patches._fixed_cudnn_attention_forward_op(
        ctx=None, query=query, key=key, value=value, return_lse=True, _save_ctx=False
    )

    batch, seq, heads, head_dim = query.shape
    assert out.shape == (batch, seq, heads, head_dim)
    # cuDNN may append a trailing singleton dim to the log-sum-exp.
    assert lse.shape[:3] == (batch, seq, heads)


@requires_cuda
def test_fixed_cudnn_forward_skips_ctx_when_save_ctx_false():
    query, key, value = _model_layout_qkv(seed=4)
    ctx = _FakeFunctionCtx()

    patches._fixed_cudnn_attention_forward_op(ctx, query, key, value, _save_ctx=False)

    assert not hasattr(ctx, "saved_tensors")
