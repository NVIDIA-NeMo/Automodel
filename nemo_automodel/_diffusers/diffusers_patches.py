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

"""Interim patches for diffusers context-parallel training bugs.

diffusers <= 0.39 ships broken ops for the ``_native_flash`` and
``_native_cudnn`` attention backends on the templated (context-parallel)
autograd path; plain inference and non-CP training use SDPA directly and are
unaffected:

1. Double-transposed key/value in backward (both backends): the forward ops
   save query/key/value already transposed to the kernel layout
   [batch, heads, seq, dim], but the backward ops transpose key/value a second
   time, so the backward kernel receives key/value as [batch, seq, heads, dim]
   while query is [batch, heads, seq, dim]. ``_native_flash`` fails loudly
   ("Number of heads in key/value must divide number of heads in query");
   ``_native_cudnn`` accepts the mismatched shapes as a degenerate GQA case
   and produces NaN/garbage gradients instead of raising.
2. Missing log-sum-exp in the cuDNN forward: ``_cudnn_attention_forward_op``
   calls the cuDNN kernel with ``compute_log_sumexp=return_lse``, which is
   False during training, so it saves ``lse=None`` to the autograd context and
   the backward kernel then computes garbage from it.

The patches below are feature-detected against the buggy source lines, so they
automatically become no-ops once the upstream fixes land.
"""

import inspect
import logging

import torch

logger = logging.getLogger(__name__)

_BUGGY_KV_TRANSPOSE_MARKER = "key = key.transpose(1, 2).contiguous()"
_BUGGY_CUDNN_LSE_MARKER = "compute_log_sumexp=return_lse"
_APPLIED_PATCHES: set[str] = set()


def _fixed_native_flash_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrected backward op for diffusers' ``_native_flash`` attention backend.

    Identical to diffusers' ``_native_flash_attention_backward_op`` except it
    does not re-transpose key/value: the forward op already saved query, key,
    and value in the kernel layout, so only ``grad_out`` needs transposing in
    and the gradients transposing back out.

    Args:
        ctx: Autograd context. ``ctx.saved_tensors`` holds query, key, value of
            shape [batch, heads, seq, head_dim] (kernel layout, saved
            pre-transposed by the forward op), out and lse in the same kernel
            layout, plus the flash bookkeeping tensors (cum_seq_q, cum_seq_k,
            philox_seed, philox_offset).
        grad_out: Gradient w.r.t. the attention output, of shape
            [batch, seq, heads, head_dim] (model layout).

    Returns:
        Tuple of (grad_query, grad_key, grad_value), each of shape
        [batch, seq, heads, head_dim] (model layout, matching the tensors the
        caller passed to the attention op).
    """
    query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

    grad_out = grad_out.transpose(1, 2).contiguous()

    grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp=lse,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=ctx.max_q,
        max_k=ctx.max_k,
        dropout_p=ctx.dropout_p,
        is_causal=ctx.is_causal,
        scale=ctx.scale,
    )
    grad_query, grad_key, grad_value = (x.transpose(1, 2).contiguous() for x in (grad_query, grad_key, grad_value))

    return grad_query, grad_key, grad_value


def _fixed_cudnn_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config=None,
):
    """Corrected forward op for diffusers' ``_native_cudnn`` attention backend.

    Identical to diffusers' ``_cudnn_attention_forward_op`` except it computes
    the log-sum-exp whenever the autograd context is saved (``_save_ctx``), not
    only when the caller requests it: the backward kernel requires a valid
    ``lse`` and produces garbage from the ``None`` the buggy forward saves.

    Args:
        ctx: Autograd context populated for the paired backward op.
        query: Tensor of shape [batch, seq, heads, head_dim] (model layout).
        key: Tensor of shape [batch, seq_kv, heads, head_dim] (model layout).
        value: Tensor of shape [batch, seq_kv, heads, head_dim] (model layout).
        attn_mask: Optional attention bias broadcastable to
            [batch, heads, seq, seq_kv].
        dropout_p: Attention dropout probability.
        is_causal: Whether to apply a causal mask.
        scale: Optional softmax scale override.
        enable_gqa: Not supported by cuDNN attention; must be False.
        return_lse: Whether to also return the log-sum-exp.
        _save_ctx: Whether to save tensors on ``ctx`` for backward.
        _parallel_config: Parallel config threaded through by the templated
            context-parallel path (unused here).

    Returns:
        Attention output of shape [batch, seq, heads, head_dim], or a tuple of
        (output, lse) with lse of shape [batch, seq, heads] when ``return_lse``.
    """
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for cuDNN attention.")

    tensors_to_save = ()

    # Contiguous is a must here! Calling cuDNN backend with aten ops produces incorrect results
    # if the input tensors are not contiguous.
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    tensors_to_save += (query, key, value)

    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_mask,
            compute_log_sumexp=return_lse or _save_ctx,
            dropout_p=dropout_p,
            is_causal=is_causal,
            return_debug_mask=False,
            scale=scale,
        )
    )

    tensors_to_save += (out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset)
    if _save_ctx:
        ctx.save_for_backward(*tensors_to_save)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.attn_mask = attn_mask
        ctx.max_q = max_q
        ctx.max_k = max_k

    out = out.transpose(1, 2).contiguous()
    if return_lse:
        lse = lse.transpose(1, 2).contiguous()
        return out, lse
    return out


def _fixed_cudnn_attention_backward_op(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrected backward op for diffusers' ``_native_cudnn`` attention backend.

    Identical to diffusers' ``_cudnn_attention_backward_op`` except it does not
    re-transpose key/value: the forward op already saved query, key, and value
    in the kernel layout, so only ``grad_out`` needs transposing in and the
    gradients transposing back out. Unpatched, the shape mismatch makes cuDNN
    produce NaN/garbage gradients (it does not always raise).

    Args:
        ctx: Autograd context. ``ctx.saved_tensors`` holds query, key, value of
            shape [batch, heads, seq, head_dim] (kernel layout, saved
            pre-transposed by the forward op), out and lse in the same kernel
            layout, plus the cuDNN bookkeeping tensors (cum_seq_q, cum_seq_k,
            philox_seed, philox_offset).
        grad_out: Gradient w.r.t. the attention output, of shape
            [batch, seq, heads, head_dim] (model layout).

    Returns:
        Tuple of (grad_query, grad_key, grad_value), each of shape
        [batch, seq, heads, head_dim] (model layout, matching the tensors the
        caller passed to the attention op).
    """
    query, key, value, out, lse, cum_seq_q, cum_seq_k, philox_seed, philox_offset = ctx.saved_tensors

    grad_out = grad_out.transpose(1, 2).contiguous()

    grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp=lse,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        attn_bias=ctx.attn_mask,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=ctx.max_q,
        max_k=ctx.max_k,
        dropout_p=ctx.dropout_p,
        is_causal=ctx.is_causal,
        scale=ctx.scale,
    )
    grad_query, grad_key, grad_value = (x.transpose(1, 2).contiguous() for x in (grad_query, grad_key, grad_value))

    return grad_query, grad_key, grad_value


def _apply_op_patch(op_name: str, fixed_op, buggy_marker: str) -> bool:
    """Replace a diffusers attention op with a fixed version if it is buggy.

    Feature-detects the bug by inspecting the installed function's source for
    ``buggy_marker``; if the marker is absent (upstream fixed, or the function
    was already patched), this is a no-op. Idempotent per op name.

    Args:
        op_name: Attribute name of the op on
            ``diffusers.models.attention_dispatch``.
        fixed_op: Replacement function with the same signature.
        buggy_marker: Source line identifying the buggy implementation.

    Returns:
        True if the patch was applied (now or by a previous call), False if the
        installed diffusers does not need it.
    """
    if op_name in _APPLIED_PATCHES:
        return True

    from diffusers.models import attention_dispatch

    current = getattr(attention_dispatch, op_name, None)
    if current is None:
        logger.info("[CP patch] diffusers has no %s; nothing to patch", op_name)
        return False

    try:
        source = inspect.getsource(current)
    except (OSError, TypeError):
        logger.warning("[CP patch] Could not inspect %s source; not patching", op_name)
        return False

    if buggy_marker not in source:
        logger.info("[CP patch] diffusers %s already fixed upstream; not patching", op_name)
        return False

    setattr(attention_dispatch, op_name, fixed_op)
    _APPLIED_PATCHES.add(op_name)
    logger.info(
        "[CP patch] Applied fix for diffusers %s (broken on the context-parallel training path in diffusers<=0.39)",
        op_name,
    )
    return True


def apply_native_flash_backward_patch() -> bool:
    """Replace diffusers' buggy ``_native_flash`` backward op if present.

    Returns:
        True if the patch was applied (now or by a previous call), False if the
        installed diffusers does not need it.
    """
    return _apply_op_patch(
        "_native_flash_attention_backward_op",
        _fixed_native_flash_attention_backward_op,
        _BUGGY_KV_TRANSPOSE_MARKER,
    )


def apply_cudnn_attention_patch() -> bool:
    """Replace diffusers' buggy ``_native_cudnn`` forward and backward ops if present.

    The cuDNN backend has two independent bugs on the context-parallel training
    path (missing log-sum-exp in forward, double-transposed key/value in
    backward); both must be patched for correct gradients.

    Returns:
        True if either patch was applied (now or by a previous call), False if
        the installed diffusers does not need them.
    """
    forward_applied = _apply_op_patch(
        "_cudnn_attention_forward_op",
        _fixed_cudnn_attention_forward_op,
        _BUGGY_CUDNN_LSE_MARKER,
    )
    backward_applied = _apply_op_patch(
        "_cudnn_attention_backward_op",
        _fixed_cudnn_attention_backward_op,
        _BUGGY_KV_TRANSPOSE_MARKER,
    )
    return forward_applied or backward_applied
