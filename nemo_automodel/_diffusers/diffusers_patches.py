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

diffusers <= 0.39 ships a broken backward op for the ``_native_flash``
attention backend: the forward op saves query/key/value already transposed to
the kernel layout, but the backward op transposes key/value a second time, so
the flash backward kernel receives key/value as [batch, seq, heads, dim] while
query is [batch, heads, seq, dim] and fails with "Number of heads in key/value
must divide number of heads in query". The bug only triggers on the templated
(context-parallel) autograd path; plain inference and non-CP training use SDPA
directly and are unaffected.

The patch below is feature-detected against the buggy source line, so it
automatically becomes a no-op once the upstream fix lands.
"""

import inspect
import logging

import torch

logger = logging.getLogger(__name__)

_BUGGY_SOURCE_MARKER = "key = key.transpose(1, 2).contiguous()"
_PATCH_APPLIED = False


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


def apply_native_flash_backward_patch() -> bool:
    """Replace diffusers' buggy ``_native_flash`` backward op if present.

    Feature-detects the double-transpose bug by inspecting the installed
    function's source; if the buggy line is absent (upstream fixed, or the
    function was already patched), this is a no-op. Idempotent.

    Returns:
        True if the patch was applied (now or by a previous call), False if the
        installed diffusers does not need it.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    from diffusers.models import attention_dispatch

    current = getattr(attention_dispatch, "_native_flash_attention_backward_op", None)
    if current is None:
        logger.info("[CP patch] diffusers has no _native_flash_attention_backward_op; nothing to patch")
        return False

    try:
        source = inspect.getsource(current)
    except (OSError, TypeError):
        logger.warning("[CP patch] Could not inspect _native_flash_attention_backward_op source; not patching")
        return False

    if _BUGGY_SOURCE_MARKER not in source:
        logger.info("[CP patch] diffusers _native_flash backward already fixed upstream; not patching")
        return False

    attention_dispatch._native_flash_attention_backward_op = _fixed_native_flash_attention_backward_op
    _PATCH_APPLIED = True
    logger.info(
        "[CP patch] Applied fix for diffusers _native_flash_attention_backward_op double-transpose "
        "(broken backward on the context-parallel path in diffusers<=0.39)"
    )
    return True
