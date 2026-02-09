# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Kernel and attention patching utilities.

Functions for SDPA, Liger-kernel, and attention-implementation overrides.
These are stateless helpers used during model construction.
"""

import functools
import inspect
import logging
import types

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.shared.import_utils import safe_import

HAS_LIGER_KERNEL, liger_kernel_trf = safe_import("liger_kernel.transformers")
HAS_FA, _ = safe_import("flash_attn")
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FA else "sdpa"

logger = logging.getLogger(__name__)


def _assert_same_signature(original, patched):
    """
    Raise AssertionError if the two call signatures differ.
    """
    sig_orig = inspect.signature(original)
    sig_patch = inspect.signature(patched)

    if sig_orig != sig_patch:
        raise AssertionError(f"Signature mismatch:\n  original: {sig_orig}\n  patched : {sig_patch}")


def _patch_attention(obj, sdpa_method=None):
    """
    Wrap the `forward` method of `obj` in an `sdap_kernel` context manager.

    Args:
        obj: Any object with a `.forward(*args, **kwargs)` method.
        sdpa_method (list[SDPBackend], optional): Ordered list of SDPBackend
            implementations to attempt. If None, defaults to
            [CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH].

    Returns:
        The same `obj` with its `.forward` method patched.
    """
    if sdpa_method is None:
        sdpa_method = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    orig_forward = obj.forward

    def patch_method(method):
        func = method.__func__

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with sdpa_kernel(sdpa_method):
                return func(self, *args, **kwargs)

        wrapper.__doc__ = "SDPA kernel patch\n" + inspect.getdoc(method)
        return types.MethodType(wrapper, method.__self__)  # re-bind

    obj.forward = patch_method(obj.forward)
    # runtime check
    _assert_same_signature(orig_forward, obj.forward)

    logger.info("Patched model with SDPA method= %s", sdpa_method)
    return obj


def _patch_liger_kernel(model):
    """
    Patches a model with liger-kernel and sdpa_kernel

    Args:
        model (nn.Module): the model to patch
        use_liger_kernel (bool): Applies liger-kernel to model Default True.
        use_sdpa_patching (bool): Enables model patching with SDPA kernel optim. Default True.
        sdpa_method (list[SDPBackend], optional): Ordered list of SDPBackend
            implementations to attempt. If None, defaults to
            [CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH].
    Returns:
        nn.Module: the patched model
    """
    if not HAS_LIGER_KERNEL:
        logger.warning("Asked to use Liger Kernel, but could not import")
        return model

    # Unit tests may pass lightweight mocks; skip patching in that case.
    # (The wrapper logic itself is tested separately by patching this function.)
    if not isinstance(model, torch.nn.Module):
        logger.warning("Skipping Liger Kernel patch for non-nn.Module model: %s", type(model))
        return model

    try:
        liger_kernel_trf._apply_liger_kernel_to_instance(model=model)
        logger.info("Applied liger-kernel to model")
        return model
    except Exception:
        logger.warning("Failed to apply liger-kernels to model; falling back to eager")
        del model
        raise RuntimeError("Failed to patch model")


def _get_next_fallback_attn(attn_implementation: str) -> str:
    """
    Get the next attention implementation in the priority list, in reverse order.

    If a model does not support a given attention implementation, the next
    implementation in the priority list is returned.

    If the current attention implementation is not in the priority list, it uses eager.

    Args:
        attn_implementation (str): The current attention implementation.

    Returns:
        str: The next attention implementation in the priority list.
    """
    priorities = [
        "eager",
        "sdpa",
        "flash_attention_2",
        "flash_attention_3",
    ]
    if attn_implementation in priorities:
        pos = priorities.index(attn_implementation)
        return priorities[max(0, pos - 1)]
    else:
        return priorities[0]


def _apply_preload_overrides(tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel):
    """
    Compute final attention implementation and liger-kernel flag based on TP/CP and packed sequence constraints.
    """
    if (tp_size > 1 or cp_size > 1):
        logger.info("Disabling Liger kernel with TP ({}) or CP ({})".format(tp_size, cp_size))
        use_liger_kernel = False

    if cp_size > 1:
        attn_implementation = "sdpa"
        logger.warning("Packed sequence is supported only with SDPA. Setting model's attn_implementation to sdpa")

    if has_packed_sequence:
        if cp_size == 1:
            assert HAS_FA, "Flash Attention is not available"
            attn_implementation = "flash_attention_2"
            logger.warning(
                "Packed sequence is supported only with Flash Attention. "
                "Setting model's attn_implementation to flash_attention_2"
            )
        else:
            # TODO: support packed sequence with CP size > 1
            raise ValueError("Packed sequence is only supported with CP size 1")
    return attn_implementation, use_liger_kernel


def _verify_sdpa_support(model, cp_size):
    """
    Validate SDPA support when CP is enabled for HF models.
    """
    if cp_size > 1:
        if getattr(model, "_supports_sdpa", False) is False:
            raise ValueError("Model does not support SDPA required for context parallelism")
