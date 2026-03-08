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

"""Flash Attention packing support via monkey-patching.

When ``attn_implementation="flash_attention_2"`` and neat packing is enabled,
the collater produces an **indexed** attention mask ``[B, S]`` where each
position contains the 1-based document index (0 = padding).  For example::

    [1, 1, 2, 2, 2, 0]   # 2 tokens in doc 1, 3 in doc 2, 1 padding

To make HuggingFace's flash attention path use ``flash_attn_varlen_func``
with per-document ``cu_seqlens``, we monkey-patch two functions:

1. ``transformers.modeling_flash_attention_utils._get_unpad_data`` — extracts
   per-document sequence lengths from the indexed mask and builds cu_seqlens.
2. ``transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask`` —
   returns the 2D indexed mask as-is, bypassing 4D mask creation.

This is the same approach used by LlamaFactory.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_seqlens_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract per-document sequence lengths from an indexed attention mask.

    Args:
        attention_mask: ``[B, S]`` integer tensor where each position contains
            the 1-based document index (0 = padding).

    Returns:
        1D tensor of all individual document lengths across the batch.

    Example::

        >>> get_seqlens_in_batch(torch.tensor([[1, 1, 2, 2, 2, 0],
        ...                                    [1, 2, 2, 3, 3, 3]]))
        tensor([2, 3, 1, 2, 3])
    """
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts = torch.zeros((bsz, max_num), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)

    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]
    return seqlens


def get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prepare indices and cu_seqlens for ``flash_attn_varlen_func``.

    This is a drop-in replacement for
    ``transformers.modeling_flash_attention_utils._get_unpad_data``
    that handles **indexed** attention masks (values 1, 2, 3, …) instead of
    binary (0/1) masks.  Each unique non-zero value is treated as a separate
    document, so ``flash_attn_varlen_func`` applies causal attention
    *within* each document without cross-document attention.

    Returns:
        indices: Indices of non-padding tokens from the flattened sequence.
        cu_seqlens: Cumulative sequence lengths (starts from 0).
        max_seqlen_in_batch: Largest document length in the batch.

    Example::

        >>> get_unpad_data(torch.tensor([[1, 1, 2, 2, 2, 0],
        ...                              [1, 2, 2, 3, 3, 3]]))
        (tensor([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]),
         tensor([ 0,  2,  5,  6,  8, 11], dtype=torch.int32),
         3)
    """
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def configure_packing(attn_implementation: str = "sdpa") -> None:
    """Apply monkey-patches for flash attention packing support.

    Call this **after** the model is created and **before** training starts.
    Only patches when ``attn_implementation == "flash_attention_2"``.

    Args:
        attn_implementation: The attention implementation used by the model.
    """
    if attn_implementation != "flash_attention_2":
        return

    # Patch 1: Replace _get_unpad_data to handle indexed attention masks
    import transformers.modeling_flash_attention_utils

    transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data

    # Patch 2: Replace create_causal_mask for Qwen3-VL models
    # When using flash_attention_2 with packing, we pass the indexed mask
    # directly to the attention layer (bypassing 4D mask creation).
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl

        def _create_causal_mask(
            config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, **kwargs
        ):
            return attention_mask

        modeling_qwen3_vl.create_causal_mask = _create_causal_mask
    except ImportError:
        pass

    try:
        from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

        def _create_causal_mask_moe(
            config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, **kwargs
        ):
            return attention_mask

        modeling_qwen3_vl_moe.create_causal_mask = _create_causal_mask_moe
    except ImportError:
        pass

    logger.info("Configured flash attention packing: patched _get_unpad_data and create_causal_mask.")
