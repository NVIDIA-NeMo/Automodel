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


def _passthrough_create_causal_mask(
    config=None, input_embeds=None, inputs_embeds=None, attention_mask=None,
    cache_position=None, past_key_values=None, position_ids=None, **kwargs
):
    """Replacement for ``create_causal_mask`` that returns the mask unchanged."""
    return attention_mask


def get_attn_implementation(cfg_model):
    """Determine the attention backend from model config.

    Custom models store it in ``backend.attn``; HF models use ``attn_implementation``.
    """
    if cfg_model is not None and hasattr(cfg_model, "backend") and hasattr(cfg_model.backend, "attn"):
        return cfg_model.backend.attn
    if cfg_model is not None:
        return cfg_model.get("attn_implementation", "sdpa")
    return "sdpa"


# Model modules whose ``create_causal_mask`` must be patched for neat packing.
# TODO: perhaps its for ALL models.
_PACKING_PATCH_MODULES = [
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "transformers.models.qwen2_vl.modeling_qwen2_vl",
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
]


def configure_packing(attn_implementation: str = "sdpa") -> None:
    """Apply monkey-patches for packed-sequence training with flash_attention_2.

    Only patches when ``attn_implementation == "flash_attention_2"``.

    Args:
        attn_implementation: The attention implementation used by the model.
    """
    if attn_implementation != "flash_attention_2":
        return

    import sys

    import transformers.modeling_flash_attention_utils

    transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data

    # Each model module imports create_causal_mask into its own namespace at
    # import time, so we must patch each module individually.
    for mod_name in _PACKING_PATCH_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "create_causal_mask"):
            mod.create_causal_mask = _passthrough_create_causal_mask

    logger.info(
        "Configured packing (%s): patched create_causal_mask in %d model modules.",
        attn_implementation,
        sum(1 for m in _PACKING_PATCH_MODULES if sys.modules.get(m) is not None),
    )
