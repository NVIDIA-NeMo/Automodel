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


def is_indexed_packed_mask(attention_mask: torch.Tensor | None) -> bool:
    """Return ``True`` iff ``attention_mask`` is an Automodel-style indexed packing mask.

    The Automodel ``neat_packed_vlm_collater`` (and the LLM equivalent) encode
    packed-sample boundaries by marking document ``i`` (1-based) with the
    integer ``i`` and using ``0`` for padding (e.g. ``[1, 1, 1, 2, 2, 3, 3, 0, 0]``).
    Any value greater than ``1`` is therefore a sufficient signal that two or
    more documents are packed into the same row.  A standard 0/1 attention mask
    never has values > 1.
    """
    if attention_mask is None:
        return False
    if attention_mask.dtype == torch.bool:
        return False
    if attention_mask.dim() != 2:
        return False
    return bool((attention_mask > 1).any().item())


def _passthrough_create_causal_mask(
    config=None,
    input_embeds=None,
    inputs_embeds=None,
    attention_mask=None,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
    **kwargs,
):
    """Replacement for ``create_causal_mask`` that passes through packed masks.

    FA2 handles masking internally, so always pass through.  For non-FA2
    backends, pass through packed masks but delegate normal 2D masks to HF.
    """
    if config is not None and getattr(config, "_attn_implementation", None) == "flash_attention_2":
        return attention_mask

    if attention_mask is not None:
        if attention_mask.ndim == 4:
            return attention_mask
        if attention_mask.max() > 1:
            return attention_mask

    from transformers.masking_utils import create_causal_mask

    embeds = inputs_embeds if inputs_embeds is not None else input_embeds
    return create_causal_mask(
        config=config,
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        **kwargs,
    )


def get_attn_implementation(cfg_model):
    """Determine the attention backend from model config.

    Custom models store it in ``backend.attn``; HF models use ``attn_implementation``.
    """
    if cfg_model is not None and hasattr(cfg_model, "backend") and hasattr(cfg_model.backend, "attn"):
        return cfg_model.backend.attn
    if cfg_model is not None:
        return cfg_model.get("attn_implementation", "sdpa")
    return "sdpa"


def _transformers_version() -> str:
    """Best-effort transformers version string for shim diagnostics."""
    try:
        import transformers

        return str(transformers.__version__)
    except Exception:  # pragma: no cover - transformers is a hard dependency
        return "unknown"


def _patch_preprocess_mask_arguments_for_packing() -> None:
    """Keep indexed packing masks intact in newer Transformers.

    Transformers 5.x preprocesses 2D attention masks before dispatching
    attention. For flash attention this can coerce integer indexed masks
    (``1, 2, ...`` per packed document) to bool masks, losing the document
    boundaries that ``get_unpad_data`` needs. Preserve non-bool 2D masks for
    FA2/FA3 so the patched flash-attention path can derive per-document
    ``cu_seqlens``.
    """
    try:
        import transformers.masking_utils as masking_utils
    except (ImportError, AttributeError):
        logger.warning(
            "Packed-sequence mask shim not installed: transformers.masking_utils is unavailable "
            "(transformers %s). If this transformers version preprocesses 2D masks before flash "
            "attention, indexed packing masks may be coerced to bool, silently enabling "
            "cross-document attention.",
            _transformers_version(),
        )
        return

    if getattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", False):
        return

    original_preprocess = getattr(masking_utils, "_preprocess_mask_arguments", None)
    if original_preprocess is None:
        logger.warning(
            "Packed-sequence mask shim not installed: transformers.masking_utils has no "
            "_preprocess_mask_arguments (transformers %s). If this transformers version preprocesses "
            "2D masks before flash attention, indexed packing masks may be coerced to bool, silently "
            "enabling cross-document attention.",
            _transformers_version(),
        )
        return

    preprocess_result_len: int | None = None

    def _infer_preprocess_result_len(args, kwargs, inputs_embeds, attention_mask) -> int:
        """Infer the installed Transformers return arity without losing packed masks.

        Calls the original ``_preprocess_mask_arguments`` once with a bool 4D probe
        mask (which it passes through untouched) so the indexed 2D packing mask is
        never handed to it, and measures the length of the returned tuple.

        Args:
            args: Positional arguments of the intercepted call; when present,
                index 2 is the attention mask and is replaced by the probe mask.
            kwargs: Keyword arguments of the intercepted call; used (and probed)
                when the attention mask was passed by keyword.
            inputs_embeds: ``[B, S, H]`` input embeddings tensor used to size the
                probe mask (batch and query length).
            attention_mask: ``[B, S]`` indexed packing mask (1-based document
                index per position, 0 = padding); only its trailing dimension is
                read, as the probe's KV length.

        Returns:
            Length of the tuple returned by the original function, or the
            historical arity 7 when probing is impossible.
        """
        if not isinstance(inputs_embeds, torch.Tensor):
            return 7

        batch_size = inputs_embeds.shape[0] if inputs_embeds.ndim > 0 else 1
        q_length = inputs_embeds.shape[1] if inputs_embeds.ndim > 1 else 1
        kv_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim >= 2
            else q_length
        )
        probe_mask = torch.zeros(
            (batch_size, 1, q_length, kv_length),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )

        patched_args = list(args)
        patched_kwargs = dict(kwargs)
        if len(patched_args) > 2:
            patched_args[2] = probe_mask
        else:
            patched_kwargs["attention_mask"] = probe_mask

        try:
            return len(original_preprocess(*patched_args, **patched_kwargs))
        except Exception:
            logger.warning(
                "Packed-sequence mask shim could not probe the _preprocess_mask_arguments return "
                "arity (transformers %s); assuming the historical arity of 7. If this transformers "
                "version changed the signature, packing masks may be mishandled and cross-document "
                "attention silently enabled.",
                _transformers_version(),
            )
            return 7

    def _patched_preprocess_mask_arguments(*args, **kwargs):
        nonlocal preprocess_result_len
        config = kwargs.get("config", args[0] if len(args) > 0 else None)
        inputs_embeds = kwargs.get(
            "inputs_embeds",
            kwargs.get("input_embeds", args[1] if len(args) > 1 else None),
        )
        attention_mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        attn_impl = getattr(config, "_attn_implementation", None) or getattr(
            config, "_attn_implementation_internal", None
        )
        if (
            attention_mask is not None
            and isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.dtype != torch.bool
            and attn_impl in ("flash_attention_2", "flash_attention_3")
        ):
            if preprocess_result_len is None:
                preprocess_result_len = _infer_preprocess_result_len(args, kwargs, inputs_embeds, attention_mask)
            return (True, attention_mask, *([None] * max(preprocess_result_len - 2, 0)))
        return original_preprocess(*args, **kwargs)

    masking_utils._preprocess_mask_arguments = _patched_preprocess_mask_arguments
    masking_utils._nemo_automodel_packing_preprocess_patched = True


# Model modules whose ``create_causal_mask`` must be patched for neat packing.
# TODO: perhaps its for ALL models.
_PACKING_PATCH_MODULES = [
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "transformers.models.qwen2_vl.modeling_qwen2_vl",
    "transformers.models.qwen3.modeling_qwen3",
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
    _patch_preprocess_mask_arguments_for_packing()

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
