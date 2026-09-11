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

"""Model-side Flash Attention packing support via monkey-patching.

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
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch

from nemo_automodel.components.models.common.utils import AttentionBackend

logger = logging.getLogger(__name__)

_FLASH_ATTN_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")

PackedMaskType = Literal["block_causal", "document_ids"]


@dataclass(frozen=True)
class PackingCapabilities:
    """Model-owned requirements for dataset packing and model adaptation."""

    packed_mask_type: PackedMaskType
    requires_packed_sequence_metadata: bool = False
    patch_transformers: bool = False


@runtime_checkable
class PackingMetadataConsumer(Protocol):
    """Model that needs dataset-constructed metadata for packed recurrent state."""

    requires_packed_sequence_metadata: bool


@runtime_checkable
class PackedMaskConsumer(Protocol):
    """Model that owns masking and consumes compact document IDs."""

    packed_mask_type: PackedMaskType


@runtime_checkable
class AttentionBackendSelection(Protocol):
    """Typed attention selection carried by a built custom model."""

    @property
    def attn(self) -> AttentionBackend:
        """Selected native attention backend."""
        ...


class UnpadData(Protocol):
    """Dataset-owned mask conversion accepted by the model-side HF adapter."""

    def __call__(self, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Convert a mask of shape [batch, sequence] to flat varlen metadata."""
        ...


def get_packing_capabilities(
    attn_implementation: str,
    *,
    model: torch.nn.Module | None = None,
) -> PackingCapabilities:
    """Map a model attention implementation to semantic packed-data requirements.

    Args:
        attn_implementation: Attention implementation resolved from the built model.
        model: Optional built model, inspected only through
            :class:`PackingMetadataConsumer`.

    Returns:
        Structural capabilities consumed by dataset packing. Backend names do not
        cross the dataset boundary.
    """
    model = getattr(model, "module", model)
    requires_metadata = isinstance(model, PackingMetadataConsumer) and model.requires_packed_sequence_metadata
    model_mask_type = model.packed_mask_type if isinstance(model, PackedMaskConsumer) else None
    if attn_implementation == "fa4":
        return PackingCapabilities(packed_mask_type="document_ids", requires_packed_sequence_metadata=True)
    if attn_implementation in _FLASH_ATTN_IMPLEMENTATIONS:
        return PackingCapabilities(
            packed_mask_type="document_ids",
            requires_packed_sequence_metadata=requires_metadata,
            patch_transformers=True,
        )
    return PackingCapabilities(
        packed_mask_type=model_mask_type or "block_causal",
        requires_packed_sequence_metadata=requires_metadata,
    )


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


def flatten_packed_sequence_metadata(
    packed_token_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt batch-major dataset metadata to one consumer-local flat stream.

    The dataset representation keeps a leading batch axis so pipeline schedules
    can split it safely. Recurrent model kernels consume a single flattened
    token stream, so the model converts only its current microbatch here.
    """
    if packed_token_indices.ndim == 1 and cu_seqlens.ndim == 1:
        if (
            cu_seqlens.numel() < 2
            or int(cu_seqlens[0].item()) != 0
            or int(cu_seqlens[-1].item()) != packed_token_indices.numel()
        ):
            raise ValueError("Flat packed sequence metadata must start at zero and cover every token index")
        return packed_token_indices, cu_seqlens
    if packed_token_indices.shape != (batch_size, sequence_length) or cu_seqlens.ndim != 2:
        raise ValueError(
            "Packed sequence metadata does not match the current [batch, sequence] layout: "
            f"indices={tuple(packed_token_indices.shape)}, cu_seqlens={tuple(cu_seqlens.shape)}, "
            f"batch={batch_size}, sequence={sequence_length}."
        )

    valid = packed_token_indices >= 0
    row_offsets = torch.arange(batch_size, device=packed_token_indices.device)[:, None] * sequence_length
    flat_indices = (packed_token_indices + row_offsets)[valid].to(torch.long)
    lengths: list[torch.Tensor] = []
    for row_idx, row in enumerate(cu_seqlens):
        boundaries = row[row >= 0]
        if boundaries.numel() and (
            int(boundaries[0].item()) != 0
            or int(boundaries[-1].item()) != int(valid[row_idx].sum().item())
            or bool((boundaries[1:] < boundaries[:-1]).any().item())
        ):
            raise ValueError("Each packed sequence metadata row must start at zero and cover its valid tokens")
        if boundaries.numel() > 1:
            lengths.append(boundaries[1:] - boundaries[:-1])
    if not lengths:
        raise ValueError("Packed sequence metadata must describe at least one document")
    document_lengths = torch.cat(lengths)
    flat_cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(document_lengths, dim=0, dtype=cu_seqlens.dtype),
        (1, 0),
    )
    if int(flat_cu_seqlens[-1].item()) != flat_indices.numel():
        raise ValueError("Packed token indices and cumulative lengths describe different token counts")
    return flat_indices, flat_cu_seqlens


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

    Flash attention (FA2/FA3/FA4) handles masking internally, so always pass
    through.  For other backends, pass through packed masks but delegate
    normal 2D masks to HF.
    """
    if config is not None and getattr(config, "_attn_implementation", None) in _FLASH_ATTN_IMPLEMENTATIONS:
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


def _model_attn_implementation(model: torch.nn.Module) -> str | None:
    """Return the packing-relevant attention backend an already-built model runs with.

    ``model.config._attn_implementation`` is a Transformers *dispatch key*, whose
    vocabulary is wider than the mask layouts packing knows about: when flash
    attention is requested but only the ``kernels`` package provides it,
    Transformers records a kernels-hub id instead of the mainline name. Those ids
    are mapped back so a model genuinely running varlen flash attention is packed
    as such. Other live string keys are returned unchanged and therefore use
    packing's conservative block-causal default.
    """
    # DDP does not proxy attribute access to the model it wraps, so read through it.
    model = getattr(model, "module", model)
    attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if attn_implementation in _FLASH_ATTN_IMPLEMENTATIONS or attn_implementation in ("sdpa", "eager"):
        return attn_implementation
    try:
        from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK
    except ImportError:
        return attn_implementation if isinstance(attn_implementation, str) else None
    for mainline, kernel_id in FLASH_ATTN_KERNEL_FALLBACK.items():
        if kernel_id == attn_implementation:
            return mainline
    return attn_implementation if isinstance(attn_implementation, str) else None


def get_model_attn_implementation(model: torch.nn.Module) -> str:
    """Return the attention implementation used by a built model.

    Custom models expose a typed ``backend.attn``. Hugging Face models record
    their resolved dispatch key on ``model.config``; this reflects preload and
    fallback decisions that are intentionally absent from the recipe config.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected a built torch.nn.Module, got {type(model).__name__}")
    model = getattr(model, "module", model)
    backend = getattr(model, "backend", None)
    if isinstance(backend, AttentionBackendSelection):
        return backend.attn
    return _model_attn_implementation(model) or "sdpa"


def _patch_preprocess_mask_arguments_for_packing() -> None:
    """Keep indexed packing masks intact for the supported FA2 path.

    Transformers 5.x preprocesses 2D attention masks before dispatching
    attention. For flash attention this can coerce integer indexed masks
    (``1, 2, ...`` per packed document) to bool masks, losing the document
    boundaries that ``get_unpad_data`` needs. Preserve indexed 2D masks for
    FA2 so the patched flash-attention path can derive per-document
    ``cu_seqlens``. Validate the private Transformers contract before installing
    the shim so an incompatible dependency fails instead of silently enabling
    cross-document attention.
    """
    import transformers

    try:
        import transformers.masking_utils as masking_utils
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot enable FA2 neat packing because transformers.masking_utils is unavailable "
            f"in transformers {transformers.__version__}. Refusing to continue because losing "
            "indexed mask values would enable cross-document attention."
        ) from exc

    if getattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", False):
        return

    original_preprocess = getattr(masking_utils, "_preprocess_mask_arguments", None)
    if original_preprocess is None:
        raise RuntimeError(
            "Cannot enable FA2 neat packing because transformers.masking_utils has no "
            f"_preprocess_mask_arguments in transformers {transformers.__version__}. Refusing to "
            "continue because losing indexed mask values would enable cross-document attention."
        )

    # A 4D mask takes Transformers' immediate pass-through branch, so this
    # constant-size probe verifies the private call and return contract without
    # allocating an O(sequence^2) tensor for a real long-context batch.
    probe_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
    try:
        preprocess_result_template = original_preprocess(
            config=None,
            inputs_embeds=torch.zeros((1, 1, 1)),
            attention_mask=probe_mask,
            past_key_values=None,
            position_ids=None,
            layer_idx=None,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot enable FA2 neat packing because transformers "
            f"{transformers.__version__} has an incompatible _preprocess_mask_arguments signature. "
            "Refusing to continue because losing indexed mask values would enable cross-document attention."
        ) from exc

    if (
        not isinstance(preprocess_result_template, tuple)
        or len(preprocess_result_template) < 2
        or preprocess_result_template[0] is not True
        or preprocess_result_template[1] is not probe_mask
    ):
        raise RuntimeError(
            "Cannot enable FA2 neat packing because transformers "
            f"{transformers.__version__} returned an incompatible _preprocess_mask_arguments "
            "early-exit result. Refusing to continue because losing indexed mask values would "
            "enable cross-document attention."
        )

    def _patched_preprocess_mask_arguments(*args, **kwargs):
        """Preserve indexed masks while matching the installed private HF API.

        Args:
            *args: Positional HF arguments. When present, index 1 is an input
                tensor of shape [batch, sequence, hidden] and index 2 is an
                attention mask of shape [batch, sequence].
            **kwargs: Keyword form of the same HF arguments.

        Returns:
            Tuple matching the installed Transformers preprocessing result. For
            indexed FA2 masks, the first entries are ``True`` and the unchanged
            mask tensor of shape [batch, sequence].
        """
        config = kwargs.get("config", args[0] if len(args) > 0 else None)
        attention_mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        attn_impl = getattr(config, "_attn_implementation", None) or getattr(
            config, "_attn_implementation_internal", None
        )
        if attn_impl in _FLASH_ATTN_IMPLEMENTATIONS and is_indexed_packed_mask(attention_mask):
            return (
                preprocess_result_template[0],
                attention_mask,
                *preprocess_result_template[2:],
            )
        return original_preprocess(*args, **kwargs)

    masking_utils._preprocess_mask_arguments = _patched_preprocess_mask_arguments
    masking_utils._nemo_automodel_packing_preprocess_patched = True


# Model modules whose ``create_causal_mask`` must be patched for neat packing.
# TODO: perhaps its for ALL models.
_PACKING_PATCH_MODULES = [
    "transformers.models.llama.modeling_llama",
    "transformers.models.qwen3.modeling_qwen3",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "transformers.models.qwen2_vl.modeling_qwen2_vl",
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
]


def configure_packing(
    attn_implementation: str,
    *,
    model: torch.nn.Module | None = None,
    unpad_data: UnpadData | None = None,
) -> PackingCapabilities:
    """Configure the model consumer and return its dataset packing contract.

    Hugging Face flash-attention variants require private Transformers adapters
    for the indexed document map. Native consumers receive explicit metadata and
    need no patch. The conversion callable is injected by the recipe so this
    model component never imports the dataset implementation.

    Args:
        attn_implementation: The attention implementation used by the model.
        model: Optional built model declaring additional packed-metadata needs.
        unpad_data: Dataset-owned callable that converts an indexed mask of shape
            [batch, sequence] to flat indices, cumulative lengths, and maximum
            sequence length.

    Returns:
        Structural packed-data requirements for the dataset collater.

    Raises:
        ValueError: If a Transformers adapter is required without ``unpad_data``.
    """
    capabilities = get_packing_capabilities(attn_implementation, model=model)
    if not capabilities.patch_transformers:
        return capabilities
    if unpad_data is None:
        raise ValueError("Hugging Face flash-attention packing requires a dataset-owned unpad_data callable")

    import sys

    import transformers.modeling_flash_attention_utils

    _patch_preprocess_mask_arguments_for_packing()
    transformers.modeling_flash_attention_utils._get_unpad_data = unpad_data

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
    return capabilities
