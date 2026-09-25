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
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

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
    uses_native_fa4: bool = False


@runtime_checkable
class PackingMetadataConsumer(Protocol):
    """Model that needs dataset-constructed metadata for packed recurrent state."""

    requires_packed_sequence_metadata: bool


@runtime_checkable
class PackedMaskConsumer(Protocol):
    """Model that owns masking and consumes compact document IDs."""

    packed_mask_type: PackedMaskType


@runtime_checkable
class NativeFA4Consumer(Protocol):
    """Model whose attention layers invoke Automodel's native FA4 callable."""

    _uses_native_fa4: bool


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
        model: Optional built model, inspected through explicit packed-metadata,
            mask, and native-FA4 consumer capabilities.

    Returns:
        Structural capabilities consumed by dataset packing. Backend names do not
        cross the dataset boundary.
    """
    model = getattr(model, "module", model)
    requires_metadata = isinstance(model, PackingMetadataConsumer) and model.requires_packed_sequence_metadata
    model_mask_type = model.packed_mask_type if isinstance(model, PackedMaskConsumer) else None
    if attn_implementation == "fa4":
        uses_native_fa4 = isinstance(model, NativeFA4Consumer) and model._uses_native_fa4
        return PackingCapabilities(
            packed_mask_type="document_ids",
            requires_packed_sequence_metadata=uses_native_fa4 or requires_metadata,
            patch_transformers=not uses_native_fa4,
            uses_native_fa4=uses_native_fa4,
        )
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
    """Normalize batch-major packed metadata to a single flat token stream.

    Args:
        packed_token_indices: Tensor of shape [batch, sequence] containing
            row-local token indices with ``-1`` padding, or a pre-flattened
            tensor of shape [tokens] containing indices into [batch, sequence].
        cu_seqlens: Tensor of shape [batch, max_documents + 1] containing
            row-local cumulative lengths with ``-1`` padding, or a pre-flattened
            tensor of shape [documents + 1].
        batch_size: Batch dimension of the padded token layout.
        sequence_length: Sequence dimension of the padded token layout.

    Returns:
        Flat token indices of shape [tokens] into the padded [batch, sequence]
        layout and cumulative document lengths of shape [documents + 1].

    Raises:
        ValueError: If the layouts, boundaries, or represented token counts are
            inconsistent.
    """
    if packed_token_indices.ndim == 1 and cu_seqlens.ndim == 1:
        invalid = cu_seqlens.numel() < 2
        if not invalid:
            invalid = bool(
                (
                    (cu_seqlens[0] != 0)
                    | (cu_seqlens[-1] != packed_token_indices.numel())
                    | (cu_seqlens[1:] < cu_seqlens[:-1]).any()
                ).item()
            )
        if invalid:
            raise ValueError("Flat packed sequence metadata must start at zero and cover every token index")
        return packed_token_indices.to(torch.long), cu_seqlens

    if packed_token_indices.shape != (batch_size, sequence_length) or cu_seqlens.ndim != 2:
        raise ValueError(
            "Packed sequence metadata does not match the current [batch, sequence] layout: "
            f"indices={tuple(packed_token_indices.shape)}, cu_seqlens={tuple(cu_seqlens.shape)}, "
            f"batch={batch_size}, sequence={sequence_length}."
        )
    if cu_seqlens.shape[0] != batch_size or cu_seqlens.shape[1] == 0:
        raise ValueError(
            "Packed sequence cumulative lengths must have the same batch dimension as token indices "
            "and at least one boundary per row: "
            f"indices={tuple(packed_token_indices.shape)}, cu_seqlens={tuple(cu_seqlens.shape)}."
        )

    valid_tokens = packed_token_indices >= 0
    valid_boundaries = cu_seqlens >= 0
    boundary_after_padding = valid_boundaries & ((~valid_boundaries).cumsum(dim=1) > 0)
    boundary_counts = valid_boundaries.sum(dim=1)
    last_boundary_indices = (boundary_counts - 1).clamp_min(0).unsqueeze(1)
    last_boundaries = cu_seqlens.gather(1, last_boundary_indices).squeeze(1)
    valid_boundary_pairs = valid_boundaries[:, 1:] & valid_boundaries[:, :-1]
    boundary_deltas = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
    invalid_rows = (
        ((boundary_counts == 0) | (cu_seqlens[:, 0] != 0))
        | (last_boundaries != valid_tokens.sum(dim=1))
        | boundary_after_padding.any(dim=1)
        | ((boundary_deltas < 0) & valid_boundary_pairs).any(dim=1)
    )

    document_lengths = boundary_deltas[valid_boundary_pairs]
    represented_token_count = document_lengths.sum() if document_lengths.numel() else cu_seqlens.new_zeros(())
    invalid = invalid_rows.any() | (represented_token_count != valid_tokens.sum()) | (document_lengths.numel() == 0)
    if bool(invalid.item()):
        raise ValueError(
            "Each packed sequence metadata row must start at zero, be monotonic, and cover its valid tokens"
        )

    row_offsets = torch.arange(batch_size, device=packed_token_indices.device)[:, None] * sequence_length
    flat_indices = (packed_token_indices + row_offsets)[valid_tokens].to(torch.long)
    flat_cu_seqlens = F.pad(torch.cumsum(document_lengths, dim=0, dtype=cu_seqlens.dtype), (1, 0))
    return flat_indices, flat_cu_seqlens


def _flatten_packed_metadata_at_model_entry(
    _module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Flatten packed metadata once after pipeline microbatch splitting.

    Args:
        _module: Native FA4 model receiving the forward call.
        args: Positional model inputs. Tensor entries retain their model-specific
            layouts and are returned unchanged.
        kwargs: Model inputs containing ``packed_token_indices`` of shape
            [batch, sequence] or [tokens] and ``cu_seqlens`` of shape
            [batch, max_documents + 1] or [documents + 1].

    Returns:
        Updated positional and keyword inputs whose packed token indices have
        shape [tokens] and cumulative lengths have shape [documents + 1], or
        ``None`` when neither packed-metadata tensor was supplied.

    Raises:
        ValueError: If only one metadata tensor is supplied or either layout is
            inconsistent with the current microbatch.
    """
    packed_token_indices = kwargs.get("packed_token_indices")
    cu_seqlens = kwargs.get("cu_seqlens")
    if packed_token_indices is None and cu_seqlens is None:
        return None
    if not isinstance(packed_token_indices, torch.Tensor) or not isinstance(cu_seqlens, torch.Tensor):
        raise ValueError("Native FA4 packed_token_indices and cu_seqlens must be tensors supplied together")

    if packed_token_indices.ndim == 2:
        batch_size, sequence_length = packed_token_indices.shape
    else:
        batch_size, sequence_length = 1, packed_token_indices.numel()
    flat_indices, flat_cu_seqlens = flatten_packed_sequence_metadata(
        packed_token_indices,
        cu_seqlens,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    updated_kwargs = dict(kwargs)
    updated_kwargs["packed_token_indices"] = flat_indices
    updated_kwargs["cu_seqlens"] = flat_cu_seqlens
    return args, updated_kwargs


def _install_native_fa4_metadata_hook(model: torch.nn.Module) -> None:
    """Install the once-per-forward native FA4 metadata normalizer."""
    model = getattr(model, "module", model)
    if getattr(model, "_native_fa4_metadata_hook_handle", None) is not None:
        return
    handle = model.register_forward_pre_hook(
        _flatten_packed_metadata_at_model_entry,
        with_kwargs=True,
    )
    setattr(model, "_native_fa4_metadata_hook_handle", handle)


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
    hf_implementation = _model_attn_implementation(model)
    uses_native_fa4 = isinstance(model, NativeFA4Consumer) and model._uses_native_fa4
    if hf_implementation == "flash_attention_4" and not uses_native_fa4:
        return hf_implementation
    if isinstance(backend, AttentionBackendSelection):
        if backend.attn == "fa4" and not uses_native_fa4:
            return hf_implementation or "sdpa"
        return backend.attn
    return hf_implementation or "sdpa"


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
    if capabilities.uses_native_fa4:
        if model is None:
            raise ValueError("Native FA4 packing requires the built model")
        _install_native_fa4_metadata_hook(model)
        return capabilities
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
