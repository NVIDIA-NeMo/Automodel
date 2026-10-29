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

"""Model-derived packing contracts and native metadata normalization.

HF flash attention consumes the public varlen kwargs built by dataset collaters.
Native FA4 and recurrent consumers use separate unpadded document metadata.
Neither path modifies Transformers globals.
"""

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common.utils import AttentionBackend
from nemo_automodel.shared.packed_seq import (
    FLASH_VARLEN_KWARGS,
    indexed_mask_to_4d_block_causal,
    packed_seq_params_from_doc_ids,
    to_flash_attention_kwargs,
)

logger = logging.getLogger(__name__)

_FLASH_ATTN_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")

PackedMaskType = Literal["block_causal", "document_ids", "flash_varlen"]


@dataclass(frozen=True)
class PackingCapabilities:
    """Model-derived loader requests and native attention setup information.

    ``packed_mask_type`` and ``return_packed_sequence_metadata`` apply only to
    batched packed inputs. THD derives kernel metadata from lengths instead.
    ``return_document_ids`` requests original membership in either format.
    ``uses_native_fa4`` is model setup information, not an emitted batch field.
    ``max_sequence_length`` is the configured decoder input-token limit per packed
    document, before batch flattening; it applies to either output format.
    """

    packed_mask_type: PackedMaskType
    return_packed_sequence_metadata: bool = False
    uses_native_fa4: bool = False
    return_document_ids: bool = False
    max_sequence_length: int | None = None


@runtime_checkable
class PackingMetadataConsumer(Protocol):
    """Consumer of AutoModel's gathered-token indices and cumulative lengths.

    ``packed_token_indices`` selects non-padding tokens; ``cu_seqlens`` and
    ``max_seqlen`` describe that gathered stream, not physical padded storage.
    Qwen3.5, Kimi Linear/K3 use these boundaries for recurrent-state processing;
    AutoModel's FA4 path also needs the unpadded layout. This is independent of
    the data loader or packing algorithm. The batch-major bundle is normalized
    by ``flatten_packed_sequence_metadata`` after PP splitting, where needed.
    """

    requires_packed_sequence_metadata: bool


@runtime_checkable
class DocumentAttentionMaskConsumer(Protocol):
    """Model that composes document eligibility with its own attention policy."""

    def prepare_document_attention_mask(
        self,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        mm_token_type_ids: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Compose selected document attention with a model-owned policy.

        Args:
            attention_mask: Integer selected IDs [batch, sequence], a prepared
                mask [batch, 1, sequence, sequence], a mapping of such masks,
                or None for native masking.
            mm_token_type_ids: Optional modality IDs [batch, sequence].
            document_ids: Optional original membership [batch, sequence], used
                only by this model policy rather than attention/MTP segmentation.

        Returns:
            A prepared mask or mapping of masks [batch, 1, sequence, sequence],
            or the unchanged input. Implementations do not mutate input tensors.
        """
        ...


@runtime_checkable
class PackedMaskConsumer(Protocol):
    """Model that owns masking and consumes compact document IDs."""

    packed_mask_type: PackedMaskType


@runtime_checkable
class NativeFA4Consumer(Protocol):
    """Model whose attention layers invoke Automodel's native FA4 callable."""

    _uses_native_fa4: bool


@runtime_checkable
class PackedMetadataEntryOwner(Protocol):
    """Model whose attention policy needs metadata at a specific forward entry."""

    def packed_metadata_entry_module(self) -> torch.nn.Module:
        """Return the called module that owns packed attention preparation."""
        ...


@runtime_checkable
class PackedAttentionBackendValidator(Protocol):
    """Model with additional constraints beyond a backend's generic contract."""

    def validate_packed_attention_backend(self, attn_implementation: str) -> None:
        """Reject a backend that cannot preserve this model's packed policy."""
        ...


@runtime_checkable
class _HFAttentionConsumer(Protocol):
    """Model whose active attention path dispatches through its HF config."""

    @property
    def _uses_hf_attention(self) -> bool:
        """Whether HF's dispatch key controls attention rather than backend.attn."""
        ...


@runtime_checkable
class AttentionBackendSelection(Protocol):
    """Typed attention selection carried by a built custom model."""

    @property
    def attn(self) -> AttentionBackend:
        """Selected native attention backend."""
        ...


def get_packing_capabilities(
    attn_implementation: str,
    *,
    model: torch.nn.Module | None = None,
) -> PackingCapabilities:
    """Describe loader requests without configuring or validating packed execution.

    This cheap query also serves unpacked and THD loaders. It does not install
    hooks or invoke model packed-backend validators. Batched packing support is
    checked separately by ``configure_packing``; THD has its own validation.

    Args:
        attn_implementation: Resolved attention implementation.
        model: Optional built consumer declaring its metadata requirements.

    Returns:
        Loader return requests plus model-owned native FA4 setup information.
        Mask representation and gathered-token requests are batched-only;
        original document membership can be requested in either format.
        The input-token limit comes from the decoder's configured
        ``max_position_embeddings``, independently of attention layout.
    """
    model = getattr(model, "module", model)
    model = getattr(model, "_orig_mod", model)
    config = getattr(model, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config(decoder=True) if callable(get_text_config) else config
    max_sequence_length = getattr(text_config, "max_position_embeddings", None)
    if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
        max_sequence_length = None
    requires_metadata = isinstance(model, PackingMetadataConsumer) and model.requires_packed_sequence_metadata
    model_mask_type = model.packed_mask_type if isinstance(model, PackedMaskConsumer) else None
    uses_native_fa4 = attn_implementation == "fa4" and isinstance(model, NativeFA4Consumer) and model._uses_native_fa4
    mask_type: PackedMaskType = (
        "document_ids"
        if uses_native_fa4
        else "flash_varlen"
        if attn_implementation in _FLASH_ATTN_IMPLEMENTATIONS
        else model_mask_type or "block_causal"
    )
    return PackingCapabilities(
        packed_mask_type=mask_type,
        return_packed_sequence_metadata=requires_metadata or uses_native_fa4,
        return_document_ids=isinstance(model, DocumentAttentionMaskConsumer),
        uses_native_fa4=uses_native_fa4,
        max_sequence_length=max_sequence_length,
    )


def get_packing_capabilities_for_models(models: Sequence[torch.nn.Module]) -> PackingCapabilities:
    """Combine loader return requests across local model or pipeline stages.

    This query does not configure models or validate an execution format.
    Batched packed execution separately checks that stage mask layouts agree.

    Args:
        models: Nonempty sequence of built local stages consuming the same batch.

    Returns:
        The first stage's mask interface, the union of optional return requests,
        and the smallest declared input-token limit across local stages.

    Raises:
        ValueError: If no model stages were supplied.
    """
    if not models:
        raise ValueError("Packing requirements need at least one model")
    contracts = [get_packing_capabilities(get_model_attn_implementation(model), model=model) for model in models]
    return replace(
        contracts[0],
        return_packed_sequence_metadata=any(contract.return_packed_sequence_metadata for contract in contracts),
        return_document_ids=any(contract.return_document_ids for contract in contracts),
        uses_native_fa4=any(contract.uses_native_fa4 for contract in contracts),
        max_sequence_length=min(
            (contract.max_sequence_length for contract in contracts if contract.max_sequence_length is not None),
            default=None,
        ),
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
        ValueError: If pre-packed THD is requested, only one metadata tensor is
            supplied, or either layout is inconsistent with the current microbatch.
    """
    if kwargs.get("qkv_format") == "thd":
        raise ValueError("Native FA4 does not support pre-packed THD inputs. Use NEAT packing.")
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


def _strip_unused_packed_metadata_at_model_entry(
    _module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Keep dataset-only NEAT offsets away from a legacy THD consumer.

    Args:
        _module: Model that does not require explicit NEAT metadata.
        args: Positional model inputs, returned unchanged in their original layouts.
        kwargs: Model inputs. ``packed_token_indices`` of shape [batch, sequence]
            or [tokens] identifies NEAT metadata; its companion ``cu_seqlens``
            has shape [batch, max_documents + 1] or [documents + 1].

    Returns:
        Inputs without NEAT token indices, cumulative lengths, and integer
        ``max_seqlen``, or None if NEAT metadata is absent. Legacy THD boundaries
        supplied without token indices are preserved.
    """
    if "packed_token_indices" not in kwargs:
        return None
    return args, {
        key: value for key, value in kwargs.items() if key not in {"packed_token_indices", "cu_seqlens", "max_seqlen"}
    }


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
    """Return the NEAT/BSHD attention implementation used by a built model.

    Custom models expose a typed ``backend.attn``. Hugging Face models record
    their resolved dispatch key on ``model.config``; this reflects preload and
    fallback decisions that are intentionally absent from the recipe config.
    Models mixing native and HF attention declare which dispatch handles BSHD.
    A separate pre-packed THD path does not determine the NEAT mask layout.

    Args:
        model: Constructed model, optionally wrapped in DDP.

    Returns:
        The active attention backend or HF dispatch key, normalized for packing.

    Raises:
        TypeError: If model is not a torch.nn.Module.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected a built torch.nn.Module, got {type(model).__name__}")
    model = getattr(model, "module", model)
    backend = getattr(model, "backend", None)
    hf_implementation = _model_attn_implementation(model)
    if isinstance(model, _HFAttentionConsumer) and model._uses_hf_attention:
        return hf_implementation or "sdpa"
    if isinstance(backend, AttentionBackendSelection):
        return backend.attn
    return hf_implementation or "sdpa"


def _allows_ordinary_causal_attention(document_ids: torch.Tensor) -> bool:
    """Whether each [B,S] attention-ID map is one eligible prefix plus padding.

    Any positive ID is valid, independently in each batch document. Prefix or
    interior exclusions and distinct subdocuments require an explicit mask.
    Trailing padding cannot affect earlier causal outputs; its loss is ignored.
    """
    allowed = document_ids > 0
    return not bool((allowed & ((document_ids != document_ids[:, :1]) | (~allowed).cumsum(dim=-1).gt(0))).any())


def _prepare_packed_attention_at_model_entry(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    packed_mask_type: PackedMaskType,
    attn_implementation: str,
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Materialize the selected document segmentation after PP splitting.

    Args:
        module: Consumer owning the attention implementation.
        args: Forward inputs, including token IDs [batch, sequence] or hidden
            states [batch, sequence, hidden].
        kwargs: Forward inputs with integer ``_packed_seq_ids`` [batch, sequence]
            describing effective attention and MTP boundaries, zero at excluded
            slots. CPU IDs allow host Flash offset construction before transfer.
            Optional ``document_ids`` [batch, sequence] retain original membership
            for a model-owned policy. Positions [batch, sequence] or
            [position_axes, batch, sequence] are passed through unchanged.
        packed_mask_type: Required attention input representation.
        attn_implementation: Configured backend, used for mask dtype.

    Returns:
        Updated args/kwargs retaining all input layouts. Selected IDs move to
        the input device. Flash adds int32 cumulative lengths [documents + 1]
        and integer maxima; SDPA/eager adds masks [batch, 1, sequence, sequence]
        on that device, or a boolean [batch, sequence] mask for ordinary causal
        attention. Existing masks, public Flash metadata and THD inputs pass
        through. No input tensor is mutated.
    """
    document_ids = kwargs.get("_packed_seq_ids")
    if document_ids is None or kwargs.get("qkv_format") == "thd":
        return None
    if kwargs.get("attention_mask") is not None or any(kwargs.get(key) is not None for key in FLASH_VARLEN_KWARGS):
        return None
    if not isinstance(document_ids, torch.Tensor) or document_ids.ndim != 2:
        raise ValueError("_packed_seq_ids must be a tensor of shape [batch, sequence]")
    # Runtime token/hidden-state inputs determine the device even on PP stages
    # whose embedding or output modules have been removed.
    inputs = (*args, *(kwargs.get(key) for key in ("input_ids", "inputs_embeds", "hidden_states")))
    input_tensor = next((value for value in inputs if isinstance(value, torch.Tensor)), None)
    parameter = next(module.parameters(), None)
    device = (
        input_tensor.device
        if input_tensor is not None
        else parameter.device
        if parameter is not None
        else document_ids.device
    )
    dtype = (
        input_tensor.dtype
        if input_tensor is not None and input_tensor.is_floating_point()
        else parameter.dtype
        if parameter is not None
        else torch.float32
    )
    updated = dict(kwargs)
    updated["_packed_seq_ids"] = document_ids.to(device=device, non_blocking=True)
    if packed_mask_type == "flash_varlen":
        if document_ids.device.type != "cpu":
            raise ValueError("Keep _packed_seq_ids on CPU until model entry; Flash offsets are host metadata")
        params = packed_seq_params_from_doc_ids(document_ids)
        metadata = to_flash_attention_kwargs(params)
        cu_seqlens = params.cu_seqlens.to(device=device, non_blocking=True)
        metadata.update(cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens)
        updated.update(metadata)
    elif packed_mask_type == "block_causal":
        if isinstance(module, DocumentAttentionMaskConsumer):
            original_ids = kwargs.get("document_ids", document_ids)
            if not isinstance(original_ids, torch.Tensor) or original_ids.shape != document_ids.shape:
                raise ValueError("Original document_ids must have shape [batch, sequence]")
            updated["attention_mask"] = module.prepare_document_attention_mask(
                document_ids.to(device=device, non_blocking=True),
                kwargs.get("mm_token_type_ids"),
                original_ids.to(device=device, non_blocking=True),
            )
        elif _allows_ordinary_causal_attention(document_ids):
            # HF otherwise infers packing from reset padding positions during training.
            updated["attention_mask"] = torch.ones(document_ids.shape, dtype=torch.bool, device=device)
        else:
            allowed = indexed_mask_to_4d_block_causal(document_ids.to(device=device, non_blocking=True))
            updated["attention_mask"] = (
                torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)
                if attn_implementation == "eager"
                else allowed
            )
    else:
        raise ValueError(f"Unsupported compact attention contract: {packed_mask_type}")
    return args, updated


def validate_flash_packing_support(attn_implementation: str = "sdpa", model: torch.nn.Module | None = None) -> None:
    """Fail loudly when flash neat packing cannot be consumed by the backend or model.

    Neat packing under flash attention relies on the varlen FlashAttention
    contract: the collater emits ``FlashAttentionKwargs`` (``cu_seq_lens_q``/
    ``cu_seq_lens_k``/``max_length_q``/``max_length_k``) which HuggingFace threads
    into ``flash_attn_varlen_func``. This function verifies, before training
    starts, that (1) the installed Transformers build exposes that contract and
    (2) the model's ``forward`` can actually receive it -- either as ``**kwargs``,
    explicit varlen parameters, or the custom-model ``_packed_seq_ids`` map. Any
    gap raises here instead of silently dropping the cumulative lengths and
    enabling cross-document attention.

    Non-flash backends (``sdpa`` / ``eager``) use a 4D block-causal mask built by
    the collater and need no validation.

    Args:
        attn_implementation: The attention implementation the model runs with.
        model: Optional already-built model (or DDP wrapper) to check for the
            ability to receive the typed packing contract. Skipped when ``None``
            or when the model's ``forward`` signature cannot be introspected.

    Raises:
        RuntimeError: If a flash backend is requested but the installed
            Transformers ``_flash_attention_forward`` does not accept the public
            varlen kwargs, or the model's ``forward`` can consume neither the
            varlen kwargs nor ``_packed_seq_ids``.
    """
    target = getattr(model, "module", model)
    target = getattr(target, "_orig_mod", target)
    if isinstance(target, PackedAttentionBackendValidator):
        target.validate_packed_attention_backend(attn_implementation)
    if attn_implementation not in _FLASH_ATTN_IMPLEMENTATIONS:
        return

    import transformers

    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot enable flash-attention neat packing: "
            "transformers.modeling_flash_attention_utils._flash_attention_forward is unavailable in "
            f"transformers {transformers.__version__}. Upgrade transformers or use sdpa/eager packing."
        ) from exc

    available = set(inspect.signature(_flash_attention_forward).parameters)
    missing = set(FLASH_VARLEN_KWARGS) - available
    if missing:
        raise RuntimeError(
            f"Cannot enable flash-attention neat packing: transformers {transformers.__version__} lacks "
            f"the varlen FlashAttention kwargs {sorted(missing)}. Upgrade transformers or use sdpa/eager packing."
        )

    _validate_model_consumes_packed_contract(model)


def _validate_model_consumes_packed_contract(model: torch.nn.Module | None) -> None:
    """Raise if ``model``'s ``forward`` cannot receive the typed packing contract.

    A model consumes the contract when its ``forward`` accepts ``**kwargs`` (HF
    models thread ``FlashAttentionKwargs`` this way), names ``_packed_seq_ids``
    (the custom-model document map), or names all four varlen kwargs explicitly.
    A ``forward`` whose signature cannot be introspected is rejected.

    Args:
        model: Already-built model, or a DDP wrapper exposing ``.module``.
    """
    if model is None:
        return
    target = getattr(model, "module", model)
    forward = getattr(target, "forward", None)
    if forward is None:
        return
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot introspect {type(target).__name__}.forward to verify it consumes packed-sequence "
            "metadata; use sdpa/eager packing or expose **kwargs on the model forward."
        ) from exc
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    if "_packed_seq_ids" in params:
        return
    # Transformers enters its varlen path only when all four cumulative-length
    # kwargs are present; a forward exposing only some of them would have the rest
    # dropped by ``filter_forward_kwargs``, so require the full set here.
    if set(FLASH_VARLEN_KWARGS).issubset(params):
        return
    raise RuntimeError(
        f"Cannot enable flash-attention neat packing: {type(target).__name__}.forward accepts neither "
        "**kwargs, all four varlen FlashAttention kwargs, nor _packed_seq_ids. Add **kwargs to the "
        "model forward or use sdpa/eager packing."
    )


def configure_packing(
    attn_implementation: str,
    *,
    model: torch.nn.Module | None = None,
) -> PackingCapabilities:
    """Configure the model consumer and return its dataset packing contract.

    HF attention accepts public masks/varlen kwargs or a CPU batch-major
    attention document map. Maps are materialized at the model-owned entry point
    (HF decoder by default) without modifying Transformers globals. Native FA4 receives batch-major metadata
    flattened at model entry after microbatch splitting. Other consumers strip
    unused native metadata so it cannot accidentally select their THD path.

    Args:
        attn_implementation: Attention implementation used by the built model.
        model: Optional built model declaring additional packed-metadata needs.

    Returns:
        Structural packed-data requirements for the dataset collater.

    Raises:
        ValueError: If native FA4 packing lacks a declared compatible consumer.
        RuntimeError: If HF flash attention cannot consume the public varlen contract.
    """
    capabilities = get_packing_capabilities(attn_implementation, model=model)
    if attn_implementation == "fa4" and not capabilities.uses_native_fa4:
        target = getattr(model, "module", model)
        target = getattr(target, "_orig_mod", target)
        if not isinstance(target, PackedMaskConsumer):
            raise ValueError("Native FA4 packing requires a model declaring native FA4 support.")
    if not capabilities.uses_native_fa4:
        validate_flash_packing_support(attn_implementation, model=model)
    if model is not None and not capabilities.return_packed_sequence_metadata:
        model = getattr(model, "module", model)
        if getattr(model, "_unused_packed_metadata_hook_handle", None) is None:
            model._unused_packed_metadata_hook_handle = model.register_forward_pre_hook(
                _strip_unused_packed_metadata_at_model_entry, with_kwargs=True
            )
    if capabilities.uses_native_fa4:
        if model is None:
            raise ValueError("Native FA4 packing requires the built model")
        _install_native_fa4_metadata_hook(model)
        return capabilities
    if model is not None and capabilities.packed_mask_type in ("flash_varlen", "block_causal"):
        from transformers import PreTrainedModel

        uncompiled = getattr(model, "_orig_mod", model)
        if isinstance(uncompiled, PackedMetadataEntryOwner):
            entry = uncompiled.packed_metadata_entry_module()
        else:
            entry = uncompiled.get_decoder() if isinstance(uncompiled, PreTrainedModel) else uncompiled
        selection = (capabilities.packed_mask_type, attn_implementation)
        handle = getattr(entry, "_packed_attention_hook_handle", None)
        if handle is not None and getattr(entry, "_packed_attention_hook_selection", None) != selection:
            handle.remove()
            handle = None
        if handle is None:
            entry._packed_attention_hook_selection = selection
            entry._packed_attention_hook_handle = entry.register_forward_pre_hook(
                partial(
                    _prepare_packed_attention_at_model_entry,
                    packed_mask_type=capabilities.packed_mask_type,
                    attn_implementation=attn_implementation,
                ),
                with_kwargs=True,
            )
    return capabilities


def validate_thd_packing_support(models: Sequence[torch.nn.Module]) -> None:
    """Reject THD inputs before loader construction if a local stage cannot consume them.

    THD uses the model's seq_lens path rather than the mask/Flash conversion
    installed by configure_packing_for_models. Validate the existing model/backend
    capability without selecting a different backend or disabling packing.
    """
    from nemo_automodel._transformers.capabilities import ModelSupports

    for model in models:
        model = getattr(model, "module", model)
        model = getattr(model, "_orig_mod", model)
        if not ModelSupports(model).supports_sequence_packing:
            raise ValueError(
                f"{type(model).__name__} with attention backend {get_model_attn_implementation(model)!r} "
                "cannot consume THD seq_lens inputs; select a supported model/backend or packed input format"
            )


def configure_packing_for_models(models: Sequence[torch.nn.Module]) -> PackingCapabilities:
    """Validate and install the adapters for batched packed execution.

    Always obtain loader requests with ``get_packing_capabilities_for_models``.
    Call this setup only for packed batched input. THD validates its length-based
    interface separately and must not install these mask/Flash/gather hooks.

    Args:
        models: Nonempty sequence of built local stages consuming the same batch.

    Returns:
        The same loader requests as the side-effect-free model query.

    Raises:
        ValueError: If stages require incompatible batched mask layouts.
    """
    contract = get_packing_capabilities_for_models(models)
    if any(
        get_packing_capabilities(get_model_attn_implementation(model), model=model).packed_mask_type
        != contract.packed_mask_type
        for model in models
    ):
        raise ValueError("Pipeline stages must agree on the batched packed mask layout")
    for model in models:
        configure_packing(get_model_attn_implementation(model), model=model)
    return contract
