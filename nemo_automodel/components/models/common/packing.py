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

"""Flash Attention packing helpers for neat sequence packing.

When neat packing is enabled the collater produces an indexed attention mask
``[B, S]`` where each position contains the 1-based document index (0 = padding).
For example::

    [1, 1, 2, 2, 2, 0]   # 2 tokens in doc 1, 3 in doc 2, 1 padding

HuggingFace models consume this through the *public* varlen FlashAttention path:
the collater emits ``FlashAttentionKwargs`` (``cu_seq_lens_q``/``cu_seq_lens_k``/
``max_length_q``/``max_length_k``) built by
:mod:`nemo_automodel.components.datasets.packed_seq`, so no private Transformers
function is monkeypatched.

This module keeps the shared helpers that in-tree custom models
(``qwen3_5``, ``kimi_*``, ...) use to derive per-document ``cu_seqlens`` inside their
own attention, plus :func:`get_attn_implementation` used by recipes and
:func:`validate_flash_packing_support`, which fails loudly when a flash backend
is requested on a Transformers build that lacks the public varlen contract.
"""

import inspect
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_FLASH_ATTN_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")


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


def _model_attn_implementation(model) -> str | None:
    """Return the packing-relevant attention backend an already-built model runs with.

    ``model.config._attn_implementation`` is a Transformers *dispatch key*, whose
    vocabulary is wider than the mask layouts packing knows about: when flash
    attention is requested but only the ``kernels`` package provides it,
    Transformers records a kernels-hub id instead of the mainline name. Those ids
    are mapped back so a model genuinely running varlen flash attention is packed
    as such. Any key that still names no known layout yields ``None``, leaving the
    caller on the configured value.
    """
    # DDP does not proxy attribute access to the model it wraps, so read through it.
    model = getattr(model, "module", model)
    attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if attn_implementation in _FLASH_ATTN_IMPLEMENTATIONS or attn_implementation in ("sdpa", "eager"):
        return attn_implementation
    try:
        from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK
    except ImportError:
        return None
    for mainline, kernel_id in FLASH_ATTN_KERNEL_FALLBACK.items():
        if kernel_id == attn_implementation:
            return mainline
    return None


def get_attn_implementation(cfg_model, model=None) -> str:
    """Determine the attention backend from model config.

    Custom models store it in ``backend.attn``; HF models use ``attn_implementation``.

    Args:
        cfg_model: Model config node, which records what was *requested*.
        model: Optional already-built model, preferred over ``cfg_model`` for HF
            models because it records what was actually *resolved*. Model
            construction may pick a different backend than the config asks for:
            packed runs are force-switched onto flash attention
            (``_apply_preload_overrides``), an unavailable backend is downgraded
            on retry, and an omitted key defaults to flash attention rather than
            to sdpa. None of those are written back to the config. An HF model
            configured with ``te`` reports ``sdpa`` here, which is what it runs
            with TE attention injected on top.
    """
    if cfg_model is not None and hasattr(cfg_model, "backend") and hasattr(cfg_model.backend, "attn"):
        return cfg_model.backend.attn
    resolved = _model_attn_implementation(model)
    if resolved is not None:
        return resolved
    if cfg_model is not None:
        return cfg_model.get("attn_implementation", "sdpa")
    return "sdpa"


# FlashAttentionKwargs the collater emits and the flash varlen path consumes.
_PACKED_VARLEN_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")


def validate_flash_packing_support(attn_implementation: str = "sdpa", model: torch.nn.Module | None = None) -> None:
    """Fail loudly when flash neat packing cannot be consumed by the backend or model.

    Neat packing under flash attention relies on the *public* varlen contract:
    the collater emits ``FlashAttentionKwargs`` (``cu_seq_lens_q``/
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
    if attn_implementation not in _FLASH_ATTN_IMPLEMENTATIONS:
        return

    import transformers

    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot enable flash-attention neat packing because "
            f"transformers.modeling_flash_attention_utils._flash_attention_forward is unavailable in "
            f"transformers {transformers.__version__}. Refusing to continue because the packed "
            "cumulative sequence lengths would be dropped, enabling cross-document attention."
        ) from exc

    available = set(inspect.signature(_flash_attention_forward).parameters)
    missing = set(_PACKED_VARLEN_KWARGS) - available
    if missing:
        raise RuntimeError(
            "Cannot enable flash-attention neat packing because transformers "
            f"{transformers.__version__} does not support the public varlen FlashAttention kwargs "
            f"{sorted(missing)}. Upgrade transformers or use sdpa/eager packing. Refusing to continue "
            "because dropping the cumulative sequence lengths would enable cross-document attention."
        )

    _validate_model_consumes_packed_contract(model)


def _validate_model_consumes_packed_contract(model: torch.nn.Module | None) -> None:
    """Raise if ``model``'s ``forward`` cannot receive the typed packing contract.

    A model consumes the contract when its ``forward`` accepts ``**kwargs`` (HF
    models thread ``FlashAttentionKwargs`` this way), names an explicit varlen
    parameter, or names ``_packed_seq_ids`` (the custom-model document map).
    Signatures that cannot be introspected are treated as permissive so this
    check never rejects a valid model it simply cannot read.

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
    except (TypeError, ValueError):
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    if set(params) & {*_PACKED_VARLEN_KWARGS, "_packed_seq_ids"}:
        return
    raise RuntimeError(
        f"Cannot enable flash-attention neat packing because {type(target).__name__}.forward accepts "
        "neither **kwargs, the varlen FlashAttention kwargs, nor _packed_seq_ids, so the typed packing "
        "metadata would be silently dropped and enable cross-document attention. Add **kwargs to the "
        "model forward or use sdpa/eager packing."
    )
