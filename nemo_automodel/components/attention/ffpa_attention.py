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

"""FFPA attention bindings for HF ``ALL_ATTENTION_FUNCTIONS`` / ``ALL_MASK_ATTENTION_FUNCTIONS``.

Routes ``head_dim=512`` bf16/fp16 layers through the CuTeDSL FFPA kernel,
sliding-window (flex ``BlockMask``) layers through FlexAttention, and falls back
to SDPA (or eager for softcap) for other unsupported configurations.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

logger = logging.getLogger(__name__)

_FFPA_HEAD_DIM = 512
_REGISTERED: bool = False
_FALLBACK_WARNED: set[str] = set()

# Sentinel "no window" for the varlen custom op (matches ffpa_attn).
_VARLEN_WIN_NONE = -(2**31)

_SDPA_FN: Callable | None = None
_EAGER_FN: Callable | None = None
_FLEX_FN: Callable | None = None
_FFPA_HIGH_LEVEL: tuple[Callable, Any] | None = None
_FFPA_LOW_LEVEL_READY: bool | None = None


def _warn_once(reason: str, message: str, *, target: str = "sdpa") -> None:
    if reason in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(reason)
    logger.warning("ffpa fallback to %s: %s (reason=%r)", target, message, reason)


def _get_sdpa() -> Callable | None:
    global _SDPA_FN
    if _SDPA_FN is None:
        try:
            from transformers.integrations.sdpa_attention import sdpa_attention_forward

            _SDPA_FN = sdpa_attention_forward
        except Exception:
            return None
    return _SDPA_FN


def _get_eager() -> Callable | None:
    global _EAGER_FN
    if _EAGER_FN is None:
        try:
            from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward

            _EAGER_FN = eager_attention_forward
        except Exception:
            return None
    return _EAGER_FN


def _get_flex() -> Callable | None:
    global _FLEX_FN
    if _FLEX_FN is None:
        try:
            from transformers.integrations.flex_attention import flex_attention_forward

            _FLEX_FN = flex_attention_forward
        except Exception:
            return None
    return _FLEX_FN


def _is_block_mask(mask: Any) -> bool:
    """True for a flex ``BlockMask`` — the marker that a layer should run on FlexAttention."""
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except Exception:
        return False
    return isinstance(mask, BlockMask)


def _ffpa_low_level_ready() -> bool:
    global _FFPA_LOW_LEVEL_READY
    if _FFPA_LOW_LEVEL_READY is None:
        try:
            import ffpa_attn.cute  # noqa: F401

            _ = torch.ops.ffpa_attn._fwd_cute
            _ = torch.ops.ffpa_attn._bwd_cute
            _FFPA_LOW_LEVEL_READY = True
        except Exception:
            _FFPA_LOW_LEVEL_READY = False
    return _FFPA_LOW_LEVEL_READY


def _ffpa_varlen_ready() -> bool:
    """Whether the FFPA CuTeDSL *varlen* ops are importable and registered."""
    if not _ffpa_low_level_ready():
        return False
    try:
        _ = torch.ops.ffpa_attn._varlen_fwd_cute
        _ = torch.ops.ffpa_attn._varlen_bwd_cute
        return True
    except Exception:
        return False


def _ffpa_varlen_fwd(
    q_pack: torch.Tensor,
    k_pack: torch.Tensor,
    v_pack: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FFPA CuTeDSL varlen forward on packed THD inputs.

    Windowing, softcap, and packed GQA are disabled at this seam.

    Args:
        q_pack: Float16 or bfloat16 CUDA tensor of shape [total_query_tokens, query_heads, head_dim] in packed
            THD layout. QKV share dtype and device and are made contiguous before dispatch.
        k_pack: Key tensor of shape [total_key_value_tokens, key_value_heads, head_dim] in packed THD layout.
        v_pack: Value tensor of shape [total_key_value_tokens, key_value_heads, value_head_dim] in packed THD
            layout.
        cu_q: Int32 CUDA tensor of shape [batch + 1] on the same device as ``q_pack``, containing cumulative
            query-token offsets from 0 through ``total_query_tokens``.
        cu_k: Int32 CUDA tensor of shape [batch + 1] on the same device as ``q_pack``, containing cumulative
            key/value-token offsets from 0 through ``total_key_value_tokens``.
        max_q: Maximum query sequence length represented by ``cu_q``.
        max_k: Maximum key/value sequence length represented by ``cu_k``.
        scale: Pre-softmax scaling factor applied to query-key scores.
        causal: Whether to apply the tail-aligned causal mask within each packed sequence.

    Returns:
        A tuple containing an output tensor of shape [total_query_tokens, query_heads, value_head_dim] with the
        dtype and device of ``q_pack``, and a float32 LSE tensor of shape [query_heads, total_query_tokens] on the
        same device. Neither output aliases an input.
    """
    # Cast the dynamic torch op packet to the callable signature expected by the type checker.
    varlen_fwd_cute = cast(
        "Callable[..., tuple[torch.Tensor, torch.Tensor]]",
        torch.ops.ffpa_attn._varlen_fwd_cute,
    )
    return varlen_fwd_cute(
        q_pack.contiguous(),
        k_pack.contiguous(),
        v_pack.contiguous(),
        cu_q,
        cu_k,
        int(max_q),
        int(max_k),
        float(scale),
        bool(causal),
        _VARLEN_WIN_NONE,
        _VARLEN_WIN_NONE,
        0.0,
        False,
    )


@functools.lru_cache(maxsize=None)
def _ffpa_sm80_split_d_backward(device: torch.device, head_dim: int, head_dim_v: int) -> bool:
    """Whether ffpa-attn runs its generic SM80 Split-D backward for this device and head dims.

    That kernel accumulates dK/dV across tiles through an HBM buffer in the activation dtype, so
    every partial sum loses bits to a bf16 round trip and the error grows with the chunk length
    (measured on the SM80 path, head_dim 512: ~4e-3 relative L2 at 256 tokens, ~1e-2 at 2K, ~1.5e-2 at 4K,
    three to four times the Flex ring). Its ``grad_kv_storage_dtype`` option widens that buffer
    to fp32; the SM90/SM100 kernels accumulate in registers/TMEM and reject the option. Mirrors
    the library's own dispatch so the two cannot disagree; if that dispatch cannot be imported,
    the backward keeps the library defaults and says so once.
    """
    try:
        from ffpa_attn.cute import _backward_impl_for_device, _ffpa_attn_backward_sm80
    except ImportError as exc:
        _warn_once(
            "ffpa_backward_dispatch",
            f"cannot inspect the ffpa-attn backward dispatch ({exc}); dK/dV keep the activation-dtype storage",
            target="ffpa-attn defaults",
        )
        return False
    return _backward_impl_for_device(device, head_dim, head_dim_v) is _ffpa_attn_backward_sm80


def _ffpa_varlen_bwd_sm80(
    grad_out_pack: torch.Tensor,
    q_pack: torch.Tensor,
    k_pack: torch.Tensor,
    v_pack: torch.Tensor,
    out_pack: torch.Tensor,
    lse_pack: torch.Tensor,
    bounds_q: list[int],
    bounds_k: list[int],
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SM80 varlen backward as per-document dense calls, so every segment gets the fp32 dK/dV
    storage of :func:`_ffpa_dense_bwd`.

    ffpa-attn's SM80 varlen backward is this same per-segment decomposition, but it allocates the
    segment buffers in the activation dtype, so its storage-dtype option never reaches them.
    """

    def sdpa_layout(packed: torch.Tensor) -> torch.Tensor:  # [T, H, D] -> [1, H, T, D]
        return packed.transpose(0, 1).unsqueeze(0)

    dq = torch.zeros_like(q_pack)
    dk = torch.zeros_like(k_pack)
    dv = torch.zeros_like(v_pack)
    for q0, q1, k0, k1 in zip(bounds_q[:-1], bounds_q[1:], bounds_k[:-1], bounds_k[1:]):
        if q1 == q0 or k1 == k0:
            continue
        dq_seg, dk_seg, dv_seg = _ffpa_dense_bwd(
            sdpa_layout(grad_out_pack[q0:q1]),
            sdpa_layout(q_pack[q0:q1]),
            sdpa_layout(k_pack[k0:k1]),
            sdpa_layout(v_pack[k0:k1]),
            sdpa_layout(out_pack[q0:q1]),
            lse_pack[:, q0:q1].unsqueeze(0).contiguous(),
            scale=scale,
            causal=causal,
        )
        dq[q0:q1] = dq_seg[0].transpose(0, 1)
        dk[k0:k1] = dk_seg[0].transpose(0, 1)
        dv[k0:k1] = dv_seg[0].transpose(0, 1)
    return dq, dk, dv


def _ffpa_varlen_bwd(
    grad_out_pack: torch.Tensor,
    q_pack: torch.Tensor,
    k_pack: torch.Tensor,
    v_pack: torch.Tensor,
    out_pack: torch.Tensor,
    lse_pack: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    *,
    scale: float,
    causal: bool,
    cu_q_host: list[int] | None = None,
    cu_k_host: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run FFPA CuTeDSL varlen backward with the caller's globally merged output and LSE.

    On the generic SM80 path the per-document segments run through the dense op with fp32
    dK/dV storage (see :func:`_ffpa_sm80_split_d_backward`).

    Args:
        grad_out_pack: Output-gradient tensor of shape [total_query_tokens, query_heads, value_head_dim] in
            packed THD layout.
        q_pack: Float16 or bfloat16 CUDA tensor of shape [total_query_tokens, query_heads, head_dim] in packed
            THD layout. QKV, ``grad_out_pack``, and ``out_pack`` share dtype and device and are made contiguous
            before dispatch.
        k_pack: Key tensor of shape [total_key_value_tokens, key_value_heads, head_dim] in packed THD layout.
        v_pack: Value tensor of shape [total_key_value_tokens, key_value_heads, value_head_dim] in packed THD
            layout.
        out_pack: Caller's globally merged output tensor of shape
            [total_query_tokens, query_heads, value_head_dim] in packed THD layout.
        lse_pack: Caller's globally merged float32 LSE tensor of shape [query_heads, total_query_tokens] on the
            same device as ``q_pack``.
        cu_q: Int32 CUDA tensor of shape [batch + 1] on the same device as ``q_pack``, containing cumulative
            query-token offsets.
        cu_k: Int32 CUDA tensor of shape [batch + 1] on the same device as ``q_pack``, containing cumulative
            key/value-token offsets.
        max_q: Maximum query sequence length represented by ``cu_q``.
        max_k: Maximum key/value sequence length represented by ``cu_k``.
        scale: Pre-softmax scaling factor applied to query-key scores.
        causal: Whether to apply the tail-aligned causal mask within each packed sequence.
        cu_q_host: Host-side copy of ``cu_q`` when the caller already holds one; spares the SM80 path a
            device-to-host sync.
        cu_k_host: Host-side copy of ``cu_k``, likewise.

    Returns:
        A ``(dq, dk, dv)`` tuple in packed THD layout. Each gradient has the shape, dtype, and device of its
        corresponding QKV input and does not alias an input.
    """
    if _ffpa_sm80_split_d_backward(q_pack.device, q_pack.shape[-1], v_pack.shape[-1]):
        bounds_q = cu_q.tolist() if cu_q_host is None else cu_q_host
        bounds_k = cu_k.tolist() if cu_k_host is None else cu_k_host
        return _ffpa_varlen_bwd_sm80(
            grad_out_pack, q_pack, k_pack, v_pack, out_pack, lse_pack, bounds_q, bounds_k, scale=scale, causal=causal
        )
    # Cast the dynamic torch op packet to the callable signature expected by the type checker.
    varlen_bwd_cute = cast(
        "Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]",
        torch.ops.ffpa_attn._varlen_bwd_cute,
    )
    return varlen_bwd_cute(
        q_pack.contiguous(),
        k_pack.contiguous(),
        v_pack.contiguous(),
        out_pack.contiguous(),
        grad_out_pack.contiguous(),
        lse_pack.contiguous(),
        cu_q,
        cu_k,
        int(max_q),
        int(max_k),
        float(scale),
        bool(causal),
        _VARLEN_WIN_NONE,
        _VARLEN_WIN_NONE,
        0.0,
        None,
    )


def _ffpa_dense_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FFPA CuTeDSL dense forward on SDPA-layout tensors.

    The returned per-chunk LSE supports the ring online-softmax merge.

    Args:
        q: Float16 or bfloat16 CUDA tensor of shape [batch, query_heads, query_sequence, head_dim] in SDPA
            layout. QKV share dtype, device, and head dimension.
        k: Key tensor of shape [batch, key_value_heads, key_value_sequence, head_dim] in SDPA layout.
        v: Value tensor of shape [batch, key_value_heads, key_value_sequence, value_head_dim] in SDPA layout.
        scale: Pre-softmax scaling factor applied to query-key scores.
        causal: Whether to apply the causal mask.

    Returns:
        A tuple containing an output tensor of shape [batch, query_heads, query_sequence, value_head_dim] with
        the dtype and device of ``q``, and a float32 LSE tensor of shape [batch, query_heads, query_sequence] on
        the same device. Neither output aliases an input.
    """
    from ffpa_attn.cute import _ffpa_attn_forward_cute

    return _ffpa_attn_forward_cute(q, k, v, float(scale), bool(causal), return_lse=True)


def _ffpa_dense_bwd(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run FFPA CuTeDSL dense backward with the caller's globally merged output and LSE.

    Args:
        grad_out: Output-gradient tensor of shape [batch, query_heads, query_sequence, value_head_dim] in SDPA
            layout.
        q: Float16 or bfloat16 CUDA tensor of shape [batch, query_heads, query_sequence, head_dim] in SDPA
            layout. QKV, ``grad_out``, and ``out`` share dtype and device.
        k: Key tensor of shape [batch, key_value_heads, key_value_sequence, head_dim] in SDPA layout.
        v: Value tensor of shape [batch, key_value_heads, key_value_sequence, value_head_dim] in SDPA layout.
        out: Caller's globally merged output tensor of shape
            [batch, query_heads, query_sequence, value_head_dim] in SDPA layout.
        lse: Caller's globally merged float32 LSE tensor of shape [batch, query_heads, query_sequence] on the
            same device as ``q``.
        scale: Pre-softmax scaling factor applied to query-key scores.
        causal: Whether to apply the causal mask.

    Returns:
        A ``(dq, dk, dv)`` tuple in SDPA layout. Each gradient has the shape, dtype, and device of its
        corresponding QKV input and does not alias an input; GQA ``dk`` and ``dv`` use ``key_value_heads``.
    """
    from ffpa_attn.cute import _ffpa_attn_backward_cute

    storage = (
        {"grad_kv_storage_dtype": torch.float32}
        if _ffpa_sm80_split_d_backward(q.device, q.shape[-1], v.shape[-1])
        else {}
    )
    return _ffpa_attn_backward_cute(grad_out, q, k, v, out, lse, float(scale), bool(causal), **storage)


def _get_ffpa_high_level() -> tuple[Callable, Any] | None:
    global _FFPA_HIGH_LEVEL
    if _FFPA_HIGH_LEVEL is None:
        try:
            from ffpa_attn import ffpa_attn_func
            from ffpa_attn.functional import CuTeDSLBackend

            _FFPA_HIGH_LEVEL = ffpa_attn_func, CuTeDSLBackend()
        except Exception:
            return None
    return _FFPA_HIGH_LEVEL


def ffpa_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable | None = None,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> "torch.Tensor | BlockMask | None":
    """Build the attention mask used by the registered FFPA backend.

    Args:
        batch_size: Number of sequences in the batch.
        q_length: Query sequence length.
        kv_length: Key/value sequence length.
        q_offset: Global offset of the first query token.
        kv_offset: Global offset of the first key/value token.
        mask_function: Optional index-based mask predicate.
        attention_mask: Optional padding mask tensor of shape ``[batch, kv_sequence]``
            or a prebuilt mask tensor of shape ``[batch, 1, query_sequence, kv_sequence]``.
        dtype: Floating-point dtype used when a backend needs an additive mask.
        **kwargs: Mask-factory metadata forwarded by Transformers.

    Returns:
        A boolean tensor of shape ``[batch, kv_sequence]`` in padding layout or
        ``[batch, 1, query_sequence, kv_sequence]`` in SDPA layout, a ``BlockMask`` for FlexAttention, or
        ``None`` when causal attention needs no explicit mask. A padding-layout result may alias
        ``attention_mask``.
    """
    # Vision / audio sub-encoders share this registry but need 4D float masks.
    cfg = kwargs.get("config")
    model_type = getattr(cfg, "model_type", "") or ""
    if "vision" in model_type or "audio" in model_type:
        return None

    if mask_function is not None:
        from transformers.masking_utils import causal_mask_function, flex_attention_mask

        if mask_function is not causal_mask_function:
            device = kwargs.pop("device", None) or (attention_mask.device if attention_mask is not None else "cpu")
            sliding_backend = str(getattr(cfg, "ffpa_sliding_attn_backend", "flex")).lower()
            if sliding_backend == "sdpa":
                from transformers.masking_utils import sdpa_mask

                mask_builder = sdpa_mask
            elif sliding_backend == "flex":
                # BlockMask preserves sliding sparsity and selects the Flex path in ffpa_attention_forward.
                mask_builder = flex_attention_mask
            else:
                raise ValueError(f"ffpa_sliding_attn_backend must be 'flex' or 'sdpa', got {sliding_backend!r}.")
            return mask_builder(
                batch_size=batch_size,
                q_length=q_length,
                kv_length=kv_length,
                q_offset=q_offset,
                kv_offset=kv_offset,
                mask_function=mask_function,
                attention_mask=attention_mask,
                device=device,
                **kwargs,
            )

    if attention_mask is None:
        return None
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(dtype=torch.bool)
    if attention_mask.shape[-1] != kv_length:
        attention_mask = attention_mask[:, -kv_length:]
    return None if bool(attention_mask.all()) else attention_mask


def ffpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Route the Hugging Face attention interface through FFPA or an eligible fallback.

    Args:
        module: Attention module providing the training state and ``head_dim`` used for FFPA eligibility.
        query: Query tensor of shape [batch, query_heads, query_sequence, head_dim] in SDPA layout.
        key: Key tensor of shape [batch, key_value_heads, key_value_sequence, head_dim] in SDPA layout.
        value: Value tensor of shape [batch, key_value_heads, key_value_sequence, value_head_dim] in SDPA layout.
            QKV share dtype and device. FFPA requires float16 or bfloat16 CUDA tensors with head dimension 512.
        attention_mask: Optional boolean padding tensor of shape [batch, key_value_sequence] or additive tensor
            of shape [batch, 1, query_sequence, key_value_sequence]. A Flex ``BlockMask`` produced by
            ``ffpa_mask`` may also be supplied for the FlexAttention route.
        dropout: Dropout probability. Nonzero training dropout makes the FFPA route ineligible.
        scaling: Optional pre-softmax scaling factor applied to query-key scores.
        softcap: Optional score cap; when present, the eager fallback owns execution.
        **kwargs: Additional Hugging Face attention arguments forwarded to the selected backend.

    Returns:
        A tuple containing an output tensor of shape [batch, query_sequence, query_heads, value_head_dim] with
        the dtype and device of ``query``, and either eager-fallback attention weights of shape
        [batch, query_heads, query_sequence, key_value_sequence] or ``None``.
    """
    # Flex BlockMask inputs route to Flex; tensor masks continue through FFPA eligibility and fallback checks.
    if _is_block_mask(attention_mask):
        flex = _get_flex()
        if flex is None:
            raise RuntimeError("ffpa_attention_forward: flex_attention is unavailable for sliding-window layers")
        return flex(module, query, key, value, attention_mask, scaling=scaling, softcap=softcap, **kwargs)

    def _fallback():
        if softcap is None:
            sdpa = _get_sdpa()
            if sdpa is not None:
                return sdpa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        eager = _get_eager()
        if eager is None:
            raise RuntimeError("ffpa_attention_forward: no SDPA/eager fallback importable from transformers")
        return eager(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, softcap=softcap, **kwargs
        )

    if getattr(module, "head_dim", None) != _FFPA_HEAD_DIM:
        return _fallback()
    if query.dtype not in (torch.float16, torch.bfloat16):
        _warn_once("dtype", f"need bf16/fp16, got {query.dtype}")
        return _fallback()
    if softcap is not None:
        _warn_once("softcap", f"CuTeDSL backend does not support softcap (got {softcap})", target="eager")
        return _fallback()
    if module.training and float(dropout) > 0.0:
        _warn_once("dropout", f"CuTeDSL backend rejects dropout={float(dropout):.4f}")
        return _fallback()
    # Refuse FFPA's implicit 1/sqrt(head_dim) default: Gemma4 supplies its own scaling (1.0 in Transformers 5.12.1).
    if scaling is None:
        _warn_once("scaling", "module.scaling is None — refusing ffpa default scale")
        return _fallback()
    if not _ffpa_low_level_ready():
        _warn_once("ffpa_unavailable", "ffpa_attn or torch.ops.ffpa_attn._fwd_cute not importable")
        return _fallback()

    if attention_mask is None:
        ffpa_high_level = _get_ffpa_high_level()
        if ffpa_high_level is None:
            _warn_once("ffpa_attn_func_unavailable", "ffpa_attn.ffpa_attn_func not importable")
            return _fallback()
        ffpa_fn, backend = ffpa_high_level
        # ffpa returns [B, H_q, N_q, D]; HF caller expects [B, S, H_q, D].
        out_bhnd = ffpa_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=float(scaling),
            enable_gqa=True,
            backend=backend,
        )
        return out_bhnd.transpose(1, 2).contiguous(), None

    if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
        return _fallback()
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2 or attention_mask.dtype != torch.bool:
        _warn_once(
            "unsupported_mask",
            f"ffpa expects 2D bool pad mask or 4D additive float; got "
            f"shape={tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else type(attention_mask).__name__} "
            f"dtype={getattr(attention_mask, 'dtype', None)}",
        )
        return _fallback()
    from transformers.modeling_flash_attention_utils import _pad_input, _unpad_input, _upad_input

    B, _, S, _ = query.shape
    q_pack, k_pack, v_pack, indices_q, (cu_q, cu_k), (max_q, max_k) = _upad_input(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attention_mask, S, _unpad_input
    )
    out_pack, _ = _ffpa_varlen_fwd(q_pack, k_pack, v_pack, cu_q, cu_k, max_q, max_k, scale=scaling, causal=True)
    return _pad_input(out_pack, indices_q, B, S), None


def register_ffpa_attention() -> bool:
    """Register ``"ffpa"`` in HF attention/mask registries. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return False
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as exc:
        logger.debug("transformers not importable: %s", exc)
        return False
    try:
        ALL_ATTENTION_FUNCTIONS.register("ffpa", ffpa_attention_forward)
        ALL_MASK_ATTENTION_FUNCTIONS.register("ffpa", ffpa_mask)
    except Exception as exc:
        logger.warning("failed to register 'ffpa': %s", exc)
        return False
    _REGISTERED = True
    logger.info("registered ALL_ATTENTION_FUNCTIONS['ffpa'] + ALL_MASK_ATTENTION_FUNCTIONS['ffpa']")
    return True


def setup_ffpa_backend(cp_size: int, has_packed_sequence: bool) -> None:
    """Validate FFPA preconditions, then register the HF ``"ffpa"`` backend.

    The ``"ffpa"`` backend calls the FFPA op directly, bypassing both the ring-CP
    SDPA swap and packed-sequence masking; for CP or packing use
    ``attn_implementation="sdpa"`` + ``text_config.cp_full_attn_backend="ffpa"`` instead.
    """
    if cp_size > 1:
        raise ValueError(
            f"attn_implementation='ffpa' cannot be combined with context parallelism (cp_size={cp_size}); "
            "use attn_implementation='sdpa' + text_config.cp_full_attn_backend='ffpa' for CP + FFPA."
        )
    if has_packed_sequence:
        raise ValueError("attn_implementation='ffpa' is incompatible with packed sequences.")
    register_ffpa_attention()


__all__ = [
    "ffpa_attention_forward",
    "ffpa_mask",
    "register_ffpa_attention",
    "setup_ffpa_backend",
]
