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

Routes ``head_dim=512`` bf16/fp16 layers through the CuTeDSL FFPA kernel and
falls back to SDPA (or eager for softcap) for unsupported configurations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)

_FFPA_HEAD_DIM = 512
_REGISTERED: bool = False
_FALLBACK_WARNED: set[str] = set()

# Sentinel "no window" for the varlen custom op (matches ffpa_attn).
_VARLEN_WIN_NONE = -(2**31)

_SDPA_FN: Callable | None = None
_EAGER_FN: Callable | None = None
_FFPA_FN: Callable | None = None
_CUTEDSL_BACKEND: Any = None
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
    """FFPA CuTeDSL varlen forward on packed THD inputs. Returns ``(out[T,Hq,D], lse[Hq,T])``.

    Wraps ``torch.ops.ffpa_attn._varlen_fwd_cute`` with the fixed no-window /
    no-softcap / no-pack-gqa sentinels so callers pass only the attention args.
    """
    return torch.ops.ffpa_attn._varlen_fwd_cute(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FFPA CuTeDSL varlen backward using the *caller-supplied* out/lse. Returns ``(dq, dk, dv)``.

    Exposed as an explicit op call (not the package's varlen autograd) because the
    ring backward feeds the globally merged out/lse, not a chunk-local one.
    """
    return torch.ops.ffpa_attn._varlen_bwd_cute(
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


def _get_ffpa_high_level() -> tuple[Callable | None, Any]:
    global _FFPA_FN, _CUTEDSL_BACKEND
    if _FFPA_FN is None:
        try:
            from ffpa_attn import ffpa_attn_func
            from ffpa_attn.functional import CuTeDSLBackend

            _FFPA_FN = ffpa_attn_func
            _CUTEDSL_BACKEND = CuTeDSLBackend()
        except Exception:
            return None, None
    return _FFPA_FN, _CUTEDSL_BACKEND


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
) -> torch.Tensor | None:
    """Mask factory for ``ALL_MASK_ATTENTION_FUNCTIONS["ffpa"]``."""
    # Vision / audio sub-encoders share this registry but need 4D float masks.
    cfg = kwargs.get("config")
    if cfg is not None:
        model_type = getattr(cfg, "model_type", "") or ""
        if "vision" in model_type or "audio" in model_type:
            return None

    if mask_function is not None:
        from transformers.masking_utils import causal_mask_function, eager_mask

        if mask_function is not causal_mask_function:
            return eager_mask(
                batch_size=batch_size,
                q_length=q_length,
                kv_length=kv_length,
                q_offset=q_offset,
                kv_offset=kv_offset,
                mask_function=mask_function,
                attention_mask=attention_mask,
                dtype=dtype,
                **kwargs,
            )

    if attention_mask is None:
        return None
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(dtype=torch.bool)
    if attention_mask.shape[-1] != kv_length:
        attention_mask = attention_mask[:, -kv_length:]
    if bool(attention_mask.all()):
        return None
    return attention_mask


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
    """HF attention-interface forward backed by ``torch.ops.ffpa_attn._fwd_cute``."""
    # softcap must go to eager: SDPA silently drops the kwarg.
    use_sdpa = softcap is None

    def _fallback():
        if use_sdpa:
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
    # Refuse silent default to 1/sqrt(head_dim): Gemma4 full-attn uses 1/sqrt(256), not 1/sqrt(512).
    if scaling is None:
        _warn_once("scaling", "module.scaling is None — refusing ffpa default scale")
        return _fallback()
    if not _ffpa_low_level_ready():
        _warn_once("ffpa_unavailable", "ffpa_attn or torch.ops.ffpa_attn._fwd_cute not importable")
        return _fallback()

    if attention_mask is not None:
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
            return _fallback()
        if not (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.dim() == 2
            and attention_mask.dtype == torch.bool
        ):
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

    ffpa_fn, backend = _get_ffpa_high_level()
    if ffpa_fn is None:
        _warn_once("ffpa_attn_func_unavailable", "ffpa_attn.ffpa_attn_func not importable")
        return _fallback()
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


__all__ = [
    "ffpa_attention_forward",
    "ffpa_mask",
    "register_ffpa_attention",
]
