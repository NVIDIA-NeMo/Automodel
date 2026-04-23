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

"""TransformerEngine attention injection for HuggingFace models.

Replaces ``F.scaled_dot_product_attention`` within each HF ``self_attn``
module's forward pass with TE's ``DotProductAttention``, enabling the
FlashAttention-3 kernel and FP8 training without requiring model-specific
rewrites.

The injection works by:
1. Detecting ``self_attn`` modules with a standard HF projection layout
   (separate ``q_proj``, ``k_proj``, ``v_proj``).
2. Creating a ``DotProductAttention`` instance (stored as ``module.attn_module``)
   so that :func:`_uses_te_attention` can detect it.
3. Monkey-patching ``module.forward`` to temporarily swap in a TE-backed
   replacement for ``torch.nn.functional.scaled_dot_product_attention``
   while the original HF forward runs.

Call :func:`inject_te_attention` on the model *before* FSDP wrapping and
*after* any weight loading (so head-count/head-dim values are correct).

Supported patterns
------------------
- Standard Llama-style layout: separate ``q_proj``/``k_proj``/``v_proj``,
  GQA via ``repeat_kv`` (``enable_gqa=False``) or ``enable_gqa=True``.
  Covers Llama, Gemma, Qwen2, Mistral, and most popular HF causal LMs.

Limitations (v1)
----------------
- Only causal (``is_causal=True``) and no-mask attention are supported.
  Non-trivial ``attn_mask`` tensors fall back to native SDPA with a debug log.
- Sliding-window attention is not yet handled (uses unbounded left window).
- Models using ``from torch.nn.functional import scaled_dot_product_attention``
  (a local import) will not pick up the runtime patch; affected modules are
  skipped with a warning.
"""

import logging
import types
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Attribute set on ``self_attn`` modules to hold the TE module,
# and on the top-level model to signal that injection was performed.
_TE_MODULE_ATTR = "attn_module"
_TE_MODEL_FLAG = "_te_attention_injected"


# ---------------------------------------------------------------------------
# Parameter inference
# ---------------------------------------------------------------------------


def _proj_out_features(proj: torch.nn.Module | None) -> int | None:
    """Return the output feature count of a projection module.

    Handles three layouts:
    - Standard ``nn.Linear``: reads ``proj.out_features`` directly.
    - Weight-only: reads ``proj.weight.shape[0]`` (works on meta device).
    - Wrapped linear (e.g. ``Gemma4ClippableLinear``): recurses into the
      ``proj.linear`` child module.
    """
    if proj is None:
        return None
    out = getattr(proj, "out_features", None)
    if out is not None:
        return int(out)
    w = getattr(proj, "weight", None)
    if w is not None:
        return int(w.shape[0])
    inner = getattr(proj, "linear", None)
    if inner is not None:
        return _proj_out_features(inner)
    return None


def _infer_attn_params(module: torch.nn.Module) -> dict[str, Any] | None:
    """Infer attention hyper-parameters from a HF ``self_attn`` module.

    Returns ``None`` when the module does not match the expected layout.

    Head counts are read from module attributes when present (standard HF),
    or inferred from projection ``out_features`` when absent (e.g.
    ``Gemma4TextAttention`` which stores head count only in the config).
    """
    if not (hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")):
        return None

    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        return None
    head_dim = int(head_dim)

    num_heads = getattr(module, "num_heads", None) or getattr(module, "num_attention_heads", None)
    if num_heads is None:
        # Infer from q_proj output dimension (works even on meta device).
        # Fall back to weight.shape[0] for custom linears lacking out_features
        # (e.g. Gemma4ClippableLinear).
        q_proj = getattr(module, "q_proj", None)
        q_out = _proj_out_features(q_proj)
        if q_out is None:
            return None
        num_heads = q_out // head_dim
    num_heads = int(num_heads)

    num_kv_heads = getattr(module, "num_key_value_heads", None)
    if num_kv_heads is None:
        k_proj = getattr(module, "k_proj", None)
        k_out = _proj_out_features(k_proj)
        num_kv_heads = (k_out // head_dim) if k_out is not None else num_heads
    num_kv_heads = int(num_kv_heads)

    # Sliding-window attention: convert HF's window token count to TE's
    # (left_tokens, right_tokens) convention.  (-1, 0) = unbounded / global.
    sliding_window = getattr(module, "sliding_window", None)
    if sliding_window is not None and sliding_window > 0:
        # TE window_size[0] = number of tokens to the LEFT of current position.
        # A window of W tokens (inclusive of current) → W-1 to the left.
        te_window_size = (int(sliding_window) - 1, 0)
    else:
        te_window_size = (-1, 0)

    # Some models (e.g. Gemma4) store a pre-computed softmax scale as
    # ``module.scaling`` instead of using the standard head_dim**-0.5.
    softmax_scale = getattr(module, "scaling", None)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "window_size": te_window_size,
        "softmax_scale": float(softmax_scale),
    }


# ---------------------------------------------------------------------------
# TE module creation
# ---------------------------------------------------------------------------


def _create_te_dot_product_attention(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_size: tuple[int, int] = (-1, 0),
    softmax_scale: float | None = None,
) -> "transformer_engine.pytorch.attention.DotProductAttention":  # noqa: F821
    """Instantiate a TE ``DotProductAttention`` for the given attention shape."""
    from transformer_engine.pytorch.attention import DotProductAttention

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    return DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=(head_dim, head_dim),
        attn_mask_type="causal",
        qkv_format="bshd",
        softmax_scale=softmax_scale,
        num_gqa_groups=num_kv_heads,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# SDPA replacement
# ---------------------------------------------------------------------------


def _make_te_sdpa(
    te_module: torch.nn.Module,
    num_heads: int,
    num_kv_heads: int,
    original_sdpa,
    window_size: tuple[int, int] = (-1, 0),
) -> Any:
    """Return a callable that replaces ``F.scaled_dot_product_attention``.

    The replacement:
    - Transposes Q/K/V from HF's ``[B, H, S, D]`` to TE's ``[B, S, H, D]``.
    - Undoes ``repeat_kv`` when TE can handle GQA natively.
    - Falls back to ``original_sdpa`` for non-trivial ``attn_mask`` inputs.
    - Transposes the TE output back to ``[B, H, S, D]`` before returning.
    """

    def te_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        **_unused: Any,
    ) -> torch.Tensor:
        # HF passes Q/K/V in [B, H, S, D]; TE expects [B, S, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # If repeat_kv was applied (enable_gqa=False with a GQA model), undo it:
        # after transpose k/v are [B, S, H, D] but TE needs [B, S, Hkv, D].
        if not enable_gqa and num_kv_heads < num_heads and k.shape[2] > num_kv_heads:
            step = k.shape[2] // num_kv_heads
            k = k[:, :, ::step, :].contiguous()
            v = v[:, :, ::step, :].contiguous()

        # Non-trivial masks are not yet converted; fall back to native SDPA.
        if attn_mask is not None:
            logger.debug("TE attention: non-None attn_mask encountered; falling back to native SDPA for this call.")
            return original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        mask_type = "causal" if is_causal else "no_mask"
        out = te_module(q, k, v, attn_mask_type=mask_type, window_size=window_size)

        # TE returns [B, S, H, D]; transpose back to HF's [B, H, S, D].
        return out.transpose(1, 2).contiguous()

    return te_sdpa


# ---------------------------------------------------------------------------
# Forward patching
# ---------------------------------------------------------------------------


def _patch_module_forward(module: torch.nn.Module, te_sdpa) -> None:
    """Shadow ``module.forward`` with a version that uses TE for SDPA."""
    # Capture the *class-level* (unbound) forward method so that we call the
    # original implementation rather than our own patched instance method.
    original_forward = type(module).forward

    def patched_forward(inner_self, *args, **kwargs):
        orig = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = te_sdpa
        try:
            return original_forward(inner_self, *args, **kwargs)
        finally:
            torch.nn.functional.scaled_dot_product_attention = orig

    # Bind to instance to shadow the class method lookup.
    module.forward = types.MethodType(patched_forward, module)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def inject_te_attention_into_module(module: torch.nn.Module) -> bool:
    """Inject TE attention into a single HF ``self_attn`` module.

    Returns ``True`` on success, ``False`` when the module does not match
    the expected layout.
    """
    params = _infer_attn_params(module)
    if params is None:
        return False

    # Capture the original SDPA *before* any patching so the fallback path
    # inside ``te_sdpa`` always reaches the real kernel.
    original_sdpa = F.scaled_dot_product_attention

    te_module = _create_te_dot_product_attention(**params)
    te_sdpa = _make_te_sdpa(
        te_module=te_module,
        num_heads=params["num_heads"],
        num_kv_heads=params["num_kv_heads"],
        original_sdpa=original_sdpa,
        window_size=params["window_size"],
    )
    _patch_module_forward(module, te_sdpa)

    # Expose ``attn_module`` so ``_uses_te_attention`` detects this module.
    setattr(module, _TE_MODULE_ATTR, te_module)
    return True


def inject_te_attention(model: torch.nn.Module) -> None:
    """Walk *model* and inject TE attention into all compatible ``self_attn`` modules.

    Skips modules that already carry ``attn_module`` (i.e. custom models or
    modules that were already patched).  Sets ``model._te_attention_injected``
    on success so that :func:`_uses_te_attention` can short-circuit the walk.
    """
    injected = 0
    for name, module in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        if hasattr(module, _TE_MODULE_ATTR):
            # Custom model or already patched.
            continue
        if inject_te_attention_into_module(module):
            injected += 1
            logger.debug("Injected TE attention into %s", name)

    if injected > 0:
        logger.info("Injected TE DotProductAttention into %d self_attn module(s).", injected)
        setattr(model, _TE_MODEL_FLAG, True)
    else:
        logger.warning(
            "inject_te_attention: no compatible self_attn modules found. "
            "The model may not use the standard HF q_proj/k_proj/v_proj layout."
        )
