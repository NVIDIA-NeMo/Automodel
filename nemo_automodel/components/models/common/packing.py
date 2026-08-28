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

from nemo_automodel.shared.packed_sequence import get_seqlens_in_batch as get_seqlens_in_batch
from nemo_automodel.shared.packed_sequence import get_unpad_data

logger = logging.getLogger(__name__)

_FLASH_ATTN_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")

# HF-style ``attn_implementation`` values that the custom-model path implements natively as a
# ``BackendConfig.attn`` backend. Only FA4 has one; sdpa/eager/magi and FA2/FA3 do not, and are
# deliberately absent so a recipe pairing e.g. ``attn_implementation: sdpa`` (which only gates HF
# config validation) with ``backend.attn: te`` keeps running TE.
_ATTN_IMPL_TO_NATIVE_BACKEND = {"flash_attention_4": "fa4"}


def native_backend_from_attn_implementation(cfg_model) -> str | None:
    """Return the native ``backend.attn`` requested via HF-style ``attn_implementation``, if any.

    Custom models select attention through ``backend.attn``, so an ``attn_implementation`` naming a
    flash variant used to be silently ignored on that path -- the model quietly ran whatever
    ``backend.attn`` said. Honor the request when the value names a backend the native path actually
    implements; return ``None`` (leave ``backend.attn`` alone) otherwise.

    Args:
        cfg_model: Model config node.

    Returns:
        The native backend name, or ``None`` when ``attn_implementation`` has no native equivalent.
    """
    if cfg_model is None:
        return None
    # Config nodes expose .get(); plain namespaces (and test doubles) may not.
    getter = getattr(cfg_model, "get", None)
    impl = getter("attn_implementation", None) if callable(getter) else getattr(cfg_model, "attn_implementation", None)
    if not isinstance(impl, str):
        return None
    native = _ATTN_IMPL_TO_NATIVE_BACKEND.get(impl)
    if native is None and impl in _FLASH_ATTN_IMPLEMENTATIONS:
        logger.warning(
            "model.attn_implementation=%r has no custom-model equivalent; the native attention "
            "backend is selected by model.backend.attn (currently %r) and this key only affects "
            "HF-implemented submodules such as a VLM vision tower.",
            impl,
            getattr(getattr(cfg_model, "backend", None), "attn", None),
        )
    return native


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
        return native_backend_from_attn_implementation(cfg_model) or cfg_model.backend.attn
    resolved = _model_attn_implementation(model)
    if resolved is not None:
        return resolved
    if cfg_model is not None:
        return cfg_model.get("attn_implementation", "sdpa")
    return "sdpa"


def apply_attn_implementation_to_backend(cfg_model) -> None:
    """Promote a native-equivalent ``attn_implementation`` onto ``backend.attn``, in place.

    Custom models build attention from ``backend.attn``, so setting only ``attn_implementation``
    would leave the requested backend unbuilt even though :func:`get_attn_implementation` reports
    it -- the packed mask and the attention module would then disagree. Rewrite the config once,
    before the model is instantiated, so both follow the same key.

    No-op unless ``attn_implementation`` names a backend the native path implements, so recipes
    that intentionally pair ``attn_implementation`` (HF submodules, e.g. a VLM vision tower) with a
    different ``backend.attn`` are untouched.

    Args:
        cfg_model: Model config node, mutated in place.
    """
    native = native_backend_from_attn_implementation(cfg_model)
    if native is None:
        return
    backend = getattr(cfg_model, "backend", None)
    if backend is None or not hasattr(backend, "attn"):
        return
    if backend.attn == native:
        return
    logger.info(
        "model.attn_implementation selects the native %r attention backend (was backend.attn=%r).",
        native,
        backend.attn,
    )
    backend.attn = native


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


def configure_packing(attn_implementation: str = "sdpa") -> None:
    """Apply monkey-patches for packed-sequence training with flash attention.

    Only patches when ``attn_implementation`` is a flash-attention variant
    (``flash_attention_2`` / ``flash_attention_3`` / ``flash_attention_4``);
    transformers routes all three through the same varlen wrapper, so the
    ``_get_unpad_data`` patch applies uniformly.

    Args:
        attn_implementation: The attention implementation used by the model.
    """
    if attn_implementation not in _FLASH_ATTN_IMPLEMENTATIONS:
        return

    import sys

    import transformers.modeling_flash_attention_utils

    _patch_preprocess_mask_arguments_for_packing()
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
