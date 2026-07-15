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

import torch
import torch.nn as nn


def get_controlling_tie_word_embeddings(config: object, model_class_name: str) -> bool:
    """Resolve the ``tie_word_embeddings`` flag that actually controls LM-head tying.

    Hugging Face ties ``lm_head`` based on the top-level config flag, not a
    nested ``text_config``. Omni thinker models are the exception: their full
    wrapper config stores the controlling flag under ``thinker_config``.

    Args:
        config: Model config exposing the relevant tying flag.
        model_class_name: Name of the model class that owns the config.

    Returns:
        The controlling ``tie_word_embeddings`` value.
    """
    omni_thinker_models = (
        "Qwen2_5OmniThinkerForConditionalGeneration",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    )
    if any(name in model_class_name for name in omni_thinker_models):
        thinker_config = getattr(config, "thinker_config", config)
        return bool(getattr(thinker_config, "tie_word_embeddings", False))

    composite_top_level_models = (
        "Mistral3FP8VLMForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    )
    if any(name in model_class_name for name in composite_top_level_models):
        return bool(getattr(config, "tie_word_embeddings", False))

    if hasattr(config, "tie_word_embeddings"):
        return bool(config.tie_word_embeddings)

    text_config = getattr(config, "get_text_config", lambda: None)()
    return bool(getattr(text_config, "tie_word_embeddings", False))


def is_tied_word_embeddings(model: nn.Module) -> bool:
    """Return whether the model config requests tied input/output embeddings."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    return get_controlling_tie_word_embeddings(config, type(model).__name__)


def _normalize_param_name(name: str) -> str:
    """Strip wrapper-specific prefixes from a parameter name."""
    return name.replace("_orig_mod.", "")


def get_lm_head_weight_and_name(model: nn.Module) -> tuple[torch.Tensor | None, str | None]:
    """Return the first ``lm_head.weight`` parameter and its normalized FQN."""
    for name, param in model.named_parameters(remove_duplicate=False):
        normalized_name = _normalize_param_name(name)
        if "lm_head" in normalized_name and normalized_name.endswith(".weight"):
            return param, normalized_name
    return None, None


def get_input_embeddings_weight_and_name(model: nn.Module) -> tuple[torch.Tensor | None, str | None]:
    """Return the input embedding weight and normalized FQN if locally present."""
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        try:
            input_embeddings = get_input_embeddings()
        except Exception:
            input_embeddings = None
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            for name, param in model.named_parameters(remove_duplicate=False):
                if param is input_embeddings.weight:
                    return param, _normalize_param_name(name)

    candidate_suffixes = (
        "embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    )
    for name, param in model.named_parameters(remove_duplicate=False):
        normalized_name = _normalize_param_name(name)
        if normalized_name.endswith(candidate_suffixes):
            return param, normalized_name
    return None, None


def _same_tensor_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors alias the same local storage."""
    if left is right:
        return True

    def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        to_local = getattr(tensor, "to_local", None)
        if callable(to_local):
            try:
                return to_local()
            except RuntimeError:
                return tensor
        return tensor

    left_local = _local_tensor(left)
    right_local = _local_tensor(right)
    if left_local is right_local:
        return True
    if left_local.device.type == "meta" or right_local.device.type == "meta":
        return False

    try:
        return (
            left_local.untyped_storage().data_ptr() == right_local.untyped_storage().data_ptr()
            and left_local.storage_offset() == right_local.storage_offset()
        )
    except RuntimeError:
        return False


def has_local_tied_lm_head(model: nn.Module) -> bool:
    """Return whether local input/output embedding tensors actually share storage.

    This is stricter than :func:`is_tied_word_embeddings`: pipeline stages can
    retain a tied config flag without owning both tensors, and speculative draft
    models can intentionally use different vocabulary sizes.
    """
    if not is_tied_word_embeddings(model):
        return False
    lm_head_weight, _ = get_lm_head_weight_and_name(model)
    input_embeddings_weight, _ = get_input_embeddings_weight_and_name(model)
    if lm_head_weight is None or input_embeddings_weight is None:
        return False
    if tuple(lm_head_weight.shape) != tuple(input_embeddings_weight.shape):
        return False
    return _same_tensor_storage(lm_head_weight, input_embeddings_weight)


def _get_module_by_normalized_name(model: nn.Module, normalized_module_name: str) -> nn.Module | None:
    """Return a module by FQN after applying wrapper-prefix normalization."""
    if normalized_module_name == "":
        return model
    for name, module in model.named_modules():
        if _normalize_param_name(name) == normalized_module_name:
            return module
    return None


def ensure_tied_lm_head(model: nn.Module) -> bool:
    """Ensure a local tied LM head actually aliases the input embedding.

    Hugging Face ``tie_weights()`` is attempted first so model-specific tying
    rules remain authoritative. Direct assignment is the fallback for wrapped
    models where the generic method no longer reaches the local pair.

    Args:
        model: Model or pipeline stage to inspect and update.

    Returns:
        ``True`` if the local LM head and input embedding are tied afterward.
    """
    if not is_tied_word_embeddings(model):
        return False
    if has_local_tied_lm_head(model):
        return True

    lm_head_weight, _ = get_lm_head_weight_and_name(model)
    input_embeddings_weight, _ = get_input_embeddings_weight_and_name(model)
    if lm_head_weight is not None and input_embeddings_weight is not None:
        if tuple(lm_head_weight.shape) != tuple(input_embeddings_weight.shape):
            return False

    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        try:
            tie_weights()
        except AttributeError:
            pass
        if has_local_tied_lm_head(model):
            return True

    lm_head_weight, lm_head_param_name = get_lm_head_weight_and_name(model)
    input_embeddings_weight, _ = get_input_embeddings_weight_and_name(model)
    if lm_head_weight is None or lm_head_param_name is None or input_embeddings_weight is None:
        return False
    if tuple(lm_head_weight.shape) != tuple(input_embeddings_weight.shape):
        return False

    lm_head_module_name = lm_head_param_name.rsplit(".", 1)[0]
    lm_head_module = _get_module_by_normalized_name(model, lm_head_module_name)
    if lm_head_module is None or not hasattr(lm_head_module, "weight"):
        return False

    try:
        lm_head_module.weight = input_embeddings_weight
    except (AttributeError, TypeError, RuntimeError):
        return False

    return has_local_tied_lm_head(model)
