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

"""Model resolution and initialization helpers.

Functions for resolving which model class to use (custom vs HF), downloading
weights, applying config overrides, and instantiating the model.
"""

import inspect
import logging
import os
from contextlib import contextmanager

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

import nemo_automodel.components.distributed.utils as dist_utils
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.distributed.init_utils import get_local_world_size_preinit, get_world_size_safe
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.utils.model_utils import resolve_trust_remote_code
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)


def _get_mixin_wrapped_class(model_class: type) -> type:
    """
    Get a class that combines HFCheckpointingMixin with the original model class.

    If the class already has the mixin, returns it unchanged.

    Args:
        model_class: The original model class (e.g., LlamaForCausalLM)

    Returns:
        A class that inherits from both HFCheckpointingMixin and model_class
    """
    # Custom models already inherit HFCheckpointingMixin
    if issubclass(model_class, HFCheckpointingMixin):
        return model_class

    # Create wrapper class that looks identical to original
    return type(
        model_class.__name__,
        (HFCheckpointingMixin, model_class),
        {
            "__module__": model_class.__module__,
            "__qualname__": model_class.__qualname__,
        },
    )


@contextmanager
def local_torch_dtype(
    dtype: torch.dtype, model_class_name: str | None = None, default_dtype: torch.dtype = torch.bfloat16
):
    """
    Locally change the torch default dtype to `dtype`, and restore the old one upon exiting the context.
    If `model_class_name` is provided, it's used to provide a more helpful error message if `dtype` is not valid.
    """
    # Just a more helping error before we set `torch.set_default_dtype` later on which would crash in this case
    if isinstance(dtype, str):
        dtype = default_dtype
    if not dtype.is_floating_point:
        if model_class_name is not None:
            error_message = (
                f"{model_class_name} cannot be instantiated under `dtype={dtype}` as it's not a floating-point dtype"
            )
        else:
            error_message = f"Cannot set `{dtype}` as torch's default as it's not a floating-point dtype"
        raise ValueError(error_message)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def _is_config_compatible_with_custom_model(arch_name: str, config) -> bool:
    """
    Check if a HuggingFace config is compatible with our custom model implementation.

    Some architectures (e.g., NemotronHForCausalLM) are shared between different model versions
    (v2 vs v3) but our custom implementation only supports specific versions. This function
    validates that the config has the required attributes for the custom implementation.

    Args:
        arch_name: The architecture name (e.g., "NemotronHForCausalLM")
        config: The HuggingFace config object

    Returns:
        True if the config is compatible with our custom implementation, False otherwise
    """
    # NemotronHForCausalLM: Our custom implementation is for v3 (MoE model)
    # v3 requires n_routed_experts, v2 does not have this attribute
    if arch_name == "NemotronHForCausalLM":
        return hasattr(config, "n_routed_experts") and config.n_routed_experts is not None

    # All other architectures are assumed compatible
    return True


def get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs):
    """
    Get the HF config for the model.
    """
    kwargs = kwargs.copy()
    trust_remote_code = kwargs.pop("trust_remote_code", resolve_trust_remote_code(pretrained_model_name_or_path))
    hf_config = kwargs.get("config", None)
    if hf_config is None:
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
    return hf_config


def get_is_hf_model(config, force_hf):
    """
    Resolve trust_remote_code default and determine if model is HF-based.
    """
    # Finally make sure flash_attention is available
    architectures = getattr(config, "architectures", None) or []
    is_hf_model = (not architectures or architectures[0] not in ModelRegistry.model_arch_name_to_cls) or force_hf
    return is_hf_model


def _download_model_weights(hf_config, pretrained_model_name_or_path):
    if not os.path.isdir(pretrained_model_name_or_path):
        num_nodes = (get_world_size_safe() % get_local_world_size_preinit()) + 1  # 1-indexed
        if num_nodes > 1:
            logger.info(
                "Downloading model weights on %d nodes. This incurs high storage usage. "
                "It is recommended to download once with `hf download` and pass in the "
                "downloaded path to the `pretrained_model_name_or_path` argument.",
                num_nodes,
            )
        # Import via module reference (vs bound name) so unit tests can patch
        # `nemo_automodel.components.distributed.utils.FirstRankPerNode`.
        with dist_utils.FirstRankPerNode():
            snapshot_download(pretrained_model_name_or_path)


def _init_model(
    cls,
    pretrained_model_name_or_path_or_config,
    attn_implementation,
    torch_dtype,
    quantization_config,
    force_hf,
    *model_args,
    **kwargs,
):
    torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch_dtype
    is_pretrained_init = isinstance(pretrained_model_name_or_path_or_config, str)  # The caller is .from_pretrained
    hf_config = (
        get_hf_config(pretrained_model_name_or_path_or_config, attn_implementation, **kwargs)
        if is_pretrained_init
        else pretrained_model_name_or_path_or_config
    )
    pretrained_model_name_or_path = (
        pretrained_model_name_or_path_or_config if is_pretrained_init else getattr(hf_config, "name_or_path")
    )
    # for models still access config.pad_token_id after v5 removes it in PretrainedConfig
    if not hasattr(hf_config, "pad_token_id"):
        hf_config.pad_token_id = None

    # 1. if force_hf is True, use HF model class wrapped with mixin
    if force_hf:
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        if is_pretrained_init:
            kwargs["config"] = hf_config
            model = cls._from_pretrained_parent_class(
                pretrained_model_name_or_path,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        else:
            model = cls._from_config_parent_class(
                hf_config,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        # Get HF model class and wrap with mixin
        try:
            hf_model_cls = cls._model_mapping[type(hf_config)]
        except KeyError:
            hf_model_cls = type(model)
        model.__class__ = _get_mixin_wrapped_class(hf_model_cls)
        return False, model

    architectures = get_architectures(hf_config)
    # 2. If we have a custom model implementation available, we prioritize that over HF
    if len(architectures) > 0 and architectures[0] in ModelRegistry.model_arch_name_to_cls:
        # if we are able to init the custom model, we will now download the model weights on local rank 0
        # Skip download for from_config (no pretrained path) or local paths
        if pretrained_model_name_or_path:
            _download_model_weights(hf_config, pretrained_model_name_or_path)
        logger.info(f"Using custom model implementation for {architectures[0]}")
        kwargs.pop("trust_remote_code", None)
        model_cls = ModelRegistry.model_arch_name_to_cls[architectures[0]]
        # Treat config-related kwargs as config overrides (HF behavior) and
        # avoid forwarding them into model __init__.
        init_param_names = _get_init_param_names(model_cls)
        _consume_config_overrides(hf_config, kwargs, init_param_names=init_param_names)
        kwargs = _filter_kwargs_for_init(model_cls, kwargs)
        # Override config's torch_dtype with user-requested dtype so model __init__ uses correct dtype
        if torch_dtype != "auto":
            hf_config.torch_dtype = torch_dtype
        with local_torch_dtype(torch_dtype, model_cls.__name__):
            return True, model_cls(hf_config, *model_args, **kwargs)

    # 3. fallback to HF model class wrapped with mixin

    model = None
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if is_pretrained_init:
        # Forward the (possibly patched) config so from_pretrained doesn't
        # reload a fresh copy that would lack the pad_token_id fix above.
        kwargs["config"] = hf_config
        model = cls._from_pretrained_parent_class(
            pretrained_model_name_or_path,
            *model_args,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
    else:
        model = cls._from_config_parent_class(
            hf_config,
            *model_args,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    try:
        hf_model_cls = cls._model_mapping[type(hf_config)]
    except KeyError:
        hf_model_cls = type(model)
    model.__class__ = _get_mixin_wrapped_class(hf_model_cls)
    return False, model


def get_architectures(hf_config):
    """
    Get the architectures from the HF config.
    """
    architectures = []
    if hasattr(hf_config, "architectures"):
        architectures = hf_config.architectures or []
    return architectures


def _get_init_param_names(model_cls) -> set[str]:
    """
    Best-effort extraction of explicit __init__ parameter names (excluding `self`).

    Returns an empty set if the signature cannot be inspected.
    """
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {k for k in sig.parameters.keys() if k != "self"}


def _consume_config_overrides(config, kwargs: dict, *, init_param_names: set[str] | None = None) -> None:
    """
    Mimic HF from_pretrained behavior: treat config-related kwargs as config overrides,
    not model __init__ kwargs.

    For custom model implementations we instantiate via `model_cls(config, **kwargs)`,
    so passing config flags like `output_hidden_states` would crash. This helper moves
    such keys onto the config and removes them from `kwargs`.
    """
    if init_param_names is None:
        init_param_names = set()
    # Prefer `to_dict()` to capture the canonical set of config fields.
    try:
        config_keys = set(config.to_dict().keys())
    except Exception:
        config_keys = set(getattr(config, "__dict__", {}).keys())

    for k in list(kwargs.keys()):
        # If the model explicitly declares this kwarg, keep it for __init__.
        if k in init_param_names:
            continue
        # Otherwise, if it looks like a config field, apply it to config.
        if k in config_keys:
            setattr(config, k, kwargs.pop(k))


def _filter_kwargs_for_init(model_cls, kwargs: dict) -> dict:
    """
    Filter kwargs down to what `model_cls.__init__` explicitly accepts.

    If the constructor has a `**kwargs` parameter (VAR_KEYWORD) or signature cannot be
    inspected, returns kwargs unchanged.
    """
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    # We pass `config` positionally.
    allowed.discard("config")
    return {k: v for k, v in kwargs.items() if k in allowed}
