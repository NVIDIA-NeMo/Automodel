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

import functools
import gc
import inspect
import logging
import os
import types
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.shared.torch_patches import apply_torch_patches

apply_torch_patches()
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForTextToWaveform,
    PreTrainedModel,
)
from transformers.modeling_utils import _get_resolved_checkpoint_files
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import nemo_automodel.components.distributed.utils as dist_utils
from nemo_automodel import __version__
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.distributed.init_utils import get_local_world_size_preinit, get_world_size_safe
from nemo_automodel.components.utils.model_utils import resolve_trust_remote_code
from nemo_automodel.shared.import_utils import safe_import
from nemo_automodel.shared.utils import dtype_from_str

HAS_LIGER_KERNEL, liger_kernel_trf = safe_import("liger_kernel.transformers")
HAS_FA, _ = safe_import("flash_attn")

logger = logging.getLogger(__name__)


class DownloadKwargs(types.SimpleNamespace):
    """Backwards-compatible container for download-related kwargs.

    Kept as a stable symbol for unit tests / downstream integrations.
    """


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


def _assert_same_signature(original, patched):
    """
    Raise AssertionError if the two call signatures differ.
    """
    sig_orig = inspect.signature(original)
    sig_patch = inspect.signature(patched)

    if sig_orig != sig_patch:
        raise AssertionError(f"Signature mismatch:\n  original: {sig_orig}\n  patched : {sig_patch}")


def _patch_attention(obj, sdpa_method=None):
    """
    Wrap the `forward` method of `obj` in an `sdap_kernel` context manager.

    Args:
        obj: Any object with a `.forward(*args, **kwargs)` method.
        sdpa_method (list[SDPBackend], optional): Ordered list of SDPBackend
            implementations to attempt. If None, defaults to
            [CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH].

    Returns:
        The same `obj` with its `.forward` method patched.
    """
    if sdpa_method is None:
        sdpa_method = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    orig_forward = obj.forward

    def patch_method(method):
        func = method.__func__

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with sdpa_kernel(sdpa_method):
                return func(self, *args, **kwargs)

        wrapper.__doc__ = "SDPA kernel patch\n" + inspect.getdoc(method)
        return types.MethodType(wrapper, method.__self__)  # re-bind

    obj.forward = patch_method(obj.forward)
    # runtime check
    _assert_same_signature(orig_forward, obj.forward)

    logging.info("Patched model with SDPA method= {}".format(sdpa_method))
    return obj


def _patch_liger_kernel(model):
    """
    Patches a model with liger-kernel and sdpa_kernel

    Args:
        model (nn.Module): the model to patch
        use_liger_kernel (bool): Applies liger-kernel to model Default True.
        use_sdpa_patching (bool): Enables model patching with SDPA kernel optim. Default True.
        sdpa_method (list[SDPBackend], optional): Ordered list of SDPBackend
            implementations to attempt. If None, defaults to
            [CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH].
    Returns:
        nn.Module: the patched model
    """
    if not HAS_LIGER_KERNEL:
        logging.warning("Asked to use Liger Kernel, but could not import")
        return model

    # Unit tests may pass lightweight mocks; skip patching in that case.
    # (The wrapper logic itself is tested separately by patching this function.)
    if not isinstance(model, torch.nn.Module):
        logging.warning("Skipping Liger Kernel patch for non-nn.Module model: %s", type(model))
        return model

    try:
        liger_kernel_trf._apply_liger_kernel_to_instance(model=model)
        logging.info("Applied liger-kernel to model")
        return model
    except Exception:
        logging.warning("Failed to apply liger-kernels to model; falling back to eager")
        del model
        raise RuntimeError("Failed to patch model")


def _get_next_fallback_attn(attn_implementation: str) -> str:
    """
    Get the next attention implementation in the priority list, in reverse order.

    If a model does not support a given attention implementation, the next
    implementation in the priority list is returned.

    If the current attention implementation is not in the priority list, it uses eager.

    Args:
        attn_implementation (str): The current attention implementation.

    Returns:
        str: The next attention implementation in the priority list.
    """
    priorities = [
        "eager",
        "sdpa",
        "flash_attention_2",
        "flash_attention_3",
    ]
    if attn_implementation in priorities:
        pos = priorities.index(attn_implementation)
        return priorities[max(0, pos - 1)]
    else:
        return priorities[0]


def get_hf_config(pretrained_model_name_or_path, attn_implementation, kwargs):
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


def _pop_tp_cp_has_packed(kwargs):
    """
    Extract and remove TP/CP/packed flags from kwargs.
    """
    tp_size = kwargs.pop("tp_size", 1)
    cp_size = kwargs.pop("cp_size", 1)
    has_packed_sequence = kwargs.pop("has_packed_sequence", False)
    return tp_size, cp_size, has_packed_sequence


def _apply_preload_overrides(is_hf_model, tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel):
    """
    Compute final attention implementation and liger-kernel flag based on TP/CP and packed sequence constraints.
    """
    if is_hf_model and (tp_size > 1 or cp_size > 1):
        logger.info("Disabling Liger kernel with TP ({}) or CP ({})".format(tp_size, cp_size))
        use_liger_kernel = False

    if cp_size > 1 and is_hf_model:
        attn_implementation = "sdpa"
        logger.warning("Packed sequence is supported only with SDPA. Setting model's attn_implementation to sdpa")

    if is_hf_model and has_packed_sequence:
        if cp_size == 1:
            assert HAS_FA, "Flash Attention is not available"
            attn_implementation = "flash_attention_2"
            logger.warning(
                "Packed sequence is supported only with Flash Attention. "
                "Setting model's attn_implementation to flash_attention_2"
            )
        else:
            # TODO: support packed sequence with CP size > 1
            raise ValueError("Packed sequence is only supported with CP size 1")
    return attn_implementation, use_liger_kernel


def _verify_sdpa_support(model, is_hf_model, cp_size):
    """
    Validate SDPA support when CP is enabled for HF models.
    """
    if cp_size > 1 and is_hf_model and hasattr(model, "_supports_sdpa"):
        if model._supports_sdpa is False:
            raise ValueError("Model does not support SDPA required for context parallelism")


def _download_model_weights(hf_config, pretrained_model_name_or_path):
    if not os.path.isdir(pretrained_model_name_or_path):
        num_nodes = (get_world_size_safe() % get_local_world_size_preinit()) + 1  # 1-indexed
        if num_nodes > 1:
            logging.info(
                f"""Downloading model weights on {num_nodes} nodes. This incurs high storage usage.
                It is recommended to download once with `hf download` and pass in the downloaded path to the `pretrained_model_name_or_path` argument."""
            )
        def _download():
            _get_resolved_checkpoint_files(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                subfolder="",
                variant=None,
                gguf_file=None,
                from_tf=False,
                from_flax=False,
                use_safetensors=None,
                cache_dir=None,
                force_download=False,
                proxies=None,
                local_files_only=False,
                token=None,
                user_agent={"file_type": "model", "framework": "pytorch", "from_auto_class": False},
                revision="main",
                commit_hash=getattr(hf_config, "_commit_hash", None),
                is_remote_code=False,
                transformers_explicit_filename=None,
            )

        # Barrier only makes sense when distributed is initialized.
        # Import via module reference (vs bound name) so unit tests can patch
        # `nemo_automodel.components.distributed.utils.FirstRankPerNode`.
        if torch.distributed.is_initialized():
            with dist_utils.FirstRankPerNode():
                _download()
        else:
            _download()


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


class _BaseNeMoAutoModelClass(_BaseAutoModelClass):
    """
    Drop-in replacement for ``_BaseAutoModelClass`` that includes custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    TODO(@akoumpa): extend this beyond liger_kernel.

    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.
    """

    @classmethod
    def _from_pretrained_parent_class(cls, *args, **kwargs):
        name = cls.__name__
        if name.startswith("NeMo"):
            cls.__name__ = name[4:]
        model = super().from_pretrained(*args, **kwargs)
        cls.__name__ = name
        return model

    @classmethod
    def _from_config_parent_class(cls, *args, **kwargs):
        name = cls.__name__
        if name.startswith("NeMo"):
            cls.__name__ = name[4:]
        model = super().from_config(*args, **kwargs)
        cls.__name__ = name
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: Optional[List[SDPBackend]] = None,
        torch_dtype="auto",
        attn_implementation: str = "flash_attention_2",
        quantization_config=None,
        force_hf: bool = False,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate and (optionally) patch a causal-language model.

        This is a light wrapper around
        `transformers.AutoModelForCausalLM.from_pretrained` that can
        automatically apply Liger and/or SDPA (scaled-dot-product
        attention) kernel optimizations.

        Args:
            pretrained_model_name_or_path (str | os.PathLike): Hugging Face
                hub repo ID or local path accepted by
                `AutoModelForCausalLM.from_pretrained`.
            *model_args: Positional arguments forwarded verbatim to
                `AutoModelForCausalLM.from_pretrained`.
            use_liger_kernel (bool, default=True): If `True`, try to patch
                the model with Liger kernels for faster inference/training.
            use_sdpa_patching (bool, default=True): If `True`, patch the
                model with SDPA-based attention optimizations.
            sdpa_method (list[SDPBackend] | None, optional): Explicit list of
                SDPA back-ends to consider when `use_sdpa_patching=True`.
            torch_dtype (str | torch.dtype | Literal["auto"], default="auto"):
                Data type passed to the underlying `from_pretrained` call.
            attn_implementation (str, default="flash_attention_2"): Desired
                attention implementation; forwarded to the HF config.
            quantization_config (optional): BitsAndBytesConfig configuration object that
                specifies all quantization settings. If provided, quantization
                will be applied to the model.
            force_hf (bool, default=False): If `True`, force the use of HF model implementation.
                If `False`, the model will be loaded using the custom model implementation if available.
            **kwargs: Additional keyword arguments forwarded verbatim to
                `AutoModelForCausalLM.from_pretrained`.

        Returns:
            transformers.PreTrainedModel: The loaded (and possibly patched)
            model instance.

        Warns:
            UserWarning: Emitted when `use_liger_kernel=True` but the Liger
            package is unavailable.

        Notes:
            If kernel patching fails, the partially constructed model is
              deleted and the method recurses once with
              `use_liger_kernel=False` or `use_sdpa_patching=False`
        """
        torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch_dtype
        is_hf_model = get_is_hf_model(
            get_hf_config(pretrained_model_name_or_path, attn_implementation, kwargs),
            force_hf,
        )
        tp_size, cp_size, has_packed_sequence = _pop_tp_cp_has_packed(kwargs)
        attn_implementation, use_liger_kernel = _apply_preload_overrides(
            is_hf_model, tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel
        )
        hf_config = get_hf_config(pretrained_model_name_or_path, attn_implementation, kwargs)

        def _retry(**override):
            """Internal helper to re-enter this function with patched args."""
            kwargs["quantization_config"] = quantization_config
            if "quantization_config" in override:
                if override["quantization_config"] is None:
                    kwargs.pop("quantization_config")
                else:
                    kwargs["quantization_config"] = override["quantization_config"]

            return cls.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=override.get("attn_implementation", attn_implementation),
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                **kwargs,
            )

        # 1. if force_hf is True, we will use the parent class to load and return the model as is
        if force_hf:
            return cls._from_pretrained_parent_class(
                pretrained_model_name_or_path,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                **kwargs,
            )
        architectures = get_architectures(hf_config)
        # 2. If we have a custom model implementation available, we prioritize that over HF
        if len(architectures) > 0 and architectures[0] in ModelRegistry.model_arch_name_to_cls:
            # if we are able to init the custom model, we will now download the model weights on local rank 0
            _download_model_weights(hf_config, pretrained_model_name_or_path)
            logger.info(f"Using custom model implementation for {architectures[0]}")
            kwargs.pop("trust_remote_code", None)
            # TODO(@akoumpa): restore weights after initialization.
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
                return model_cls(hf_config, *model_args, **kwargs)

        # 3. fallback to parent class
        model = None
        try:
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            model = cls._from_pretrained_parent_class(
                pretrained_model_name_or_path,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        except ValueError as e:
            if "does not support" in str(e):
                if model is not None:
                    del model
                attn_implementation = _get_next_fallback_attn(attn_implementation)
                logging.warning("Falling back to {} attention.".format(attn_implementation))
                return _retry(attn_implementation=attn_implementation)
            raise e

        # Kernel patching
        try:
            if use_liger_kernel:
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logging.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        # Patch sdpa attention
        try:
            if use_sdpa_patching:
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except:
            logging.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

        _verify_sdpa_support(model, is_hf_model, cp_size)

        model.config.update({"nemo_version": __version__})
        return model

    @classmethod
    def from_config(
        cls,
        config,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: Optional[List[SDPBackend]] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "flash_attention_2",
        quantization_config=None,
        force_hf: bool = False,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate a model from a ``transformers.PretrainedConfig`` and optionally
        patch it with Liger or SDPA-optimized kernels.

        Args:
            config (transformers.PretrainedConfig | str):
                The configuration object used to build the model.
                If config is passed as a string (e.g., model-id / local checkpoint),
                it will be create a config internally using AutoConfig.
            *model_args:
                Positional arguments forwarded to the underlying
                ``transformers.AutoModelForCausalLM.from_config`` call.
            use_liger_kernel (bool, optional):
                If ``True``, tries to patch the instantiated model with Liger
                optimized attention kernels. Defaults to ``True``.
            use_sdpa_patching (bool, optional):
                If ``True``, applies in-place SDPA (Scaled-Dot-Product-Attention)
                kernel optimizations wherever possible. Defaults to ``True``.
            sdpa_method (Optional[List[SDPBackend]], optional):
                One or multiple SDPA back-ends to prefer when applying SDPA
                patching. When ``None``, the default backend resolution logic is
                used. Defaults to ``None``.
            attn_implementation (str, optional):
                Specifies which attention implementation to use (e.g.,
                ``"flash_attention_2"``, ``"eager"``). Only applied when the
                base model supports this kwarg. Defaults to
                ``"flash_attention_2"``.
            force_hf (bool, default=False): If `True`, force the use of HF model implementation.
                If `False`, the model will be loaded using the custom model implementation if available.
            **kwargs:
                Additional keyword arguments forwarded to the superclass
                constructor and underlying ``from_config`` logic.

        Returns:
            transformers.PreTrainedModel: The instantiated (and possibly
            kernel-patched) model.

        Notes:
            If kernel patching fails, the partially constructed model is
              deleted and the method recurses once with
              `use_liger_kernel=False` or `use_sdpa_patching=False`
        """
        torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch.bfloat16
        kwargs["trust_remote_code"] = kwargs.get(
            "trust_remote_code", resolve_trust_remote_code(getattr(config, "name_or_path", None))
        )

        architectures = getattr(config, "architectures", None) or []
        is_hf_model = (not architectures or architectures[0] not in ModelRegistry.model_arch_name_to_cls) or force_hf
        tp_size, cp_size, has_packed_sequence = _pop_tp_cp_has_packed(kwargs)
        attn_implementation, use_liger_kernel = _apply_preload_overrides(
            is_hf_model, tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel
        )

        def _retry(**override):
            """Internal helper to re-enter this function with patched args."""
            if "quantization_config" in override:
                if override["quantization_config"] is None:
                    kwargs.pop("quantization_config")
                else:
                    kwargs["quantization_config"] = override["quantization_config"]
            return cls.from_config(
                config,
                *model_args,
                attn_implementation=override.get("attn_implementation", attn_implementation),
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                torch_dtype=torch_dtype,
                **kwargs,
            )

        # handle model_id passed as config
        if isinstance(config, str):
            config = AutoConfig.from_pretrained(
                config,
                trust_remote_code=kwargs.get("trust_remote_code", False),
                attn_implementation=attn_implementation,
            )
        # Treat config-related kwargs (e.g., output_hidden_states) as overrides on the
        # config object, not as model __init__ kwargs.
        _consume_config_overrides(config, kwargs)
        # 1. if force_hf is True, we will use the parent class to load and return the model as is
        if force_hf:
            return cls._from_config_parent_class(
                config,
                *model_args,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                **kwargs,
            )

        # 2. If we have a custom model implementation available, we prioritize that over HF
        architectures = get_architectures(config)
        if len(architectures) > 0 and architectures[0] in ModelRegistry.model_arch_name_to_cls:
            model_cls = ModelRegistry.model_arch_name_to_cls[architectures[0]]
            init_param_names = _get_init_param_names(model_cls)
            _consume_config_overrides(config, kwargs, init_param_names=init_param_names)
            kwargs = _filter_kwargs_for_init(model_cls, kwargs)
            with local_torch_dtype(torch_dtype, model_cls.__name__):
                return model_cls(config, *model_args, **kwargs)

        # 3. fallback to parent class
        model = None
        try:
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            model = cls._from_config_parent_class(
                config,
                *model_args,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                **kwargs,
            )
        except ValueError as e:
            if "does not support" in str(e):
                logging.warning("Falling back to eager attention.")
                return _retry(attn_implementation="eager")
            raise e

        # Kernel patching
        try:
            if use_liger_kernel:
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logging.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        # Patch sdpa attention
        try:
            if use_sdpa_patching:
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except:
            logging.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

        _verify_sdpa_support(model, is_hf_model, cp_size)

        model.config.update({"nemo_version": __version__})
        return model


class NeMoAutoModelForCausalLM(_BaseNeMoAutoModelClass, AutoModelForCausalLM):
    """
    Drop-in replacement for ``transformers.AutoModelForCausalLM`` that includes custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    TODO(@akoumpa): extend this beyond liger_kernel.

    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples:
    --------
    >>> model = NeMoAutoModelForCausalLM.from_pretrained("gpt2")            # try Liger
    >>> model = NeMoAutoModelForCausalLM.from_pretrained(
    ...     "gpt2", use_liger_kernel=False)                                 # skip Liger
    """

    pass


class NeMoAutoModelForImageTextToText(_BaseNeMoAutoModelClass, AutoModelForImageTextToText):
    """Drop-in replacement for ``transformers.AutoModelForImageTextToText`` with custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    @akoumpa: currently only supporting liger_kernel for demonstration purposes.

    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples:
    --------
    >>> model = NeMoAutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct") # try Liger
    >>> model = NeMoAutoModelForImageTextToText.from_pretrained(
    ...     "Qwen/Qwen2.5-VL-3B-Instruct", use_liger_kernel=False)                            # skip Liger
    """

    pass


class NeMoAutoModelForSequenceClassification(_BaseNeMoAutoModelClass, AutoModelForSequenceClassification):
    """Drop-in replacement for ``transformers.AutoModelForSequenceClassification`` with custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    @akoumpa: currently only supporting liger_kernel for demonstration purposes.

    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples:
    --------
    >>> model = NeMoAutoModelForSequenceClassification.from_pretrained("bert-base-uncased") # try Liger
    >>> model = NeMoAutoModelForSequenceClassification.from_pretrained(
    ...     "bert-base-uncased", use_liger_kernel=False)                            # skip Liger
    """

    pass


class NeMoAutoModelForTextToWaveform(_BaseNeMoAutoModelClass, AutoModelForTextToWaveform):
    """Drop-in replacement for ``transformers.AutoModelForTextToWaveform`` with custom-kernels.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    @akoumpa: currently only supporting liger_kernel for demonstration purposes.

    Notes:
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples:
    --------
    >>> model = NeMoAutoModelForTextToWaveform.from_pretrained("facebook/musicgen-small") # try Liger
    >>> model = NeMoAutoModelForTextToWaveform.from_pretrained(
    ...     "facebook/musicgen-small", use_liger_kernel=False)                            # skip Liger
    """

    pass
