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
import inspect
import logging
import types
from typing import List, Optional, Union

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from nemo_automodel import __version__
from nemo_automodel.shared.import_utils import safe_import
from nemo_automodel.shared.utils import dtype_from_str

HAS_LIGER_KERNEL, liger_kernel_trf = safe_import("liger_kernel.transformers")
logger = logging.getLogger(__name__)


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
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: Optional[List[SDPBackend]] = None,
        torch_dtype="auto",
        attn_implementation: str = "flash_attention_2",
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
            fp8_config (FP8Config, optional): FP8 configuration object that
                specifies all FP8 quantization settings. If provided, FP8 quantization
                will be applied to the model for improved performance on supported hardware.
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

        def _retry(**override):
            """Internal helper to re-enter this function with patched args."""
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

        # load model
        model = None
        try:
            name = cls.__name__
            if name.startswith("NeMo"):
                cls.__name__ = name[4:]
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
            cls.__name__ = name
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
            return _retry(use_liger_kernel=False)

        # Patch sdpa attention
        try:
            if use_sdpa_patching:
                model = _patch_attention(model, sdpa_method)
        except:
            logging.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

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
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate a model from a ``transformers.PretrainedConfig`` and optionally
        patch it with Liger or SDPA-optimized kernels.

        Args:
            config (transformers.PretrainedConfig):
                The configuration object used to build the model.
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

        def _retry(**override):
            """Internal helper to re-enter this function with patched args."""
            return cls.from_config(
                config,
                *model_args,
                attn_implementation=override.get("attn_implementation", attn_implementation),
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                **kwargs,
            )

        # load model
        try:
            name = cls.__name__
            if name.startswith("NeMo"):
                cls.__name__ = name[4:]
            model = super().from_config(
                config,
                *model_args,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            cls.__name__ = name
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
            return _retry(use_liger_kernel=False)

        # Patch sdpa attention
        try:
            if use_sdpa_patching:
                model = _patch_attention(model, sdpa_method)
        except:
            logging.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

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
