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

import logging
import types
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from nemo_automodel.package_info import __version__

from nemo_automodel.shared.import_utils import safe_import
from torch.nn.attention import sdpa_kernel, SDPBackend

HAS_LIGER_KERNEL, liger_kernel_trf = safe_import("liger_kernel.transformers")
logger = logging.getLogger(__name__)


def patch_attention(obj, sdpa_method=None):
    """
    Wrap the `forward` method of `obj` in an `sdap_kernel` context to
    enable a sequence of SDP attention backends.

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

    def patched_forward(self, *args, **kwargs):
        with sdpa_kernel(sdpa_method):
            return orig_forward(*args, **kwargs)

    obj.forward = types.MethodType(patched_forward, obj)
    return obj


class NeMoAutoModelForCausalLM(AutoModelForCausalLM):
    """
    Drop-in replacement for ``transformers.AutoModelForCausalLM`` that can
    transparently patch the loaded model with NVIDIA Liger fused-attention
    kernels for higher inference throughput.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    @akoumpa: currently only supporting liger_kernel for demonstration purposes.

    Notes
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples
    --------
    >>> model = NeMoAutoModelForCausalLM.from_pretrained("gpt2")            # try Liger
    >>> model = NeMoAutoModelForCausalLM.from_pretrained(
    ...     "gpt2", use_liger_kernel=False)                                 # skip Liger
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained causal-language-model and (optionally) patch it with
        Liger fused-attention kernels.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Repository ID or local path accepted by
            ``transformers.AutoModelForCausalLM.from_pretrained``.
        *model_args
            Positional arguments forwarded verbatim to the superclass.
        use_liger_kernel : bool, default True
            Whether to attempt patching the loaded model with Liger kernels.
        **kwargs
            Keyword arguments forwarded verbatim to the superclass.

        Returns
        -------
        transformers.PreTrainedModel
            The instantiated model, possibly Liger-patched.

        Warnings
        --------
        Emits a ``logging.warning`` if ``use_liger_kernel=True`` but the Liger
        package is not available.

        Retries
        -------
        If patching raises an exception, the method deletes the partially
        constructed model and recursively reloads it once with
        ``use_liger_kernel=False``.
        """
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        use_liger_kernel = kwargs.pop("use_liger_kernel", True)
        sdpa_method = kwargs.pop("sdpa_method", None)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
            torch_dtype=torch_dtype,
        )
        if use_liger_kernel:
            if not HAS_LIGER_KERNEL:
                logging.warning("Asked to use Liger Kernel, but could not import")
                return model
            try:
                liger_kernel_trf._apply_liger_kernel_to_instance(model=model)
            except Exception:
                del model
                # If patching failed, retry
                return cls.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    **kwargs,
                    torch_dtype=torch_dtype,
                    use_liger_kernel=False,
                )
        model = patch_attention(model, sdpa_method)
        model.config.update({"nemo_version": __version__})
        return model

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Instantiate a model from a config object and (optionally) patch it with
        Liger fused-attention kernels.

        Parameters
        ----------
        config : transformers.PretrainedConfig
            Configuration used to build the model.
        use_liger_kernel : bool, default True
            Whether to attempt patching the instantiated model with Liger
            kernels.
        **kwargs
            Additional keyword arguments forwarded to the superclass.

        Returns
        -------
        transformers.PreTrainedModel
            The instantiated model, possibly Liger-patched.

        See Also
        --------
        NeMoAutoModelForCausalLM.from_pretrained : Same logic for checkpoint
        loading.
        """
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        use_liger_kernel = kwargs.pop("use_liger_kernel", True)
        sdpa_method = kwargs.pop("sdpa_method", None)
        model = super().from_config(config, **kwargs, torch_dtype=torch_dtype)
        if use_liger_kernel:
            if not HAS_LIGER_KERNEL:
                logging.warning("Asked to use Liger Kernel, but could not import")
                return model
            try:
                liger_kernel_trf._apply_liger_kernel_to_instance(model=model)
            except Exception:
                del model
                # If patching failed, retry
                return cls.from_config(
                    config, **kwargs, use_liger_kernel=False, torch_dtype=torch_dtype
                )
        model = patch_attention(model, sdpa_method)
        model.config.update({"nemo_version": __version__})
        return model


class NeMoAutoModelForImageTextToText(AutoModelForImageTextToText):
    """
    Drop-in replacement for ``transformers.AutoModelForImageTextToText`` that can
    transparently patch the loaded model with NVIDIA Liger fused-attention
    kernels for higher inference throughput.

    The class only overrides ``from_pretrained`` and ``from_config`` to add the
    optional ``use_liger_kernel`` flag.  If the flag is ``True`` (default) and
    the Liger kernel is available, the model's attention layers are
    monkey-patched in place.  If patching fails for any reason, the call is
    retried once with ``use_liger_kernel=False`` so that users still obtain a
    functional model.


    @akoumpa: currently only supporting liger_kernel for demonstration purposes.

    Notes
    -----
    - No changes are made to the model's public API; forward signatures,
      generation utilities, and weight shapes remain identical.
    - Only decoder-style (causal) architectures are currently supported by the
      Liger patch.  Unsupported models will silently fall back.

    Examples
    --------
    >>> model = NeMoAutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct") # try Liger
    >>> model = NeMoAutoModelForImageTextToText.from_pretrained(
    ...     "Qwen/Qwen2.5-VL-3B-Instruct", use_liger_kernel=False)                            # skip Liger
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained causal-language-model and (optionally) patch it with
        Liger fused-attention kernels.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Repository ID or local path accepted by
            ``transformers.AutoModelForCausalLM.from_pretrained``.
        *model_args
            Positional arguments forwarded verbatim to the superclass.
        use_liger_kernel : bool, default True
            Whether to attempt patching the loaded model with Liger kernels.
        **kwargs
            Keyword arguments forwarded verbatim to the superclass.

        Returns
        -------
        transformers.PreTrainedModel
            The instantiated model, possibly Liger-patched.

        Warnings
        --------
        Emits a ``logging.warning`` if ``use_liger_kernel=True`` but the Liger
        package is not available.

        Retries
        -------
        If patching raises an exception, the method deletes the partially
        constructed model and recursively reloads it once with
        ``use_liger_kernel=False``.
        """
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        use_liger_kernel = kwargs.pop("use_liger_kernel", True)
        sdpa_method = kwargs.pop("sdpa_method", None)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
            torch_dtype=torch_dtype,
        )
        if use_liger_kernel:
            if not HAS_LIGER_KERNEL:
                logging.warning("Asked to use Liger Kernel, but could not import")
                return model
            try:
                liger_kernel_trf._apply_liger_kernel_to_instance(model=model)
            except Exception:
                del model
                # If patching failed, retryd
                return cls.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    **kwargs,
                    use_liger_kernel=False,
                    torch_dtype=torch_dtype,
                )
        model = patch_attention(model, sdpa_method)
        model.config.update({"nemo_version": __version__})
        return model

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Instantiate a model from a config object and (optionally) patch it with
        Liger fused-attention kernels.

        Parameters
        ----------
        config : transformers.PretrainedConfig
            Configuration used to build the model.
        use_liger_kernel : bool, default True
            Whether to attempt patching the instantiated model with Liger
            kernels.
        **kwargs
            Additional keyword arguments forwarded to the superclass.

        Returns
        -------
        transformers.PreTrainedModel
            The instantiated model, possibly Liger-patched.

        See Also
        --------
        NeMoAutoModelForImageTextToText.from_pretrained : Same logic for checkpoint
        loading.
        """
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        use_liger_kernel = kwargs.pop("use_liger_kernel", True)
        sdpa_method = kwargs.pop("sdpa_method", None)
        model = super().from_config(config, **kwargs, torch_dtype=torch_dtype)
        if use_liger_kernel:
            if not HAS_LIGER_KERNEL:
                logging.warning("Asked to use Liger Kernel, but could not import")
                return model
            try:
                liger_kernel_trf._apply_liger_kernel_to_instance(model=model)
            except Exception:
                del model
                # If patching failed, retry
                return cls.from_config(
                    config, **kwargs, use_liger_kernel=False, torch_dtype=torch_dtype
                )
        model = patch_attention(model, sdpa_method)
        model.config.update({"nemo_version": __version__})
        return model
