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

"""NeMo Auto Model classes.

Drop-in replacements for ``transformers.AutoModelFor*`` that add custom-kernel
patching, distributed infrastructure, PEFT, quantization, and checkpointing.

Heavy-lifting helpers live in sibling modules:

* ``kernel_patches`` -- SDPA / Liger kernel patching
* ``model_init`` -- model class resolution and instantiation
* ``infrastructure`` -- MeshContext, sharding, PEFT/quant application
"""

import gc
import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch.nn.attention import SDPBackend

from nemo_automodel.shared.torch_patches import apply_torch_patches

apply_torch_patches()
from huggingface_hub import constants as hf_constants  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForTextToWaveform,
    PreTrainedModel,
)
from transformers.initialization import no_init_weights  # noqa: E402
from transformers.models.auto.auto_factory import _BaseAutoModelClass  # noqa: E402
from transformers.utils import ContextManagers  # noqa: E402

from nemo_automodel.components.distributed.config import (  # noqa: E402
    DistributedConfig,
)
from nemo_automodel.components.distributed.ddp import DDPManager  # noqa: E402
from nemo_automodel.components.distributed.init_utils import get_world_size_safe  # noqa: E402
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager  # noqa: E402
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig  # noqa: E402
from nemo_automodel.components.moe.config import MoEParallelizerConfig  # noqa: E402
from nemo_automodel.components.quantization.qat import QATConfig  # noqa: E402
from nemo_automodel.components.utils.model_utils import (  # noqa: E402
    init_empty_weights,
    resolve_trust_remote_code,
)
from nemo_automodel.shared.utils import dtype_from_str  # noqa: E402

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.quantization.fp8 import FP8Config
    from nemo_automodel.components.utils.compile_utils import CompileConfig

#  Re-exports from sibling modules (backward compatibility)
# Backward-compat shim for trust_remote_code models (e.g. DeciLM)
# that import NEED_SETUP_CACHE_CLASSES_MAPPING from transformers.generation.utils.
import transformers.generation.utils as _gen_utils  # noqa: E402

from nemo_automodel._transformers.infrastructure import (  # noqa: E402, F401
    MeshContext,
    _apply_peft_and_lower_precision,
    _shard_ep_fsdp,
    _shard_pp,
    apply_model_infrastructure,
    instantiate_infrastructure,
    parallelize_for_pp,
)
from nemo_automodel._transformers.kernel_patches import (  # noqa: E402, F401
    DEFAULT_ATTN_IMPLEMENTATION,
    HAS_FA,
    HAS_LIGER_KERNEL,
    _apply_preload_overrides,
    _assert_same_signature,
    _get_next_fallback_attn,
    _patch_attention,
    _patch_liger_kernel,
    _verify_sdpa_support,
)
from nemo_automodel._transformers.model_init import (  # noqa: E402, F401
    _consume_config_overrides,
    _download_model_weights,
    _filter_kwargs_for_init,
    _get_init_param_names,
    _get_mixin_wrapped_class,
    _init_model,
    _is_config_compatible_with_custom_model,
    get_architectures,
    get_hf_config,
    get_is_hf_model,
    local_torch_dtype,
)

if not hasattr(_gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
    from transformers.cache_utils import StaticCache

    _gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache}


logger = logging.getLogger(__name__)


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
    def _build_model(
        cls,
        pretrained_model_name_or_path_or_config,
        *model_args,
        is_hf_model,
        use_liger_kernel,
        use_sdpa_patching,
        sdpa_method,
        torch_dtype,
        attn_implementation,
        quantization_config,
        force_hf,
        model_wrapper,
        autopipeline,
        parallelize_fn,
        qat_quantizer,
        mesh,
        loss_fn,
        peft_config,
        fp8_config,
        compile_config,
        load_base_model,
        **kwargs,
    ):
        """Shared model building logic for ``from_pretrained`` and ``from_config``.

        Handles pre-load overrides, meta-device initialization, model init with
        attention-fallback retry, kernel patching (Liger, SDPA) with retry, and
        full infrastructure application (sharding, PEFT, quantization, checkpointing).

        All caller-specific setup (config resolution, infrastructure instantiation,
        ``is_hf_model`` determination) is done by ``from_pretrained`` / ``from_config``
        before delegating here.
        """
        # Extract values consumed by pop; preserve them for retry.
        kwargs = dict(kwargs)  # Defensive copy so retries get clean state
        has_packed_sequence = kwargs.pop("has_packed_sequence", False)
        freeze_config = kwargs.pop("freeze_config", None)
        cache_dir = kwargs.pop("cache_dir", hf_constants.HF_HUB_CACHE)

        def _retry(**override):
            """Re-enter ``_build_model`` with overridden parameters."""
            retry_kwargs = {
                **kwargs,
                "has_packed_sequence": has_packed_sequence,
                "freeze_config": freeze_config,
                "cache_dir": cache_dir,
            }
            return cls._build_model(
                pretrained_model_name_or_path_or_config,
                *model_args,
                is_hf_model=is_hf_model,
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                torch_dtype=torch_dtype,
                attn_implementation=override.get("attn_implementation", attn_implementation),
                quantization_config=quantization_config,
                force_hf=force_hf,
                model_wrapper=model_wrapper,
                autopipeline=autopipeline,
                parallelize_fn=parallelize_fn,
                qat_quantizer=qat_quantizer,
                mesh=mesh,
                loss_fn=loss_fn,
                peft_config=peft_config,
                fp8_config=fp8_config,
                compile_config=compile_config,
                load_base_model=load_base_model,
                **retry_kwargs,
            )

        if is_hf_model:
            attn_implementation, use_liger_kernel = _apply_preload_overrides(
                mesh.tp_size,
                mesh.cp_size,
                has_packed_sequence,
                attn_implementation,
                use_liger_kernel,
            )
        device = torch.cuda.current_device()

        # Use meta device initialization when:
        # - Not using MegatronFSDPManager or DDPManager (they handle their own initialization)
        # - AND either multi-GPU (world_size > 1) or single-GPU custom model (not HF)
        # - AND not using quantization (we let HF handle BitsAndBytes; don't init meta device)
        is_meta_device = all(
            [
                not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager)),
                get_world_size_safe() > 1 or not is_hf_model,
                quantization_config is None,
            ]
        )
        init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()

        model = None  # Ensure 'model' is always bound for the except handler
        is_custom_model = None
        try:
            with init_ctx:
                is_custom_model, model = _init_model(
                    cls,
                    pretrained_model_name_or_path_or_config,
                    attn_implementation,
                    torch_dtype,
                    quantization_config,
                    force_hf,
                    *model_args,
                    **kwargs,
                )
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e) and is_meta_device:
                logger.warning(
                    "Model init hit 'Cannot copy out of meta tensor' (e.g. HF buffer .to(device)); "
                    "retrying without meta device.",
                    flush=True,
                )
                del model
                model = None
                gc.collect()
                is_meta_device = False
                is_custom_model, model = _init_model(
                    cls,
                    pretrained_model_name_or_path_or_config,
                    attn_implementation,
                    torch_dtype,
                    quantization_config,
                    force_hf,
                    *model_args,
                    **kwargs,
                )
            else:
                raise
        except ValueError as e:
            if "does not support" in str(e):
                del model
                attn_implementation = _get_next_fallback_attn(attn_implementation)
                logger.warning("Falling back to %s attention.", attn_implementation)
                return _retry(attn_implementation=attn_implementation)
            raise

        # Kernel patching
        try:
            if use_liger_kernel and not is_custom_model:
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logger.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        try:
            if use_sdpa_patching and not is_custom_model:
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except Exception:
            logger.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

        # Resolve pretrained path for checkpoint loading
        is_pretrained = isinstance(pretrained_model_name_or_path_or_config, str)
        pretrained_path = (
            pretrained_model_name_or_path_or_config
            if is_pretrained
            else getattr(pretrained_model_name_or_path_or_config, "name_or_path", "")
        )

        if is_hf_model:
            _verify_sdpa_support(model, mesh.cp_size)

        model = apply_model_infrastructure(
            model=model,
            pretrained_model_name_or_path=pretrained_path,
            mesh=mesh,
            peft_config=peft_config,
            quantization_config=quantization_config,
            fp8_config=fp8_config,
            qat_quantizer=qat_quantizer,
            loss_fn=loss_fn,
            autopipeline=autopipeline,
            parallelize_fn=parallelize_fn,
            model_wrapper=model_wrapper,
            is_meta_device=is_meta_device,
            device=device,
            compile_config=compile_config,
            load_base_model=load_base_model,
            cache_dir=cache_dir,
            freeze_config=freeze_config,
        )

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
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        quantization_config=None,
        force_hf: bool = False,
        device_mesh: Optional["DeviceMesh"] = None,
        moe_mesh: Optional["DeviceMesh"] = None,
        tp_plan: Optional[dict] = None,
        distributed_config: Optional[DistributedConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        qat_config: Optional[QATConfig] = None,
        moe_config: Optional[MoEParallelizerConfig] = None,
        activation_checkpointing: bool = False,
        peft_config: Optional[dict] = None,
        fp8_config: Optional["FP8Config"] = None,
        compile_config: Optional["CompileConfig"] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate and (optionally) patch a causal-language model.

        This is a light wrapper around
        `transformers.AutoModelForCausalLM.from_pretrained` that can
        automatically apply Liger and/or SDPA (scaled-dot-product
        attention) kernel optimizations, as well as PEFT, quantization,
        and distributed parallelism.

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
            attn_implementation (str, optional):
                Specifies which attention implementation to use (e.g.,
                ``"flash_attention_2"``, ``"eager"``). Only applied when the
                base model supports this kwarg. Defaults to ``"flash_attention_2"``,
                if flash attention is not available, defaults to ``"sdpa"``.
            quantization_config (optional): BitsAndBytesConfig configuration object that
                specifies all quantization settings. If provided, quantization
                will be applied to the model.
            force_hf (bool, default=False): If `True`, force the use of HF model implementation.
                If `False`, the model will be loaded using the custom model implementation if available.
            device_mesh (DeviceMesh | None, optional): Pre-created device mesh for
                distributed training. Parallelism sizes (tp, pp, cp, ep) are inferred
                from this. Default: None.
            moe_mesh (DeviceMesh | None, optional): FSDP2-only. Device mesh for expert
                parallelism. ep_size is inferred from this. Default: None.
            tp_plan (dict | None, optional): Custom tensor parallel plan. If provided,
                overrides the tp_plan on distributed_config. Default: None.
            distributed_config (FSDP2Config | MegatronFSDPConfig | DDPConfig | None, optional):
                Strategy-specific distributed training configuration. Default: None.
            pipeline_config (PipelineConfig | None, optional): Pipeline parallelism
                configuration including loss_fn. Default: None.
            qat_config (QATConfig | None, optional): Quantization-Aware Training
                configuration. Default: None.
            moe_config (MoEParallelizerConfig | None, optional): MoE parallelizer
                configuration. Default: None.
            activation_checkpointing (bool, default=False): Enable activation checkpointing
                for transformer blocks to reduce memory usage. Default: False.
            peft_config (dict | None, optional): PEFT/LoRA configuration dictionary.
                If provided, LoRA adapters will be applied to the model. Default: None.
            fp8_config (FP8Config | None, optional): FP8 quantization configuration.
                If provided, FP8 quantization will be applied. Default: None.
            compile_config (CompileConfig | None, optional): Configuration for torch.compile.
                If provided, the model will be compiled. Default: None.
            **kwargs: Additional keyword arguments. Notable ones include:
                - has_packed_sequence (bool): Whether using packed sequences. Default: False.
                - cache_dir (str): Cache directory for model weights.

        Returns:
            transformers.PreTrainedModel: The loaded (and possibly patched)
            model instance with all infrastructure applied.
        """
        if tp_plan is not None and distributed_config is not None:
            distributed_config.tp_plan = tp_plan

        mesh = MeshContext.from_meshes(device_mesh, moe_mesh)

        model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
            distributed_config=distributed_config,
            pipeline_config=pipeline_config,
            qat_config=qat_config,
            moe_config=moe_config,
            activation_checkpointing=activation_checkpointing,
            device=torch.device("cuda", torch.cuda.current_device()),
            mesh=mesh,
        )
        loss_fn = pipeline_config.loss_fn if pipeline_config is not None else None

        try:
            hf_config = get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs)
        except Exception as e:
            if "does not support" in str(e):
                attn_implementation = _get_next_fallback_attn(attn_implementation)
                logger.warning("Config rejected attention implementation, falling back to %s.", attn_implementation)
                hf_config = get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs)
            else:
                raise
        is_hf_model = get_is_hf_model(hf_config, force_hf)

        return cls._build_model(
            pretrained_model_name_or_path,
            *model_args,
            is_hf_model=is_hf_model,
            use_liger_kernel=use_liger_kernel,
            use_sdpa_patching=use_sdpa_patching,
            sdpa_method=sdpa_method,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            force_hf=force_hf,
            model_wrapper=model_wrapper,
            autopipeline=autopipeline,
            parallelize_fn=parallelize_fn,
            qat_quantizer=qat_quantizer,
            mesh=mesh,
            loss_fn=loss_fn,
            peft_config=peft_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            load_base_model=True,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config,
        *model_args,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: Optional[List[SDPBackend]] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        quantization_config=None,
        force_hf: bool = False,
        device_mesh: Optional["DeviceMesh"] = None,
        moe_mesh: Optional["DeviceMesh"] = None,
        tp_plan: Optional[dict] = None,
        distributed_config: Optional[DistributedConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        qat_config: Optional[QATConfig] = None,
        moe_config: Optional[MoEParallelizerConfig] = None,
        activation_checkpointing: bool = False,
        peft_config: Optional[dict] = None,
        fp8_config: Optional["FP8Config"] = None,
        compile_config: Optional["CompileConfig"] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate a model from a ``transformers.PretrainedConfig`` (no pretrained
        weights). Accepts the same infrastructure arguments as ``from_pretrained``.

        See ``from_pretrained`` for full parameter documentation.

        Args:
            config (transformers.PretrainedConfig | str):
                The configuration object used to build the model.
                If config is passed as a string (e.g., model-id / local checkpoint),
                it will create a config internally using AutoConfig.
            torch_dtype (str | torch.dtype, default="auto"):
                Data type for model parameters. If "auto", defaults to ``torch.bfloat16``.
        """
        if tp_plan is not None and distributed_config is not None:
            distributed_config.tp_plan = tp_plan

        mesh = MeshContext.from_meshes(device_mesh, moe_mesh)

        # Only instantiate infrastructure when distributed_config is provided
        model_wrapper = autopipeline = parallelize_fn = qat_quantizer = None
        loss_fn = None
        if distributed_config is not None:
            model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
                distributed_config=distributed_config,
                pipeline_config=pipeline_config,
                qat_config=qat_config,
                moe_config=moe_config,
                activation_checkpointing=activation_checkpointing,
                device=torch.device("cuda", torch.cuda.current_device()),
                mesh=mesh,
            )
            if pipeline_config is not None:
                loss_fn = pipeline_config.loss_fn

        torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch.bfloat16
        name_or_path = config if isinstance(config, str) else getattr(config, "name_or_path", None)
        kwargs["trust_remote_code"] = kwargs.get(
            "trust_remote_code", resolve_trust_remote_code(name_or_path) if name_or_path else False
        )
        if isinstance(config, str):
            try:
                config = get_hf_config(config, attn_implementation, **kwargs)
            except Exception as e:
                if "does not support" in str(e):
                    attn_implementation = _get_next_fallback_attn(attn_implementation)
                    logger.warning("Config rejected attention implementation, falling back to %s.", attn_implementation)
                    config = get_hf_config(config, attn_implementation, **kwargs)
                else:
                    raise
        _consume_config_overrides(config, kwargs)
        is_hf_model = get_is_hf_model(config, force_hf)

        return cls._build_model(
            config,
            *model_args,
            is_hf_model=is_hf_model,
            use_liger_kernel=use_liger_kernel,
            use_sdpa_patching=use_sdpa_patching,
            sdpa_method=sdpa_method,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            force_hf=force_hf,
            model_wrapper=model_wrapper,
            autopipeline=autopipeline,
            parallelize_fn=parallelize_fn,
            qat_quantizer=qat_quantizer,
            mesh=mesh,
            loss_fn=loss_fn,
            peft_config=peft_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            load_base_model=False,
            **kwargs,
        )


#  Concrete Auto-Model classes
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
