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
import inspect
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
    AutoModelForMultimodalLM,
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
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline  # noqa: E402, F401
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

from nemo_automodel._transformers.infrastructure import (
    MeshContext,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from nemo_automodel._transformers.kernel_patches import (
    DEFAULT_ATTN_IMPLEMENTATION,
    _apply_preload_overrides,
    _get_next_fallback_attn,
    _patch_attention,
    _patch_liger_kernel,
    _verify_sdpa_support,
)
from nemo_automodel._transformers.model_init import (
    _consume_config_overrides,
    _init_model,
    get_hf_config,
    get_is_hf_model,
    no_hf_meta_device,
    resolve_sdpa_method,
)

if not hasattr(_gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
    from transformers.cache_utils import StaticCache

    _gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache}


logger = logging.getLogger(__name__)

_MAX_BUILD_RETRIES = 5

_remote_code_compat_applied = False


def _patch_remote_code_compat():
    """Patch ``_finalize_model_loading`` for remote-code models written against older transformers.

    Remote-code models (``trust_remote_code=True``) may be incompatible with
    the installed transformers in several ways:

    1. Missing ``all_tied_weights_keys`` -- set in ``post_init()`` which the
       model may never call.
    2. Overridden ``tie_weights()`` with an old signature that doesn't accept
       the ``missing_keys`` kwarg added in newer transformers.

    This one-time patch wraps ``_finalize_model_loading`` to fix these issues
    on the fly.  For models that are already compatible the guards are no-ops.
    """
    global _remote_code_compat_applied
    if _remote_code_compat_applied:
        return
    _orig_finalize = PreTrainedModel._finalize_model_loading

    def _compat_finalize(model, load_config, loading_info):
        # 1. Ensure all_tied_weights_keys exists
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)

        # 2. Wrap tie_weights if it doesn't accept `missing_keys`
        model_cls = type(model)
        if model_cls.tie_weights is not PreTrainedModel.tie_weights:
            sig = inspect.signature(model_cls.tie_weights)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "missing_keys" not in sig.parameters and not has_var_kw:
                _orig_tie = model_cls.tie_weights

                def _compat_tie(self, **kwargs):
                    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    return _orig_tie(self, **accepted)

                model_cls.tie_weights = _compat_tie

        # 3. Set missing config defaults that older remote code models expect
        _config_defaults = {"use_cache": False}
        for attr, default in _config_defaults.items():
            if not hasattr(model.config, attr):
                setattr(model.config, attr, default)

        return _orig_finalize(model, load_config, loading_info)

    PreTrainedModel._finalize_model_loading = staticmethod(_compat_finalize)
    _remote_code_compat_applied = True


def _maybe_dequantize_fp8_for_peft(hf_native_quant_cfg, peft_config, pretrained_path):
    """Set ``dequantize=True`` on FP8 quantization configs when PEFT is requested.

    Returns True if the config was mutated, False otherwise.
    """
    if peft_config is not None and isinstance(pretrained_path, str):
        if isinstance(hf_native_quant_cfg, dict) and hf_native_quant_cfg.get("quant_method") == "fp8":
            hf_native_quant_cfg["dequantize"] = True
            logger.info("FP8 model with PEFT: setting dequantize=True for compatibility")
            return True
    return False


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
        try:
            model = super().from_pretrained(*args, **kwargs)
        except (AttributeError, TypeError) as e:
            if "all_tied_weights_keys" in str(e) or (isinstance(e, TypeError) and "tie_weights" in str(e)):
                logger.warning(
                    "Remote code model incompatible with installed transformers (%s). "
                    "Applying compatibility patches and retrying.",
                    e,
                )
                _patch_remote_code_compat()
                model = super().from_pretrained(*args, **kwargs)
            else:
                raise
        except OSError:
            if kwargs.get("use_safetensors") is not False:
                logger.warning(
                    "Checkpoint resolution failed; retrying with use_safetensors=False "
                    "(the model may only provide .bin checkpoints)."
                )
                kwargs["use_safetensors"] = False
                model = super().from_pretrained(*args, **kwargs)
            else:
                raise
        finally:
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
        _retry_depth=0,
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
            if _retry_depth >= _MAX_BUILD_RETRIES:
                raise
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
                _retry_depth=_retry_depth + 1,
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

        # When PEFT is requested, force dequantization of FP8-quantized models.
        # FP8 linear modules (e.g. transformers FP8Linear) have scalar parameters
        # incompatible with FSDP2, and their custom forward doesn't compose with
        # LoRA patching. Setting dequantize=True tells transformers to convert
        # FP8 weights to bf16 during loading.
        _hf_config = (
            get_hf_config(pretrained_model_name_or_path_or_config, attn_implementation, **kwargs)
            if isinstance(pretrained_model_name_or_path_or_config, str)
            else pretrained_model_name_or_path_or_config
        )
        _hf_native_quant_cfg = getattr(_hf_config, "quantization_config", None)
        if _maybe_dequantize_fp8_for_peft(_hf_native_quant_cfg, peft_config, pretrained_model_name_or_path_or_config):
            kwargs["config"] = _hf_config

        # Use meta device initialization when:
        # - Not using MegatronFSDPManager or DDPManager (they handle their own initialization)
        # - AND either multi-GPU (world_size > 1) or single-GPU custom model (not HF)
        # - AND not using quantization (we let HF handle BitsAndBytes/FP8; don't init meta device)
        #   For non-HF models, native quant config is ignored.
        is_meta_device = all(
            [
                not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager)),
                get_world_size_safe() > 1 or not is_hf_model,
                quantization_config is None and (_hf_native_quant_cfg is None or not is_hf_model),
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
        except (NotImplementedError, RuntimeError) as e:
            _meta_err_msgs = (
                "Cannot copy out of meta tensor",
                "cannot be called on meta tensors",
                "aten::equal: attempted to run this operator with Meta tensors",
            )
            if any(msg in str(e) for msg in _meta_err_msgs) and is_meta_device:
                logger.warning(
                    "Model init hit meta-tensor error (%s); retrying without meta device.",
                    type(e).__name__,
                )
                del model
                model = None
                gc.collect()
                is_meta_device = False
                with ContextManagers([no_init_weights(), no_hf_meta_device()]):
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

        from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate

        attach_capabilities_and_validate(model, mesh)

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
        sdpa_method: Optional[List[Union[SDPBackend, str]]] = None,
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
            sdpa_method (list[SDPBackend | str] | None, optional): Explicit list of
                SDPA back-ends to consider when `use_sdpa_patching=True`.
                Accepts both SDPBackend enum values and string names (e.g.
                ``["flash_attention", "efficient_attention"]``). When ``None``,
                auto-selects based on CP and activation checkpointing.
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

        sdpa_method = resolve_sdpa_method(sdpa_method, device_mesh, activation_checkpointing)

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
        sdpa_method: Optional[List[Union[SDPBackend, str]]] = None,
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

        sdpa_method = resolve_sdpa_method(sdpa_method, device_mesh, activation_checkpointing)

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
            load_base_model=kwargs.pop("load_base_model", False),
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


class NeMoAutoModelForMultimodalLM(_BaseNeMoAutoModelClass, AutoModelForMultimodalLM):
    """Drop-in replacement for ``transformers.AutoModelForMultimodalLM`` with custom-kernels."""

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


class _NeMoAutoModelForRetrievalBase:
    """Private shared base for encoder auto-models.

    Subclasses set ``_ENCODER_CLS_NAME`` to select the concrete encoder class
    from ``nemo_automodel._transformers.retrieval``.
    """

    _ENCODER_CLS_NAME: Optional[str] = None  # "BiEncoderModel" or "CrossEncoderModel"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        use_liger_kernel: bool = True,
        use_sdpa_patching: bool = True,
        sdpa_method: Optional[List[SDPBackend]] = None,
        torch_dtype="auto",
        device_mesh: Optional["DeviceMesh"] = None,
        moe_mesh: Optional["DeviceMesh"] = None,
        tp_plan: Optional[dict] = None,
        distributed_config: Optional[DistributedConfig] = None,
        moe_config: Optional[MoEParallelizerConfig] = None,
        compile_config: Optional["CompileConfig"] = None,
        peft_config: Optional[dict] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Load an encoder model with infrastructure (FSDP, PEFT, kernel patching, etc.).

        This method builds an encoder via the subclass's ``_ENCODER_CLS_NAME``,
        applies kernel patching, and then applies all infrastructure (FSDP,
        checkpointing, etc.) through ``apply_model_infrastructure()``.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier.
            attn_implementation: Attention implementation to use (e.g.,
                ``"flash_attention_2"``, ``"sdpa"``, ``"eager"``).
                Defaults to ``DEFAULT_ATTN_IMPLEMENTATION``
                (``"flash_attention_2"`` when flash-attn is installed, otherwise ``"sdpa"``).
            use_liger_kernel: Whether to apply Liger kernel optimizations.
            use_sdpa_patching: Whether to apply SDPA patching.
            sdpa_method: SDPA backend methods to use.
            torch_dtype: Data type passed to the underlying model initialization.
            device_mesh: Pre-created device mesh for distributed training.
            moe_mesh: Device mesh for expert parallelism (FSDP2 only).
            tp_plan: Custom tensor parallel plan; overrides distributed_config.tp_plan.
            distributed_config: Strategy-specific distributed training configuration.
            moe_config: MoE parallelizer configuration.
            compile_config: Configuration for torch.compile.
            peft_config: PEFT/LoRA configuration dictionary.
            **kwargs: Additional arguments passed to the encoder's ``build()`` method.

        Returns:
            Encoder model instance with loaded weights and all infrastructure applied.

        Notes:
            If kernel patching fails, the method retries with adjusted parameters.
        """
        from nemo_automodel._transformers import retrieval as _enc_mod

        encoder_cls = getattr(_enc_mod, cls._ENCODER_CLS_NAME)

        logger.info(f"Loading {cls.__name__} from {pretrained_model_name_or_path}")

        def _retry(**override):
            return cls.from_pretrained(
                pretrained_model_name_or_path,
                attn_implementation=attn_implementation,
                use_liger_kernel=override.get("use_liger_kernel", use_liger_kernel),
                use_sdpa_patching=override.get("use_sdpa_patching", use_sdpa_patching),
                sdpa_method=sdpa_method,
                torch_dtype=torch_dtype,
                device_mesh=device_mesh,
                moe_mesh=moe_mesh,
                tp_plan=tp_plan,
                distributed_config=distributed_config,
                moe_config=moe_config,
                compile_config=compile_config,
                peft_config=peft_config,
                **kwargs,
            )

        build_kwargs = dict(kwargs)
        build_kwargs.pop("tp_size", None)
        build_kwargs.pop("cp_size", None)
        build_kwargs.pop("has_packed_sequence", None)

        if tp_plan is not None and distributed_config is not None:
            distributed_config.tp_plan = tp_plan

        mesh = MeshContext.from_meshes(device_mesh, moe_mesh)

        model_wrapper, autopipeline, parallelize_fn, qat_quantizer = instantiate_infrastructure(
            distributed_config=distributed_config,
            pipeline_config=None,
            qat_config=None,
            moe_config=moe_config,
            device=torch.device("cuda", torch.cuda.current_device()),
            mesh=mesh,
        )

        device = torch.cuda.current_device()

        model = encoder_cls.build(
            model_name_or_path=pretrained_model_name_or_path,
            attn_implementation=attn_implementation,
            **build_kwargs,
        )

        try:
            if use_liger_kernel:
                logger.info("Applying Liger kernel patching to encoder")
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logger.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        try:
            if use_sdpa_patching:
                logger.info("Applying SDPA patching to encoder")
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except Exception:
            logger.warning("Retrying without SDPA patching.")
            del model
            gc.collect()
            return _retry(use_sdpa_patching=False)

        model = apply_model_infrastructure(
            model=model,  # noqa: F821
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            is_meta_device=False,
            device=device,
            model_wrapper=model_wrapper,
            mesh=mesh,
            peft_config=peft_config,
            quantization_config=None,
            fp8_config=None,
            qat_quantizer=qat_quantizer,
            loss_fn=None,
            autopipeline=autopipeline,
            parallelize_fn=parallelize_fn,
            compile_config=compile_config,
            load_base_model=False,  # encoder_cls.build already loads weights
            cache_dir=build_kwargs.get("cache_dir", hf_constants.HF_HUB_CACHE),
        )

        return model


class NeMoAutoModelBiEncoder(_NeMoAutoModelForRetrievalBase):
    """NeMo AutoModel for bi-encoder embedding tasks with full infrastructure support.

    Wraps ``BiEncoderModel.build()`` with kernel patching, PEFT, FSDP, and
    other distributed infrastructure via ``apply_model_infrastructure()``.

    Examples:
    --------
    >>> model = NeMoAutoModelBiEncoder.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> model = NeMoAutoModelBiEncoder.from_pretrained(
    ...     "meta-llama/Llama-3.2-1B",
    ...     pooling="cls",
    ...     l2_normalize=False,
    ...     distributed_config=FSDP2Config(),
    ... )
    """

    _ENCODER_CLS_NAME = "BiEncoderModel"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        pooling: str = "avg",
        l2_normalize: bool = True,
        **kwargs,
    ) -> PreTrainedModel:
        """Load a bi-encoder model with infrastructure.

        Accepts all arguments from ``_NeMoAutoModelForRetrievalBase.from_pretrained``
        plus the bi-encoder-specific parameters below.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier.
            pooling: Pooling strategy (``'avg'``, ``'cls'``, ``'last'``, etc.).
            l2_normalize: Whether to L2-normalize embeddings.
            **kwargs: Forwarded to ``_NeMoAutoModelForRetrievalBase.from_pretrained``.

        Returns:
            BiEncoderModel instance with loaded weights and all infrastructure applied.
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            pooling=pooling,
            l2_normalize=l2_normalize,
            **kwargs,
        )


class NeMoAutoModelCrossEncoder(_NeMoAutoModelForRetrievalBase):
    """NeMo AutoModel for cross-encoder scoring tasks with full infrastructure support.

    Wraps ``CrossEncoderModel.build()`` with kernel patching, PEFT, FSDP, and
    other distributed infrastructure via ``apply_model_infrastructure()``.

    Examples:
    --------
    >>> model = NeMoAutoModelCrossEncoder.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> model = NeMoAutoModelCrossEncoder.from_pretrained(
    ...     "meta-llama/Llama-3.2-1B",
    ...     distributed_config=FSDP2Config(),
    ... )
    """

    _ENCODER_CLS_NAME = "CrossEncoderModel"
