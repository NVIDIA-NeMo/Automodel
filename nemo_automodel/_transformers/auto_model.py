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
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.components.utils.compile_utils import compile_model
from nemo_automodel.shared.torch_patches import apply_torch_patches

apply_torch_patches()
from huggingface_hub import constants as hf_constants
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForTextToWaveform,
    PreTrainedModel,
)
from transformers.initialization import no_init_weights
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.utils import ContextManagers

import nemo_automodel.components.distributed.utils as dist_utils
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    _maybe_adapt_state_dict_to_hf,
)
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.init_utils import get_local_world_size_preinit, get_world_size_safe
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.quantization.fp8 import apply_fp8_to_model
from nemo_automodel.components.utils.model_utils import (
    _supports_logits_to_keep,
    init_empty_weights,
    print_trainable_parameters,
    resolve_trust_remote_code,
)
from nemo_automodel.shared.import_utils import safe_import
from nemo_automodel.shared.utils import dtype_from_str

if TYPE_CHECKING:
    from torchao.quantization.qat.linear import Int4WeightOnlyQATQuantizer, Int8DynActInt4WeightQATQuantizer

    from nemo_automodel.components.quantization.fp8 import FP8Config
    from nemo_automodel.components.utils.compile_utils import CompileConfig

HAS_LIGER_KERNEL, liger_kernel_trf = safe_import("liger_kernel.transformers")
HAS_FA, _ = safe_import("flash_attn")
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FA else "sdpa"

logger = logging.getLogger(__name__)

# Backward-compat shim for trust_remote_code models (e.g. DeciLM)
# that import NEED_SETUP_CACHE_CLASSES_MAPPING from transformers.generation.utils.
import transformers.generation.utils as _gen_utils  # noqa: E402

if not hasattr(_gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
    from transformers.cache_utils import StaticCache

    _gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache}


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

    # 1. if force_hf is True, use HF model class wrapped with mixin
    if force_hf:
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        if is_pretrained_init:
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


def _apply_peft_and_lower_precision(
    model, tp_size, autopipeline, peft_config, quantization_config, fp8_config, qat_quantizer
):
    if peft_config is not None:
        if tp_size > 1:
            logger.info("Disabling Triton with TP ({})".format(tp_size))
            peft_config.use_triton = False
        if autopipeline is not None:
            logger.info("Enabling PEFT with Pipeline Parallelism")
            logger.info("Disabling Triton with Pipeline Parallelism Enabled.")
            peft_config.use_triton = False
        apply_lora_to_linear_modules(model, peft_config, quantization_config=quantization_config)

    # FP8
    if fp8_config is not None:
        model = apply_fp8_to_model(model, config=fp8_config)

    # QAT
    if qat_quantizer is not None:
        from nemo_automodel.components.quantization.qat import prepare_qat_model

        if any(map(lambda x: x.dtype != torch.bfloat16, model.parameters())):
            logger.warning("QAT is only supported for bfloat16 models. Support will be added in future release.")
            quit(code=0)
        model, qat_mode = prepare_qat_model(model, qat_quantizer)
        # Attach helpers for delayed fake-quant toggling if desired
        model._qat_mode = qat_mode  # type: ignore[attr-defined]

    return model


def _shard_pp(autopipeline, model, loss_fn, parallelize_fn):
    trainable_params, total_params = print_trainable_parameters(model)
    # Store param info on autopipeline before splitting so it can be accessed later
    # This captures the full model's param counts before PP shards it across ranks
    autopipeline.trainable_params_before_pp = trainable_params
    autopipeline.total_params_before_pp = total_params
    if get_world_size_safe() == 1:
        logger.info("World size is 1, skipping autopipeline.")
    else:
        autopipeline.build(model, loss_fn=loss_fn, parallelize_fn=parallelize_fn)
        model = autopipeline
    return model


def _shard_ep_fsdp(model, model_wrapper, parallelize_fn):
    if parallelize_fn is not None and get_world_size_safe() > 1:
        parallelize_fn(
            model,
            world_mesh=model_wrapper.device_mesh,
            moe_mesh=model_wrapper.moe_mesh,
            dp_axis_names=(
                ("dp_replicate", "dp_shard_cp")
                if "dp_replicate" in model_wrapper.device_mesh.mesh_dim_names
                and "dp_shard_cp" in model_wrapper.device_mesh.mesh_dim_names
                else ("dp_shard_cp",)
            ),
            cp_axis_name="cp" if "cp" in model_wrapper.device_mesh.mesh_dim_names else None,
            tp_axis_name="tp" if "tp" in model_wrapper.device_mesh.mesh_dim_names else None,
            ep_axis_name="ep"
            if model_wrapper.moe_mesh is not None and "ep" in model_wrapper.moe_mesh.mesh_dim_names
            else None,
            ep_shard_axis_names=(
                ("ep_shard",)
                if model_wrapper.moe_mesh is not None and "ep_shard" in model_wrapper.moe_mesh.mesh_dim_names
                else None
            ),
        )
    elif callable(getattr(model_wrapper, "parallelize", None)):
        model = model_wrapper.parallelize(model)
        model = (
            model[0] if isinstance(model, tuple) else model
        )  # MegatronFSDP will return (model, None) since we don't pass optimizer here
    return model


def apply_model_infrastructure(
    model,
    *,
    is_hf_model,
    is_meta_device,
    device,
    model_wrapper=None,
    tp_size=1,
    cp_size=1,
    peft_config=None,
    quantization_config=None,
    fp8_config=None,
    qat_quantizer=None,
    loss_fn=None,
    autopipeline=None,
    parallelize_fn=None,
    compile_config=None,
    load_base_model=False,
    cache_dir=None,
    pretrained_model_name_or_path="",
    **_kwargs,
):
    """Apply sharding, PEFT, quantization, and checkpoint loading to a model.

    This function contains the common post-init logic shared between from_pretrained
    and from_config methods. It can also be called directly for models built via
    custom builder functions (e.g., build_gpt2_model). It handles:
    - SDPA support verification
    - PEFT and lower precision application (LoRA, FP8, QAT)
    - Loss function setup
    - Pipeline parallelism or EP/FSDP sharding
    - Device placement and compilation
    - Checkpoint loading for meta device models

    Args:
        model: The model to apply infrastructure to
        is_hf_model: Whether this is an HF model (vs custom implementation)
        is_meta_device: Whether model was initialized on meta device
        device: Target device for model
        model_wrapper: Model wrapper (FSDP2Manager, DDPManager, etc.). Default: None
        tp_size: Tensor parallelism size. Default: 1
        cp_size: Context parallelism size. Default: 1
        peft_config: PEFT/LoRA configuration dict. Default: None
        quantization_config: Quantization configuration. Default: None
        fp8_config: FP8 configuration. Default: None
        qat_quantizer: QAT quantizer instance. Default: None
        loss_fn: Loss function (may be replaced with MaskedCrossEntropy). Default: None
        autopipeline: AutoPipeline instance for pipeline parallelism. Default: None
        parallelize_fn: Function to apply parallelization (EP + FSDP2). Default: None
        compile_config: Compilation configuration. Default: None
        pretrained_model_name_or_path: Model name or path for checkpoint loading. Default: None
        load_base_model: Whether to load base model weights (True for from_pretrained). Default: False
        cache_dir: Cache directory for model weights. Default: None
        **_kwargs: Additional keyword arguments (ignored, allows passing extra kwargs)

    Returns:
        The model with all infrastructure applied
    """
    _verify_sdpa_support(model, is_hf_model, cp_size)

    # Create a dummy checkpointer. We can pass in dummy values here since we are only loading the base weights.
    ckpt_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir="",
        model_save_format="safetensors",
        model_cache_dir=cache_dir,
        model_repo_id=pretrained_model_name_or_path,
        save_consolidated=True,
        is_peft=peft_config is not None,
    )
    checkpointer = Checkpointer(
        ckpt_config,
        0,
        0,
        0,
        getattr(model_wrapper, "moe_mesh", None) if model_wrapper else None,
    )

    # Handle checkpointer config updates if checkpointer is provided
    dequantize_base_checkpoint = False
    if checkpointer is not None:
        if checkpointer.config.dequantize_base_checkpoint is None:
            # try to infer whether the base weights are quantized
            checkpointer.config.dequantize_base_checkpoint = hasattr(
                getattr(model, "config", None), "quantization_config"
            )
        dequantize_base_checkpoint = checkpointer.config.dequantize_base_checkpoint

    # Apply PEFT and lower precision if configured
    # When on meta device, wrap in init_empty_weights() so new LoRA modules are also on meta device
    # This allows copy operations between meta tensors to succeed (they're no-ops)
    peft_ctx = init_empty_weights() if is_meta_device else nullcontext()
    with peft_ctx:
        model = _apply_peft_and_lower_precision(
            model, tp_size, autopipeline, peft_config, quantization_config, fp8_config, qat_quantizer
        )

    # hold a list copy of the model state dict keys before any parallelization. To be used during checkpoint saving in safetensors format.
    pre_shard_hf_state_dict_keys = list(
        _maybe_adapt_state_dict_to_hf(model, model.state_dict(), quantization=dequantize_base_checkpoint).keys()
    )

    # Loss function check
    if not _supports_logits_to_keep(model) and not isinstance(loss_fn, MaskedCrossEntropy):
        loss_fn = MaskedCrossEntropy()

    # Apply pipeline parallelism if configured. This is the outermost parallelization.
    # Note: AutoPipeline takes care of applying PP + EP + FSDP. _shard_ep_fsdp will take care of applying EP + FSDP if no PP.
    if autopipeline is not None:
        model = _shard_pp(autopipeline, model, loss_fn, parallelize_fn)
        for part in model.parts:
            setattr(part, "_pre_shard_hf_state_dict_keys", pre_shard_hf_state_dict_keys)
    else:
        model = _shard_ep_fsdp(model, model_wrapper, parallelize_fn)
        if compile_config is not None:
            model = compile_model(model, compile_config)
        if isinstance(model_wrapper, DDPManager):
            setattr(model.module, "_pre_shard_hf_state_dict_keys", pre_shard_hf_state_dict_keys)
        else:
            setattr(model, "_pre_shard_hf_state_dict_keys", pre_shard_hf_state_dict_keys)

    # Load the checkpoint if needed and return
    # Weights need to be loaded for meta device models:
    # 1. Single GPU custom models (no parallelization but still need weights)
    # 2. When parallelize_fn was used (which will internally apply FSDP2/EP sharding)
    # 3. When FSDP2Manager.parallelize was used (but not MegatronFSDP which handles weights internally)
    should_load_checkpoint = is_meta_device and any(
        [
            get_world_size_safe() == 1,
            parallelize_fn is not None and get_world_size_safe() > 1,
            callable(getattr(model_wrapper, "parallelize", None)),
        ]
    )
    if should_load_checkpoint:
        models_to_load = model.parts if hasattr(model, "parts") else [model]
        lora_a_init = getattr(peft_config, "lora_A_init", None)
        for mp in models_to_load:
            checkpointer.load_base_model(
                mp,
                device,
                cache_dir,
                pretrained_model_name_or_path,
                lora_a_init,
                load_base_model=load_base_model,
            )

    if autopipeline is None:
        print_trainable_parameters(model)  # Once model's been sharded
        # Ensure model is on the correct device; AutoPipeline takes care of it internally
        model.to(device)

    return model


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
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        quantization_config=None,
        force_hf: bool = False,
        model_wrapper=None,
        autopipeline: AutoPipeline | None = None,
        parallelize_fn: Callable | None = None,
        peft_config: Optional[dict] = None,
        fp8_config: Optional["FP8Config"] = None,
        qat_quantizer: Optional[Union["Int4WeightOnlyQATQuantizer", "Int8DynActInt4WeightQATQuantizer"]] = None,
        loss_fn: Optional[Callable] = None,
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
            model_wrapper (optional): Parallelism wrapper instance (e.g., FSDP2Manager,
                MegatronFSDPManager, DDPManager). Used for distributed training setup
                and determines meta device initialization behavior.
            autopipeline (AutoPipeline | None, optional): AutoPipeline instance for
                pipeline parallelism. When provided, the model will be split across
                pipeline stages. Default: None.
            parallelize_fn (Callable | None, optional): Custom function to apply
                parallelization (EP + FSDP2). Default: None.
            peft_config (dict | None, optional): PEFT/LoRA configuration dictionary.
                If provided, LoRA adapters will be applied to the model. Default: None.
            fp8_config (FP8Config | None, optional): FP8 quantization configuration.
                If provided, FP8 quantization will be applied to the model. Default: None.
            qat_quantizer (Int4WeightOnlyQATQuantizer | Int8DynActInt4WeightQATQuantizer | None, optional):
                Quantization-Aware Training quantizer instance. If provided, QAT will be
                applied to the model. Default: None.
            loss_fn (Callable | None, optional): Loss function to use. If the model
                doesn't support `logits_to_keep` and loss_fn is not MaskedCrossEntropy,
                it will be replaced with MaskedCrossEntropy. This is passed to AutoPipeline. Default: None.
            compile_config (CompileConfig | None, optional): Configuration for torch.compile.
                If provided, the model will be compiled for improved performance. Default: None.
            **kwargs: Additional keyword arguments. Notable ones include:
                - tp_size (int): Tensor parallelism size. Default: 1.
                - cp_size (int): Context parallelism size. Default: 1.
                - has_packed_sequence (bool): Whether using packed sequences. Default: False.
                - cache_dir (str): Cache directory for model weights.

        Returns:
            transformers.PreTrainedModel: The loaded (and possibly patched)
            model instance with all infrastructure applied.

        Warns:
            UserWarning: Emitted when `use_liger_kernel=True` but the Liger
            package is unavailable.

        Notes:
            If kernel patching fails, the partially constructed model is
            deleted and the method recurses once with
            `use_liger_kernel=False` or `use_sdpa_patching=False`.
        """

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
                force_hf=force_hf,
                autopipeline=autopipeline,
                parallelize_fn=parallelize_fn,
                peft_config=peft_config,
                fp8_config=fp8_config,
                qat_quantizer=qat_quantizer,
                loss_fn=loss_fn,
                compile_config=compile_config,
                model_wrapper=model_wrapper,
                **kwargs,
            )

        is_hf_model = get_is_hf_model(
            get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs),
            force_hf,
        )
        tp_size, cp_size, has_packed_sequence = _pop_tp_cp_has_packed(kwargs)
        attn_implementation, use_liger_kernel = _apply_preload_overrides(
            is_hf_model, tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel
        )
        device = torch.cuda.current_device()

        # Use meta device initialization when:
        # - Not using MegatronFSDPManager or DDPManager (they handle their own initialization)
        # - AND either multi-GPU (world_size > 1) or single-GPU custom model (not HF)
        # HF models on single GPU load weights via from_pretrained, but multi-GPU needs meta device for sharding
        is_meta_device = not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager)) and (
            get_world_size_safe() > 1 or not is_hf_model
        )
        init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()

        try:
            with init_ctx:
                is_custom_model, model = _init_model(
                    cls,
                    pretrained_model_name_or_path,
                    attn_implementation,
                    torch_dtype,
                    quantization_config,
                    force_hf,
                    *model_args,
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
            if use_liger_kernel and not is_custom_model:
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logging.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        # Patch sdpa attention
        try:
            if use_sdpa_patching and not is_custom_model:
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except:
            logging.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

        model = apply_model_infrastructure(
            model=model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            is_hf_model=is_hf_model,
            cp_size=cp_size,
            tp_size=tp_size,
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
            load_base_model=True,
            cache_dir=kwargs.get("cache_dir", hf_constants.HF_HUB_CACHE),
        )

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
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
        quantization_config=None,
        force_hf: bool = False,
        model_wrapper=None,
        autopipeline: AutoPipeline | None = None,
        parallelize_fn: Callable | None = None,
        peft_config: Optional[dict] = None,
        fp8_config: Optional["FP8Config"] = None,
        qat_quantizer: Optional[Union["Int4WeightOnlyQATQuantizer", "Int8DynActInt4WeightQATQuantizer"]] = None,
        loss_fn: Optional[Callable] = None,
        compile_config: Optional["CompileConfig"] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Instantiate a model from a ``transformers.PretrainedConfig`` and optionally
        patch it with Liger or SDPA-optimized kernels, as well as apply PEFT,
        quantization, and distributed parallelism.

        This method creates a model with randomly initialized weights (no pretrained
        weights are loaded). Use ``from_pretrained`` to load pretrained weights.

        Args:
            config (transformers.PretrainedConfig | str):
                The configuration object used to build the model.
                If config is passed as a string (e.g., model-id / local checkpoint),
                it will create a config internally using AutoConfig.
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
            torch_dtype (str | torch.dtype, default="auto"):
                Data type for model parameters. If "auto", defaults to torch.bfloat16.
            attn_implementation (str, optional):
                Specifies which attention implementation to use (e.g.,
                ``"flash_attention_2"``, ``"eager"``). Only applied when the
                base model supports this kwarg. Defaults to ``"flash_attention_2"``,
                if flash attention is not available, defaults to ``"sdpa"``.
            quantization_config (optional): BitsAndBytesConfig configuration object that
                specifies all quantization settings. If provided, quantization
                will be applied to the model.
            force_hf (bool, default=False): If ``True``, force the use of HF model implementation.
                If ``False``, the model will be loaded using the custom model implementation if available.
            model_wrapper (optional): Parallelism wrapper instance (e.g., FSDP2Manager,
                MegatronFSDPManager, DDPManager). Used for distributed training setup
                and determines meta device initialization behavior.
            autopipeline (AutoPipeline | None, optional): AutoPipeline instance for
                pipeline parallelism. When provided, the model will be split across
                pipeline stages. Default: None.
            parallelize_fn (Callable | None, optional): Custom function to apply
                parallelization (EP + FSDP2).
                Called after model initialization. Default: None.
            peft_config (dict | None, optional): PEFT/LoRA configuration dictionary.
                If provided, LoRA adapters will be applied to the model. Default: None.
            fp8_config (FP8Config | None, optional): FP8 quantization configuration.
                If provided, FP8 quantization will be applied to the model. Default: None.
            qat_quantizer (Int4WeightOnlyQATQuantizer | Int8DynActInt4WeightQATQuantizer | None, optional):
                Quantization-Aware Training quantizer instance. If provided, QAT will be
                applied to the model. Default: None.
            loss_fn (Callable | None, optional): Loss function to use. If the model
                doesn't support `logits_to_keep` and loss_fn is not MaskedCrossEntropy,
                it will be replaced with MaskedCrossEntropy. This is passed to AutoPipeline. Default: None.
            compile_config (CompileConfig | None, optional): Configuration for torch.compile.
                If provided, the model will be compiled for improved performance. Default: None.
            **kwargs:
                Additional keyword arguments. Notable ones include:
                - tp_size (int): Tensor parallelism size. Default: 1.
                - cp_size (int): Context parallelism size. Default: 1.
                - has_packed_sequence (bool): Whether using packed sequences. Default: False.
                - cache_dir (str): Cache directory for model weights.

        Returns:
            transformers.PreTrainedModel: The instantiated (and possibly
            kernel-patched) model with all infrastructure applied.

        Notes:
            If kernel patching fails, the partially constructed model is
            deleted and the method recurses once with
            `use_liger_kernel=False` or `use_sdpa_patching=False`.
        """

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
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                force_hf=force_hf,
                model_wrapper=model_wrapper,
                autopipeline=autopipeline,
                parallelize_fn=parallelize_fn,
                peft_config=peft_config,
                fp8_config=fp8_config,
                qat_quantizer=qat_quantizer,
                loss_fn=loss_fn,
                compile_config=compile_config,
                **kwargs,
            )

        torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch.bfloat16
        name_or_path = config if isinstance(config, str) else getattr(config, "name_or_path", None)
        kwargs["trust_remote_code"] = kwargs.get(
            "trust_remote_code", resolve_trust_remote_code(name_or_path) if name_or_path else False
        )
        config = get_hf_config(config, attn_implementation, **kwargs) if isinstance(config, str) else config
        # Treat config-related kwargs (e.g., output_hidden_states) as overrides on the
        # config object, not as model __init__ kwargs.
        _consume_config_overrides(config, kwargs)
        is_hf_model = get_is_hf_model(config, force_hf)
        tp_size, cp_size, has_packed_sequence = _pop_tp_cp_has_packed(kwargs)
        attn_implementation, use_liger_kernel = _apply_preload_overrides(
            is_hf_model, tp_size, cp_size, has_packed_sequence, attn_implementation, use_liger_kernel
        )
        device = torch.cuda.current_device()

        # Use meta device initialization when:
        # - Not using MegatronFSDPManager or DDPManager (they handle their own initialization)
        # - AND either multi-GPU (world_size > 1) or single-GPU custom model (not HF)
        # HF models on single GPU load weights via from_config, but multi-GPU needs meta device for sharding
        is_meta_device = not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager)) and (
            get_world_size_safe() > 1 or not is_hf_model
        )
        init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()

        try:
            with init_ctx:
                is_custom_model, model = _init_model(
                    cls, config, attn_implementation, torch_dtype, quantization_config, force_hf, *model_args, **kwargs
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
            if use_liger_kernel and not is_custom_model:
                model = _patch_liger_kernel(model)
        except RuntimeError:
            logging.warning("Retrying without Liger kernels.")
            del model
            gc.collect()
            return _retry(use_liger_kernel=False)

        # Patch sdpa attention
        try:
            if use_sdpa_patching and not is_custom_model:
                model = _patch_attention(model, sdpa_method)  # noqa: F821
        except:
            logging.warning("Retrying without SDPA patching.")
            return _retry(use_sdpa_patching=False)

        model = apply_model_infrastructure(
            model=model,
            is_hf_model=is_hf_model,
            cp_size=cp_size,
            tp_size=tp_size,
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
            pretrained_model_name_or_path=getattr(config, "name_or_path"),
            load_base_model=False,
            cache_dir=kwargs.get("cache_dir", hf_constants.HF_HUB_CACHE),
        )

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
