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

"""Infrastructure instantiation and application.

Distributed manager instantiation, sharding, PEFT/quantization application,
and checkpoint loading utilities.  These free functions operate on an
already-instantiated ``nn.Module`` and have no coupling to the
``_BaseNeMoAutoModelClass`` hierarchy.

``MeshContext`` (from ``mesh``) is the single source of truth
for device meshes, parallelism sizes, and axis names.
"""

import logging
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Optional, Union

import torch

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    _maybe_adapt_state_dict_to_hf,
)
from nemo_automodel.components.distributed.config import (
    DDPConfig,
    DistributedConfig,
    FSDP2Config,
    MegatronFSDPConfig,
)
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.moe.config import MoEParallelizerConfig
from nemo_automodel.components.quantization.fp8 import apply_fp8_to_model
from nemo_automodel.components.quantization.qat import QATConfig
from nemo_automodel.components.utils.compile_utils import compile_model
from nemo_automodel.components.utils.model_utils import (
    _supports_logits_to_keep,
    apply_parameter_freezing,
    init_empty_weights,
    print_trainable_parameters,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torchao.quantization.qat.linear import Int4WeightOnlyQATQuantizer, Int8DynActInt4WeightQATQuantizer

logger = logging.getLogger(__name__)


#  PEFT / quantization helpers
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
        # Skip freeze here - will do global freeze after checkpoint loading
        apply_lora_to_linear_modules(model, peft_config, quantization_config=quantization_config, skip_freeze=True)

    # FP8
    if fp8_config is not None:
        model = apply_fp8_to_model(model, config=fp8_config)

    # QAT
    if qat_quantizer is not None:
        from nemo_automodel.components.quantization.qat import prepare_qat_model

        if any(map(lambda x: x.dtype != torch.bfloat16, model.parameters())):
            raise NotImplementedError(
                "QAT is only supported for bfloat16 models. Support will be added in future release."
            )
        model, qat_mode = prepare_qat_model(model, qat_quantizer)
        # Attach helpers for delayed fake-quant toggling if desired
        model._qat_mode = qat_mode  # type: ignore[attr-defined]

    return model


#  Sharding helpers
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


def _shard_ep_fsdp(model, model_wrapper, parallelize_fn, mesh: MeshContext):
    """Apply EP + FSDP sharding (non-PP path)."""
    if parallelize_fn is not None and get_world_size_safe() > 1:
        parallelize_fn(
            model,
            world_mesh=mesh.device_mesh,
            moe_mesh=mesh.moe_mesh,
            **mesh.parallelize_axis_kwargs(),
        )
    elif callable(getattr(model_wrapper, "parallelize", None)):
        model = model_wrapper.parallelize(model)
        model = (
            model[0] if isinstance(model, tuple) else model
        )  # MegatronFSDP will return (model, None) since we don't pass optimizer here
    return model


#  Infrastructure instantiation (config -> runtime objects)
def _instantiate_distributed(
    config: DistributedConfig,
    mesh: MeshContext,
) -> Union[FSDP2Manager, MegatronFSDPManager, DDPManager, None]:
    """Instantiate the appropriate distributed manager from config.

    Args:
        config: Distributed config (FSDP2Config, MegatronFSDPConfig, or DDPConfig).
        mesh: MeshContext holding device_mesh and moe_mesh references.

    Returns:
        The instantiated manager, or None if config is None.

    Raises:
        ValueError: If device_mesh is required but not provided.
    """
    if config is None:
        return None

    if isinstance(config, FSDP2Config):
        if mesh.device_mesh is None:
            raise ValueError("device_mesh is required for FSDP2Config")
        return FSDP2Manager(config, device_mesh=mesh.device_mesh, moe_mesh=mesh.moe_mesh)
    elif isinstance(config, MegatronFSDPConfig):
        if mesh.device_mesh is None:
            raise ValueError("device_mesh is required for MegatronFSDPConfig")
        return MegatronFSDPManager(config, device_mesh=mesh.device_mesh)
    elif isinstance(config, DDPConfig):
        return DDPManager(config)
    else:
        raise ValueError(f"Unknown distributed config type: {type(config)}")


def _instantiate_pipeline(
    config: Optional[PipelineConfig],
    mesh: MeshContext,
    device: Optional[torch.device] = None,
) -> Optional[AutoPipeline]:
    """Instantiate AutoPipeline from config.

    Args:
        config: Pipeline config. If None or pp_size <= 1, returns None.
        mesh: MeshContext holding device_mesh, moe_mesh, and axis names.
        device: Target device for pipeline computation.

    Returns:
        AutoPipeline instance, or None if pipeline parallelism is not enabled.
    """
    if config is None or mesh.device_mesh is None or mesh.pp_size <= 1:
        return None

    config_dict = config.to_dict()
    config_dict.pop("loss_fn", None)

    return AutoPipeline(
        world_mesh=mesh.device_mesh,
        moe_mesh=mesh.moe_mesh,
        device=device,
        **mesh.pipeline_axis_kwargs(),
        **config_dict,
    )


def _instantiate_qat(
    config: Optional[QATConfig],
) -> Optional[Union["Int4WeightOnlyQATQuantizer", "Int8DynActInt4WeightQATQuantizer"]]:
    if config is None:
        return None
    return config.create_quantizer()


def parallelize_for_pp(
    model: torch.nn.Module,
    *,
    model_wrapper: Optional[Union[FSDP2Manager, MegatronFSDPManager, DDPManager]] = None,
    **kwargs,
) -> torch.nn.Module:
    """Parallelize model for pipeline parallelism (non-MoE case).

    This function adapts the pipeline parallelism interface to use model_wrapper.parallelize().
    For MoE models, use parallelize_model from nemo_automodel.components.moe.parallelizer directly.

    Args:
        model: The model to parallelize.
        model_wrapper: Distributed manager instance.
        **kwargs: Additional arguments (world_mesh, moe_mesh, axis names) passed by
            AutoPipeline but unused for non-MoE parallelization.

    Returns:
        The parallelized model.
    """
    if model_wrapper is not None:
        if callable(getattr(model_wrapper, "parallelize", None)):
            model = model_wrapper.parallelize(model)
    return model


def instantiate_infrastructure(
    *,
    distributed_config: Optional[DistributedConfig] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    qat_config: Optional[QATConfig] = None,
    moe_config: Optional[MoEParallelizerConfig] = None,
    activation_checkpointing: bool = False,
    device: Optional[torch.device] = None,
    mesh: Optional[MeshContext] = None,
    # Deprecated -- prefer passing ``mesh`` directly
    device_mesh: Optional["DeviceMesh"] = None,
    moe_mesh: Optional["DeviceMesh"] = None,
    ep_size: int = 1,
) -> tuple:
    """Instantiate infrastructure objects from config classes.

    This function converts config objects into the runtime objects needed by
    apply_model_infrastructure. It provides a cleaner, more HuggingFace-like API
    where users pass config objects instead of constructing runtime objects directly.

    Args:
        distributed_config: Distributed training config (FSDP2Config, MegatronFSDPConfig,
            or DDPConfig).
        pipeline_config: Pipeline parallelism config.
        qat_config: Quantization-aware training config.
        moe_config: MoE parallelizer config (for expert parallel models).
        activation_checkpointing: Enable activation checkpointing for transformer blocks.
            Defaults to False.
        device: Target device for model.
        mesh: MeshContext holding device meshes, sizes, and axis names.
            If None, built from the legacy ``device_mesh`` / ``moe_mesh`` params.
        device_mesh: (deprecated) Device mesh for distributed operations.
        moe_mesh: (deprecated) Optional MOE mesh for expert parallelism.
        ep_size: (deprecated) Expert parallelism size. Ignored when ``mesh`` is provided.

    Returns:
        tuple: (model_wrapper, autopipeline, parallelize_fn, qat_quantizer)
            - model_wrapper: Distributed manager instance (or None)
            - autopipeline: AutoPipeline instance (or None)
            - parallelize_fn: Parallelization function (or None) - built for EP
                (MoE-specific parallelizer when ep_size > 1) or PP (via model_wrapper)
            - qat_quantizer: QAT quantizer instance (or None)
    """
    if mesh is None:
        mesh = MeshContext.from_meshes(device_mesh, moe_mesh)

    ep_size = mesh.ep_size if mesh.ep_size > 1 else ep_size

    model_wrapper = _instantiate_distributed(distributed_config, mesh)
    autopipeline = _instantiate_pipeline(pipeline_config, mesh, device)

    parallelize_fn = None
    if ep_size > 1:
        from nemo_automodel.components.moe.parallelizer import parallelize_model

        parallelize_fn = partial(
            parallelize_model, activation_checkpointing=activation_checkpointing, **moe_config.to_dict()
        )
    elif autopipeline is not None and model_wrapper is not None:
        parallelize_fn = partial(parallelize_for_pp, model_wrapper=model_wrapper)

    qat_quantizer = _instantiate_qat(qat_config)

    return model_wrapper, autopipeline, parallelize_fn, qat_quantizer


#  apply_model_infrastructure  --  the main post-init orchestration function
def apply_model_infrastructure(
    model,
    *,
    is_meta_device,
    device,
    model_wrapper=None,
    mesh=None,
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
    - PEFT and lower precision application (LoRA, FP8, QAT)
    - Loss function setup
    - Pipeline parallelism or EP/FSDP sharding
    - Device placement and compilation
    - Checkpoint loading for meta device models

    Args:
        model: The model to apply infrastructure to
        is_meta_device: Whether model was initialized on meta device
        device: Target device for model
        model_wrapper: Model wrapper (FSDP2Manager, DDPManager, etc.). Default: None
        mesh: MeshContext with parallelism sizes (tp_size, cp_size, etc.) and mesh
            references. Default: None (treated as single-GPU defaults).
        peft_config: PEFT/LoRA configuration dict. Default: None
        quantization_config: Quantization configuration. Default: None
        fp8_config: FP8 configuration. Default: None
        qat_quantizer: QAT quantizer instance. Default: None
        loss_fn: Loss function (may be replaced with MaskedCrossEntropy). Default: None
        autopipeline: AutoPipeline instance for pipeline parallelism. Default: None
        parallelize_fn: Function to apply parallelization (EP + FSDP2). Default: None
        compile_config: Compilation configuration. Default: None
        pretrained_model_name_or_path: Model name or path for checkpoint loading. Default: ""
        load_base_model: Whether to load base model weights (True for from_pretrained). Default: False
        cache_dir: Cache directory for model weights. Default: None
        **_kwargs: Additional keyword arguments (ignored, allows passing extra kwargs)

    Returns:
        The model with all infrastructure applied
    """
    if mesh is None:
        mesh = MeshContext()

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
        getattr(model_wrapper, "moe_mesh", None),
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
            model, mesh.tp_size, autopipeline, peft_config, quantization_config, fp8_config, qat_quantizer
        )

    # When no PP and no TP, load checkpoint first (unwrapped model) so weights and dtypes come from
    # the checkpoint; then apply FSDP. With TP>1 we must shard first and load after so all ranks
    # stay in sync (load-before-shard can cause NCCL collective mismatch). With PP we shard first
    # (each stage has different layers).
    # Skip load-before-shard for PEFT: base load into unwrapped PEFT then later adapter load
    # after shard can leave base/adapter out of sync (e.g. key/device mismatch). Use the
    # post-shard load path so base and adapter load in the same way as multi-GPU.
    no_pp = autopipeline is None
    no_tp = mesh.tp_size <= 1
    need_checkpoint_load = bool(pretrained_model_name_or_path and load_base_model)
    load_before_shard = (
        no_pp and no_tp and need_checkpoint_load and (peft_config is None)
    )

    checkpoint_already_loaded = False
    if load_before_shard:
        lora_a_init = getattr(peft_config, "lora_A_init", None)
        checkpointer.load_base_model(
            model,
            device,
            cache_dir,
            pretrained_model_name_or_path,
            lora_a_init,
            load_base_model=load_base_model,
        )
        checkpoint_already_loaded = True

    # hold a list copy of the model state dict keys before any parallelization. To be used during checkpoint saving in safetensors format.
    pre_shard_hf_state_dict_keys = list(
        _maybe_adapt_state_dict_to_hf(model, model.state_dict(), quantization=dequantize_base_checkpoint).keys()
    )

    # Apply freezing before sharding
    freeze_config = _kwargs.get("freeze_config")
    if freeze_config is not None:
        apply_parameter_freezing(model, freeze_config)

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
        model = _shard_ep_fsdp(model, model_wrapper, parallelize_fn, mesh)
        if compile_config is not None:
            model = compile_model(model, compile_config)
        if isinstance(model_wrapper, DDPManager):
            setattr(model.module, "_pre_shard_hf_state_dict_keys", pre_shard_hf_state_dict_keys)
        else:
            setattr(model, "_pre_shard_hf_state_dict_keys", pre_shard_hf_state_dict_keys)

    # Load the checkpoint if needed (meta path, or PP/TP path where we did not load before shard)
    should_load_checkpoint = (
        need_checkpoint_load
        and not checkpoint_already_loaded
        and (
            is_meta_device
            and any(
                [
                    get_world_size_safe() == 1,
                    parallelize_fn is not None and get_world_size_safe() > 1,
                    callable(getattr(model_wrapper, "parallelize", None)),
                ]
            )
        )
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

    # Freeze parameters after checkpoint loading and parallelization
    # This catches params created during parallelization (e.g., GroupedExpertsTE in init_token_dispatcher)
    if peft_config is not None:
        models_to_freeze = model.parts if hasattr(model, "parts") else [model]
        for mp in models_to_freeze:
            for name, param in mp.named_parameters():
                if "lora_" not in name and param.requires_grad:
                    param.requires_grad_(False)

    if autopipeline is None:
        print_trainable_parameters(model)  # Once model's been sharded
        # Ensure model is on the correct device; AutoPipeline takes care of it internally
        try:
            model.to(device)
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                logger.warning("model.to(device) failed (meta tensors); using model.to_empty(device=device) instead.")
                model.to_empty(device=device)
            else:
                raise

    return model
