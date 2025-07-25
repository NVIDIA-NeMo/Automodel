# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import signal
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

# Import model-specific tensor parallel plans from the dedicated module
from nemo_automodel.components.distributed.optimized_tp_plans import PARALLELIZE_FUNCTIONS

# TODO(boxiangw): Change to nvFSDP once it got published
HAVE_NVFSDP = False
try:
    from nvfsdp import fully_shard as nvfsdp_fully_shard

    HAVE_NVFSDP = True
except:
    pass


def apply_fsdp_sharding_recursively(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy],
    offload_policy: Optional[CPUOffloadPolicy] = None,
) -> None:
    """
    Recursively apply FSDP sharding to modules, with optimizations for ModuleList.

    This utility function traverses a model hierarchy and applies FSDP sharding
    to each module. For ModuleList instances (commonly used for transformer layers),
    it applies an optimization where the last layer doesn't reshard after forward
    since FSDP will prefetch it immediately.

    Args:
        module (nn.Module): The module to apply FSDP sharding to.
        mesh (DeviceMesh): The device mesh for FSDP sharding.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for FSDP.
        offload_policy (Optional[CPUOffloadPolicy]): CPU offload policy for FSDP.
            Defaults to None.

    Note:
        This function modifies the module in-place by replacing modules with their
        FSDP-wrapped versions.
    """
    if isinstance(module, nn.ModuleList):
        for layer_id, transformer_block in enumerate(module):
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(module) - 1
            fully_shard(
                transformer_block,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,
            )
            module[layer_id] = transformer_block
    else:
        for name, sub_module in module.named_children():
            apply_fsdp_sharding_recursively(sub_module, mesh, mp_policy, offload_policy)


def get_hf_tp_shard_plan(model):
    """Get the Hugging Face tensor parallel plan from the model.

    This function:
    - Retrieves TP strategies from model class, instance, and inner model levels.
    - Handles special cases for `embed_tokens` and `lm_head` for speed up.
    - Converts string-based parallel styles to DTensor parallelization strategies.

    Taken and modified from: https://github.com/NVIDIA/NeMo/blob/6c6169db01bcca73ae8ad3ac35242fadbb9a78ba/nemo/lightning/pytorch/strategies/utils.py#L532

    Args:
        model: A Hugging Face model instance

    Returns:
        dict: A dictionary mapping model component paths to their parallelization strategies

    Raises:
        AssertionError: If no TP plan is found
    """
    model_cls = type(model)
    if isinstance(model, Gemma3ForConditionalGeneration):
        inner_model = model.language_model
        model_prefix = "language_model"
    else:
        inner_model = model.model
        model_prefix = "model"

    hf_tp_plan = {}

    # model_cls._tp_plan will override model_cls after xxxForCausalLM.post_init() (transformers==4.51.3)
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        hf_tp_plan.update(model_cls._tp_plan)

    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)

    if hasattr(inner_model, "_tp_plan") and inner_model._tp_plan is not None:
        hf_tp_plan.update({f"{model_prefix}.{k}": v for k, v in inner_model._tp_plan.items()})

    assert len(hf_tp_plan) > 0, (
        f"Hugging Face tp plan is not supported for {model_cls}, please set dtensor_cfg.tensor_parallel_size to 1 or provide a custom_parallel_plan. "
        "The usage example of custom_parallel_plan can refer to `docs/design-docs/fsdp2-parallel-plan.md`."
    )

    # hf tp plan not contain embed_tokens, we add it and set to rowwise_rep
    if f"{model_prefix}.embed_tokens" not in hf_tp_plan and not model.config.tie_word_embeddings:
        hf_tp_plan[f"{model_prefix}.embed_tokens"] = "rowwise_rep"

    for k, v in hf_tp_plan.items():
        # speed up the tp plan for lm_head
        if k == "lm_head" and v == "colwise_rep" and not model.config.tie_word_embeddings:
            hf_tp_plan[k] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
        else:
            hf_tp_plan[k] = translate_to_torch_parallel_style(v)

    return hf_tp_plan


def import_class_from_path(name: str) -> Any:
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


def import_classes_from_paths(class_paths: List[str]):
    """
    Helper function to import classes from string paths.

    Args:
        class_paths (List[str]): The list of string paths to the classes.

    Returns:
        List of imported classes.
    """
    classes = []
    for path in class_paths:
        try:
            cls = import_class_from_path(path)
            classes.append(cls)
        except Exception as e:
            print(f"Warning: Could not import class from path '{path}': {e}")
    return classes


@lru_cache
def translate_to_torch_parallel_style(style: str):
    """
    Translates string descriptions to parallelism plans.

    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we translate them into torch.distributed tensor-parallel
    types.
    """
    assert isinstance(style, str), f"parallel style type should be str, but got {type(style)}"

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unknown parallel style: {style}")


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    param_dtype: torch.dtype = torch.bfloat16,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[CPUOffloadPolicy] = None,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    cpu_offload: bool = False,
    tp_shard_plan: Optional[Union[Dict[str, ParallelStyle], str]] = None,
    dp_mesh_name: str = "data_parallel",
    tp_mesh_name: str = "tensor_parallel",
    dp_cp_mesh_name: str = "dp_cp",
):
    """
    Apply parallelisms and activation checkpointing to the model.

    Enhanced version that incorporates advanced features from nemo-rl's _parallelize_model:
    - Automatic parallel plan generation based on model type
    - Custom parallel plan support (dict or string path)
    - Sequence parallel support
    - Activation checkpointing for MLP layers
    - Model validation (attention heads divisible by TP size)
    - Better fallback logic

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        param_dtype (torch.dtype): Data type for model parameters. Defaults to torch.bfloat16.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for model parallelism.
        offload_policy (Optional[CPUOffloadPolicy]): The offload policy for FSDP.
        sequence_parallel (bool): Whether to use sequence parallelism. Defaults to False.
        activation_checkpointing (bool): Whether to use activation checkpointing. Defaults to False.
        cpu_offload (bool): Whether to enable cpu offloading for FSDP. Defaults to False.
        tp_shard_plan (Optional[Union[Dict[str, ParallelStyle], str]]):
            Custom tensor parallel plan for the model. Can be:
            - A dictionary mapping module names to parallel styles
            - A string path to a dictionary or function that returns a dictionary
            If provided, this takes precedence over automatic plan generation.
        dp_mesh_name (str): Key name for the data parallel mesh in device_mesh.
            Defaults to "data_parallel".
        dp_cp_mesh_name (str): Key name for the data parallel + context parallel mesh in device_mesh.
            Used when context parallelism is enabled. Defaults to "dp_cp".
        tp_mesh_name (str): Key name for the tensor parallel mesh in device_mesh.
            Defaults to "tensor_parallel".

    Returns:
        The parallelized model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # Get model layers for later use
    model_cls = type(model)
    if isinstance(model, Gemma3ForConditionalGeneration):
        layers = model.language_model.layers
        num_attention_heads = model.config.text_config.num_attention_heads
        num_key_value_heads = model.config.text_config.num_key_value_heads
    else:
        layers = model.model.layers
        num_attention_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads

    # Set FSDP sharding mesh to context parallel mesh if CP > 1, else default to the data parallel mesh.
    dp_mesh = device_mesh[
        (
            dp_cp_mesh_name
            if dp_cp_mesh_name in _mesh_resources.root_to_flatten_mapping.get(device_mesh, {})
            else dp_mesh_name
        )
    ]

    if dp_mesh.size() > 1:
        assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    tp_mesh = device_mesh[tp_mesh_name]

    # TP sharding with enhanced plan generation
    if tp_mesh.size() > 1:
        # Validate that attention heads are divisible by TP size
        assert num_key_value_heads % tp_mesh.size() == 0, (
            f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
        )
        assert num_attention_heads % tp_mesh.size() == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
        )

        # Generate or use tensor parallel plan
        model_parallel_plan = None

        # 1. Use custom parallel plan if provided
        if tp_shard_plan is not None:
            if isinstance(tp_shard_plan, dict):
                model_parallel_plan = tp_shard_plan
                print("Using provided parallel plan (dictionary).")
            else:
                try:
                    plan_obj = import_class_from_path(tp_shard_plan)
                    if isinstance(plan_obj, FunctionType):
                        model_parallel_plan = plan_obj()
                    else:
                        model_parallel_plan = plan_obj
                    assert isinstance(model_parallel_plan, dict), (
                        f"Parallel plan must be a dictionary, got {type(model_parallel_plan)}"
                    )
                    print("Using provided parallel plan (from path).")
                except Exception as e:
                    raise ValueError(
                        f"Custom parallel plan '{tp_shard_plan}' is not valid. "
                        f"Please ensure it is one of the following:\n"
                        "1. A dictionary mapping module names to parallel styles\n"
                        "2. A path to a dictionary\n"
                        "3. A path to a function that returns a dictionary\n"
                        f"Error: {e}"
                    )

        # 2. Use optimized parallel plan based on model type
        elif model_cls in PARALLELIZE_FUNCTIONS:
            try:
                func = PARALLELIZE_FUNCTIONS[model_cls]
                model_parallel_plan = func(model, sequence_parallel)
                print("Using optimized parallel plan.")
            except Exception as e:
                print(f"Optimized parallel plan is not available: {e}. Falling back to the HF tp plan.")
                assert not sequence_parallel, "sequence_parallel is not supported in HF tp plan."
                model_parallel_plan = get_hf_tp_shard_plan(model)

        # 3. Use HF TP plan as fallback
        else:
            if model_cls not in PARALLELIZE_FUNCTIONS:
                print(f"Optimized parallel plan is not supported for {model_cls}. Falling back to the HF tp plan.")
            assert not sequence_parallel, "sequence_parallel is not supported in HF tp plan."
            model_parallel_plan = get_hf_tp_shard_plan(model)

        # Apply tensor parallelism
        if model_parallel_plan:
            parallelize_module(model, tp_mesh, model_parallel_plan)

    # Apply activation checkpointing to MLP layers if requested
    if activation_checkpointing:
        for i, layer in enumerate(layers):
            if hasattr(layer, "mlp"):
                layers[i].mlp = checkpoint_wrapper(layer.mlp)

    # Set up mixed precision policy
    if not mp_policy:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )

    # Set up offload policy
    if not offload_policy:
        offload_policy = CPUOffloadPolicy(pin_memory=False) if cpu_offload else None

    # FSDP sharding
    assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    # Find transformer layers and apply parallelisms
    apply_fsdp_sharding_recursively(model, dp_mesh, mp_policy, offload_policy)

    # Apply FSDP to the root model
    # Do not reshard after forward for root model because its parameters
    # will be used in backward immediately
    model = fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=False,
        offload_policy=offload_policy,
    )

    return model


def nvfsdp_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    optimizer=None,
    nvfsdp_unit_modules: Optional[List[str]] = None,
    tp_shard_plan: Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]] = None,
    zero_dp_strategy: int = 3,
    init_nvfsdp_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = False,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache_when_using_custom_fsdp: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    dp_mesh_name: str = "data_parallel",
    cp_mesh_name: str = "context_parallel",
    tp_mesh_name: str = "tensor_parallel",
):
    """
    Apply tensor/data parallelism (nvFSDP) and optional activation-checkpointing to the model.

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh describing the physical devices
            used for distributed training.
        nvfsdp_unit_modules (Optional[List[str]]): Names of sub-modules that should
            become individual nvFSDP units. If None, the full model is wrapped as
            a single unit.
        tp_shard_plan (Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]]):
            A tensor-parallel sharding plan.
            Keys are module names; values specify the parallel style to apply
            (e.g., RowwiseParallel, ColwiseParallel, SequenceParallel).
        data_parallel_sharding_strategy (str): Strategy for sharding parameters,
            gradients, and optimizer states across data-parallel ranks.
            Valid options include "params", "grads_params", and
            "optim_grads_params" (default).
        init_nvfsdp_with_meta_device (bool): If True, construct the model on a
            meta device first and materialize weights lazily to reduce memory
            fragmentation.
        grad_reduce_in_fp32 (bool): Reduce gradients in FP32 irrespective of the
            parameter precision to improve numerical stability.
        preserve_fp32_weights (bool): Keep a master FP32 copy of weights when
            training in reduced precision (e.g., FP16/BF16).
        overlap_grad_reduce (bool): If True, overlap gradient reduction with
            backward computation.
        overlap_param_gather (bool): If True, overlap parameter gathering with
            forward computation.
        check_for_nan_in_grad (bool): Whether to check gradients for NaNs/Infs
            before applying the optimizer step.
        average_in_collective (bool): Perform gradient averaging inside the
            collective operation instead of dividing afterward.
        disable_bucketing (bool): Disable gradient bucketing; gradients are
            reduced immediately as they are produced.
        calculate_per_token_loss (bool): Compute loss normalized by the number of
            tokens instead of the number of sequences.
        keep_fp8_transpose_cache_when_using_custom_fsdp (bool): Retain the FP8
            transpose cache when using a custom nvFSDP wrapper.
        nccl_ub (bool): Enable NCCL user-buffer API (experimental) for reduced
            latency on some networks.
        fsdp_double_buffer (bool): Enable double buffering of parameters to
            overlap communication and computation in nvFSDP.
        dp_mesh_name (str): Key name for the data parallel mesh in device_mesh.
            Defaults to "data_parallel".
        cp_mesh_name (str): Key name for the context parallel mesh in device_mesh.
            Defaults to "context_parallel".
        tp_mesh_name (str): Key name for the tensor parallel mesh in device_mesh.
            Defaults to "tensor_parallel".

    NOTE: The passed-in model should preferably reside on the meta device.
    Otherwise, ensure the model fits into available GPU or CPU memory.

    NOTE: The user must ensure that the provided tp_shard_plan is compatible
    with the model architecture.
    """
    assert HAVE_NVFSDP, (
        "nvFSDP is not installed, please visit \
        https://github.com/NVIDIA-NeMo/nvFSDP for more information"
    )

    # DP_CP ranks are sharded by FSDP.
    dp_mesh = device_mesh[dp_mesh_name]
    cp_mesh = device_mesh[cp_mesh_name]
    tp_mesh = device_mesh[tp_mesh_name]

    if dp_mesh.size() > 1:
        # TODO(boxiangw): remove this once HSDP is supported.
        assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    # TP sharding.
    if tp_mesh.size() > 1:
        parallelize_module(model, tp_mesh, tp_shard_plan)

    if cp_mesh.size() > 1:
        dp_cp_mesh_name = "dp_cp"
    else:
        dp_cp_mesh_name = "data_parallel"

    # Import nvFSDP unit modules specified by the user.
    nvfsdp_unit_modules = import_classes_from_paths(nvfsdp_unit_modules)

    # Wrap model with nvFSDP.
    model, optimizer = nvfsdp_fully_shard(
        module=model,
        optimizer=optimizer,
        fsdp_unit_modules=nvfsdp_unit_modules,
        device_mesh=device_mesh,
        dp_mesh_dim_name=dp_mesh_name,
        cp_mesh_dim_name=cp_mesh_name,
        tp_mesh_dim_name=tp_mesh_name,
        dp_cp_mesh_dim_name=dp_cp_mesh_name,
        zero_dp_strategy=zero_dp_strategy,
        init_model_with_meta_device=init_nvfsdp_with_meta_device,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        sync_grads_each_step=False,  # For better performance, avoid sync every step
        check_for_nan_in_grad=check_for_nan_in_grad,
        average_in_collective=average_in_collective,
        disable_bucketing=disable_bucketing,
        calculate_per_token_loss=calculate_per_token_loss,
        keep_fp8_transpose_cache_when_using_custom_fsdp=keep_fp8_transpose_cache_when_using_custom_fsdp,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
    )

    return model, optimizer


def _destroy_dist_connection() -> None:
    """
    Destroy process group.
    """
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)


@contextmanager
def unshard_fsdp2_model(model: nn.Module) -> Generator[None, None, None]:
    """Explicitly unshard and then reshard the FSDP2 modules. Useful for logprob inference."""
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
