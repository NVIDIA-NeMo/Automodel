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
from functools import lru_cache
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


# TODO(boxiangw): Change to nvFSDP once it got published
HAVE_NVFSDP = False
try:
    from nvfsdp import fully_shard as nvfsdp_fully_shard
    HAVE_NVFSDP = True
except:
    pass


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy = None,
    tp_shard_plan: Optional[
        Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]
    ] = None,
    offload_policy: "CPUOffloadPolicy" = None,
):
    """
    Apply parallelisms and activation checkpointing to the model.

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        mp_policy (MixedPrecisionPolicy): Mixed precision policy for model parallelism.
        tp_shard_plan (Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]]):
            A tensor parallel sharding plan. The keys should be the module names and the values should be the
            corresponding parallel styles (e.g., RowwiseParallel, ColwiseParallel, SequenceParallel).
        offload_policy (CPUOffloadPolicy): The offload policy for FSDP. If None, it will use the default policy.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    NOTE: Currently, the user is required to manually handle precision settings such as the `mp_policy` here
    because the model parallel strategy does not respect all settings of `Fabric(precision=...)` at the moment.
    NOTE: Currently, the user should make sure that custom_tp_plan is compatible with the model architecture.
    """
    if not mp_policy:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )

    def parallelize_helper(module, mesh, mp_policy):
        if isinstance(module, nn.ModuleList):
            for layer_id, transformer_block in enumerate(module):
                # Apply activation checkpointing
                # transformer_block = checkpoint_wrapper(transformer_block)
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
                parallelize_helper(sub_module, mesh, mp_policy)

    # Set FSDP sharding mesh to context parallel mesh if CP > 1, else default to the data parallel mesh.
    dp_mesh = device_mesh[
        (
            "dp_cp"
            if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(device_mesh, {})
            else "data_parallel"
        )
    ]

    if dp_mesh.size() > 1:
        assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    tp_mesh = device_mesh["tensor_parallel"]
    # TP sharding
    if tp_mesh.size() > 1:
        assert tp_shard_plan is not None
        parallelize_module(model, tp_mesh, tp_shard_plan)

    # FSDP sharding
    assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    # Find transformer layers and apply parallelisms
    parallelize_helper(model, dp_mesh, mp_policy)

    # reshard_after_forward=True based on
    # https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L359
    model = fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
        offload_policy=offload_policy,
    )

    return model


def import_classes_from_paths(class_paths: List[str]):
    """
    Helper function to import classes from string paths.

    Args:
        class_paths (List[str]): The list of string paths to the classes.
    """
    classes = []
    for path in class_paths:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        classes.append(cls)
    return classes


def nvfsdp_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    nvfsdp_unit_modules: Optional[List[str]] = None,
    tp_shard_plan: Optional[
        Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]
    ] = None,
    data_parallel_sharding_strategy: str = "optim_grads_params",
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

    NOTE: The passed-in model should preferably reside on the meta device.
    Otherwise, ensure the model fits into available GPU or CPU memory.

    NOTE: The user must ensure that the provided tp_shard_plan is compatible
    with the model architecture.
    """
    assert HAVE_NVFSDP, "nvFSDP is not installed, please visit \
        https://github.com/NVIDIA-NeMo/nvFSDP for more information"

    # DP_CP ranks are sharded by FSDP.
    dp_mesh = device_mesh[
        (
            "dp_cp"
            if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(device_mesh, {})
            else "data_parallel"
        )
    ]
    tp_mesh = device_mesh["tensor_parallel"]

    if dp_mesh.size() > 1:
        # TODO(boxiangw): remove this once HSDP is supported.
        assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    # TP sharding.
    if tp_mesh.size() > 1:
        parallelize_module(model, tp_mesh, tp_shard_plan)

    # Import nvFSDP unit modules specified by the user.
    nvfsdp_unit_modules = import_classes_from_paths(nvfsdp_unit_modules)

    # Wrap model with nvFSDP.
    model = nvfsdp_fully_shard(
        module=model,
        fsdp_unit_modules=nvfsdp_unit_modules,
        dp_cp_group=dp_mesh.get_group(),
        init_model_with_meta_device=init_nvfsdp_with_meta_device,
        data_parallel_sharding_strategy=data_parallel_sharding_strategy,
        init_nvfsdp_with_meta_device=init_nvfsdp_with_meta_device,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        check_for_nan_in_grad=check_for_nan_in_grad,
        average_in_collective=average_in_collective,
        disable_bucketing=disable_bucketing,
        calculate_per_token_loss=calculate_per_token_loss,
        keep_fp8_transpose_cache_when_using_custom_fsdp=keep_fp8_transpose_cache_when_using_custom_fsdp,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
    )

    return model


def get_hf_tp_shard_plan(model):
    """
    Get the tensor parallel sharding plan from the model.
    """
    hf_tp_shard_plan = {}
    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_shard_plan.update(model._tp_plan)
    if hasattr(model.model, "_tp_plan") and model.model._tp_plan is not None:
        hf_tp_shard_plan.update(
            {f"model.{k}": v for k, v in model.model._tp_plan.items()}
        )

    hf_tp_shard_plan = {
        k: translate_to_torch_parallel_style(v) for k, v in hf_tp_shard_plan.items()
    }
    return hf_tp_shard_plan


@lru_cache
def translate_to_torch_parallel_style(style: str):
    """
    Translates string descriptions to parallelism plans.

    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we translate them into torch.distributed tensor-parallel
    types.
    """
    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

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
        raise ValueError(f"Unsupported parallel style value: {style}")


def to_cpu(v):
    """
    Move a tensor or distributed tensor to the CPU.

    This function takes an input tensor, which can be either a `DTensor` (distributed tensor)
    or a standard `Tensor`, and ensures that it is moved to the CPU.

    Args:
        v (DTensor | Tensor | any): The input value, which can be a `DTensor`, `Tensor`, or
                                    any other object. If `DTensor`, it checks the device and
                                    moves the tensor accordingly.

    Returns:
        Tensor | any: The corresponding CPU tensor if `v` is a `DTensor` or `Tensor`,
                    otherwise returns `v` unchanged.

    Raises:
        ValueError: If `v` is a `DTensor` but its device is neither 'cuda' nor 'cpu'.

    Example:
        >>> t = torch.tensor([1, 2, 3], device='cuda')
        >>> to_cpu(t)  # Moves tensor to CPU
        tensor([1, 2, 3])

        >>> dt = DTensor(torch.tensor([4, 5, 6], device='cuda'))
        >>> to_cpu(dt)  # Moves DTensor to CPU
        tensor([4, 5, 6])
    """
    if isinstance(v, DTensor):
        if v.device.type == "cuda":
            return v.full_tensor().cpu()
        elif v.device.type == "cpu":
            return v._local_tensor
        else:
            raise ValueError("Unknown device " + str(v.device))
    elif isinstance(v, Tensor):
        return v.cpu()
    else:
        return v


def _destroy_dist_connection() -> None:
    """
    Destroy process group.
    """
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)

