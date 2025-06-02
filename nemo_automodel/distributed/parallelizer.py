import contextlib
import importlib
import signal
from functools import lru_cache
from typing import Dict, Generator, List, Optional, Set, Union

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
from nemo_automodel.distributed.nvfsdp.nvfsdp import nvFSDP
from nemo_automodel.distributed.nvfsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)


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
    """Apply parallelisms and activation checkpointing to the model.

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

    if "tensor_parallel" in device_mesh.mesh_dim_names:
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
    nvfsdp_config: Optional[DistributedDataParallelConfig] = None,
    nvfsdp_unit_modules: Optional[List[str]] = None,
    init_nvfsdp_with_meta_device: bool = False,
    tp_shard_plan: Optional[
        Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]
    ] = None,
):
    """Apply parallelisms and activation checkpointing to the model.

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        nvfsdp_config (DistributedDataParallelConfig): The distributed data parallel config.
        nvfsdp_unit_modules (Optional[List[str]]): The nvFSDP unit modules.
        init_nvfsdp_with_meta_device (bool): Whether to initialize the model with meta device.
        tp_shard_plan (Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]]):
            A tensor parallel sharding plan. The keys should be the module names and the values should be the
            corresponding parallel styles (e.g., RowwiseParallel, ColwiseParallel, SequenceParallel).
    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    NOTE: Currently, the user should make sure that custom_tp_plan is compatible with the model architecture.
    """

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

    if nvfsdp_config is None:
        # Default DDP config for nvFSDP.
        # data_parallel_sharding_strategy="optim_grads_params" is required to shard the parameters. (ZeRO-3)
        nvfsdp_config = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            data_parallel_sharding_strategy="optim_grads_params",
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=False,
        )

    # Import nvFSDP unit modules specified by the user.
    nvfsdp_unit_modules = import_classes_from_paths(nvfsdp_unit_modules)

    # Wrap model with nvFSDP.
    model = nvFSDP(
        ddp_config=nvfsdp_config,
        module=model,
        fsdp_unit_modules=nvfsdp_unit_modules,
        dp_cp_group=dp_mesh.get_group(),
        calculate_per_token_loss=False,
        init_model_with_meta_device=init_nvfsdp_with_meta_device,
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
    """Destroy process group."""
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: str,
):
    """
    Create a context parallel context.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        cp_buffers (List[torch.Tensor]): The buffers for context parallel.
        cp_seq_dims (List[int]): The sequence dimensions for context parallel.
        cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
        cp_rotate_method (str): The rotation method for context parallel, such as "allgather" or "addtoall".
    """
    from torch.distributed.tensor.experimental import context_parallel

    # TODO: uncomment this when torch.distributed.tensor.experimental._attention.set_rotate_method is available
    # from torch.distributed.tensor.experimental._attention import set_rotate_method
    # set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L138
def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    """
    Create a train context.

    Args:
        enable_loss_parallel (bool): Whether to enable loss parallelism.
        enable_compiled_autograd (bool): Whether to enable compiled autograd.
    """

    @contextlib.contextmanager
    def context(cp_context: Optional[Generator[None, None, None]] = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # currently we only support these two SDP backends.
                # TODO (xilunwu): support cuDNN backend
                stack.enter_context(
                    sdpa_kernel(
                        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context
