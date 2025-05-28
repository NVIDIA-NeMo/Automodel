from typing import Any, Optional
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from automodel.utils.import_utils import safe_import_from
from dataclasses import dataclass, field

MixedPrecisionPolicy, HAS_MIXED_PRECISION_POLICY = safe_import_from(
    "torch.distributed.fsdp", "MixedPrecisionPolicy", fallback_module="torch.distributed._composable.fsdp"
)
fully_shard, HAS_FULLY_SHARD = safe_import_from(
    "torch.distributed.fsdp", "fully_shard", fallback_module="torch.distributed._composable.fsdp"
)
CPUOffloadPolicy, HAS_CPU_OFFLOAD_POLICY = safe_import_from(
    "torch.distributed.fsdp", "CPUOffloadPolicy", fallback_module="torch.distributed._composable.fsdp"
)
from automodel.distributed.parallelizer import fsdp2_strategy_parallelize, get_hf_tp_shard_plan


@dataclass
class FSDP2Manager:
    """
    Manager for setting up and parallelizing models using FSDP2 with Tensor‐Parallel,
    Data‐Parallel, and Context‐Parallel sharding strategies.

    This manager initializes the torch.distributed process group, infers the group sizes
    for data parallelism (DP) and tensor parallelism (TP), builds the device mesh for
    distributed operations, and applies parallelization to the model using a prescribed
    TP sharding plan. It also supports mixed precision and CPU offloading options.

    Attributes:
        dp_size (Optional[int]): Data‐parallel group size. If None or non-positive, it is
            inferred from WORLD_SIZE.
        tp_size (Optional[int]): Tensor‐parallel group size. Defaults to 1 if zero/None.
        cp_size (int): Context‐parallel group size for pipeline‐like sharding.
        sequence_parallel (bool): Enables sequence parallelism in the TP plan when True.
        mp_policy (MixedPrecisionPolicy): Defines the mixed precision policy for parameters,
            reductions, and outputs.
        offload_policy (CPUOffloadPolicy): Policy to offload parameters or optimizer states
            to CPU, if specified.
        backend (str): Distributed backend to use (e.g., 'nccl' for GPUs or 'gloo' for CPUs).
        world_size (int): Total number of processes.

    Private Attributes:
        _device_mesh (Any): The Torch distributed DeviceMesh built for managing device assignments.
        _rank (int): Global rank of the current process.

    Methods:
        __post_init__():
            Automatically sets up the distributed environment after initialization.
        _setup_distributed():
            Initializes the torch.distributed process group, infers parallel sizes,
            builds the device mesh, and registers a destroy handler.
        parallelize(model):
            Applies FSDP2 and Tensor‐Parallel sharding strategies to the given model.
    """
    dp_size: Optional[int] = field(
        default=None,
        metadata={"help": "Data‐parallel group size; if None, infer from WORLD_SIZE."}
    )
    tp_size: Optional[int] = field(
        default=1,
        metadata={"help": "Tensor‐parallel group size; if None, defaults to 1."}
    )
    cp_size: int = field(
        default=1,
        metadata={"help": "Context‐parallel group size (for pipeline‐like sharding)."}
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Enable sequence parallelism in TP plan if True."}
    )
    mp_policy: MixedPrecisionPolicy = field(
        default=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            ),
        metadata={"help": "MixedPrecisionPolicy for FSDP2 (param/reduce/output dtypes)."}
    )
    offload_policy: CPUOffloadPolicy = field(
        default=None,
        metadata={"help": "CPUOffloadPolicy to offload parameters/optim states to CPU."}
    )
    backend: str = field(
        default="nccl",
        metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."}
    )

    _device_mesh: Any = field(
        default=None,
        init=False,
        metadata={"help": "Torch distributed DeviceMesh."}
    )
    _rank: int = field(
        default=None,
        init=False,
        metadata={"help": "Global rank of this process."}
    )
    world_size: int = field(
        default=None,
        # init=False,
        metadata={"help": "Total number of processes."}
    )

    def __post_init__(self):
        """
        Post-initialization hook that sets up the distributed environment.
        """
        return self._setup_distributed()

    def _setup_distributed(self):
        """
        Initializes the distributed environment:

        - Checks availability and initialization of torch.distributed.
        - Infers data-parallel and tensor-parallel sizes if not provided.
        - Builds a device mesh based on the specified mesh shape and dimension names.
        - Flattens data and context dimensions if context parallelism is enabled.

        Requires the environment variables: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.

        Raises:
            RuntimeError: If torch.distributed is not available or not initialized.

        Returns:
            FSDP2Manager: Instance with the device mesh configured.
        """
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")

        if not dist.is_initialized():
            raise RuntimeError("expected torch.distributed to be initialized")

        # infer if not provided
        self.dp_size = self.dp_size
        if self.dp_size is None or self.dp_size <= 0:
            self.dp_size = self.world_size
        self.tp_size = self.tp_size or 1

        # build mesh [dp, cp, tp]
        mesh_shape = (self.dp_size, self.cp_size, self.tp_size)
        mesh_names = ("data_parallel", "context_parallel", "tensor_parallel")
        self.device_mesh = init_device_mesh(
            device_type="cuda" if self.backend == "nccl" else "cpu",
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_names,
        )
        # flatten dp+cp if cp>1
        if self.cp_size > 1:
            self.device_mesh[("data_parallel", "context_parallel")]._flatten(mesh_dim_name="dp_cp")
        return self

    def parallelize(self, model, use_hf_tp_plan=False):
        """
        Parallelizes the given model using FSDP2 and TP sharding strategies.

        This method must be called after the distributed environment has been set up.
        It selects a TP sharding plan (currently supporting Hugging Face
        TP plan via get_hf_tp_shard_plan) and applies the FSDP2 parallelization strategy.

        Args:
            model: The model to be parallelized.

        Returns:
            The parallelized model.

        Raises:
            NotImplemented: If the required TP sharding plan is not supported.
        """
        if use_hf_tp_plan:
            tp_shard_plan = get_hf_tp_shard_plan(model)
        else:
            # TODO (tp-plan)
            tp_shard_plan = None

        fsdp2_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            mp_policy=self.mp_policy,
            tp_shard_plan=tp_shard_plan,
            offload_policy=self.offload_policy,
        )
        return model
