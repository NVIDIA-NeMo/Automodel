import os
import re
import shutil
import atexit
from pathlib import Path
from contextlib import contextmanager

from typing import Any, Dict, Union, Optional
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._api import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
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
    dp_size: Optional[int] = field(
        default=None,
        metadata={"help": "Data‐parallel group size; if None, infer from WORLD_SIZE."}
    )
    tp_size: Optional[int] = field(
        default=0,
        metadata={"help": "Tensor‐parallel group size; if None, defaults to 1."}
    )
    cp_size: int = field(
        default=0,
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
        return self._setup_distributed()

    def _setup_distributed(self):
        """
        Initialize torch.distributed process group, infer dp/tp sizes,
        build device mesh, and register destroy handler.
        Requires env vars: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
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

    def parallelize(self, model):
        """
        Apply FSDP2 + TP sharding via the provided parallelize_fn.
        Must be called after setup_distributed().
        """
        if use_hf_tp_plan:
            tp_shard_plan = get_hf_tp_shard_plan(model)
        else:
            raise NotImplemented("todo")

        fsdp2_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            mp_policy=self.mp_policy,
            tp_shard_plan=tp_shard_plan,
            offload_policy=self.offload_policy,
        )
        return model
