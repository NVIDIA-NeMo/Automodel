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

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.parallelizer import (
    fsdp2_strategy_parallelize,
    get_hf_tp_shard_plan,
)


@dataclass
class FSDP2Manager:
    """
    Manager for setting up and parallelizing models using FSDP2 with TP, DP, CP sharding.

    This manager initializes the torch.distributed process group, infers the group sizes
    for data parallelism (DP) and tensor parallelism (TP), builds the device mesh for
    distributed operations, and applies parallelization to the model using a prescribed
    TP sharding plan. It also supports mixed precision and CPU offloading options.

    Attributes:
        dp_size (Optional[int]): Data-parallel group size. If None or non-positive, it is
            inferred from WORLD_SIZE.
        dp_replicate_size (Optional[int]): Data-parallel replicate group size. If None or non-positive, it is
            inferred from dp_size. Must be a divisor of dp_size.
        tp_size (Optional[int]): Tensor-parallel group size. Defaults to 1 if zero/None.
        cp_size (int): Context-parallel group size for pipeline-like sharding.
        sequence_parallel (bool): Enables sequence parallelism in the TP plan when True.
        mp_policy (MixedPrecisionPolicy): Defines the mixed precision policy for parameters,
            reductions, and outputs.
        offload_policy (CPUOffloadPolicy): Policy to offload parameters or optimizer states
            to CPU, if specified.
        backend (str): Distributed backend to use (e.g., 'nccl' for GPUs or 'gloo' for CPUs).
        world_size (int): Total number of processes.

    Methods:
        __post_init__():
            Automatically sets up the distributed environment after initialization.
        _setup_distributed():
            Initializes the torch.distributed process group, infers parallel sizes,
            builds the device mesh, and registers a destroy handler.
        parallelize(model):
            Applies FSDP2 and Tensor-Parallel sharding strategies to the given model.
    """

    dp_size: Optional[int] = field(
        default=None,
        metadata={"help": "Data-parallel group size; if None, infer from WORLD_SIZE."},
    )
    dp_replicate_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Data-parallel replicate group size; if None, infer from dp_size. Must be a divisor of dp_size."
        },
    )
    tp_size: Optional[int] = field(
        default=1,
        metadata={"help": "Tensor-parallel group size; if None, defaults to 1."},
    )
    cp_size: Optional[int] = field(
        default=1,
        metadata={"help": "Context-parallel group size (for pipeline-like sharding)."},
    )
    pp_size: Optional[int] = field(
        default=1,
        metadata={"help": "Pipeline-parallel group size (for pipeline-like sharding)."},
    )
    sequence_parallel: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable sequence parallelism in TP plan if True."},
    )
    mp_policy: Optional[MixedPrecisionPolicy] = field(
        default=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        metadata={"help": "MixedPrecisionPolicy for FSDP2 (param/reduce/output dtypes)."},
    )
    offload_policy: Optional[CPUOffloadPolicy] = field(
        default=None,
        metadata={"help": "CPUOffloadPolicy to offload parameters/optim states to CPU."},
    )
    backend: Optional[str] = field(default="nccl", metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."})
    world_size: Optional[int] = field(
        default=None,
        # init=False,
        metadata={"help": "Total number of processes."},
    )

    def __post_init__(self):
        """
        Post-initialization hook that sets up the distributed environment.
        """
        return self._setup_distributed()

    def _setup_distributed(self):
        """
        Initializes the distributed environment.

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

        if self.tp_size is None or self.tp_size <= 0:
            self.tp_size = 1

        if self.cp_size is None or self.cp_size <= 0:
            self.cp_size = 1

        if self.pp_size is None or self.pp_size <= 0:
            self.pp_size = 1

        # infer if not provided
        if self.dp_size is None or self.dp_size <= 0:
            # Calculate dp_size to ensure dp_size * tp_size * cp_size == world_size
            total_parallel_ranks = self.tp_size * self.cp_size * self.pp_size
            if self.world_size % total_parallel_ranks != 0:
                raise ValueError(
                    f"world_size ({self.world_size}) must be divisible by (tp_size * cp_size) "
                    f"({self.tp_size} * {self.cp_size} = {total_parallel_ranks})"
                )
            self.dp_size = self.world_size // total_parallel_ranks

        if self.dp_replicate_size is None or self.dp_replicate_size <= 0:
            self.dp_replicate_size = 1

        # HSDP usecase
        # dp_size = dp_replicate_size * dp_shard_size
        # dp_shard_size < dp_size since ddp usecase is not supported by FSDP2, need to use DDPManager instead
        # TODO(boxiangw): Call DDPManager instead of FSDP2Manager for ddp usecase?
        assert self.dp_size % self.dp_replicate_size == 0, "dp_size must be a multiple of dp_replicate_size"
        assert self.dp_replicate_size < self.dp_size or self.dp_replicate_size == 1, (
            "dp_replicate_size must be less than dp_size since ddp usecase is not supported by FSDP2"
        )

        self.dp_shard_size = self.dp_size // self.dp_replicate_size

        self.device_mesh = self._get_device_mesh()

        return self

    def _get_device_mesh(self):
        mesh_shape = (self.pp_size, self.dp_replicate_size, self.dp_shard_size, self.cp_size, self.tp_size)
        mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
        for shape, name in zip(mesh_shape, mesh_names):
            assert isinstance(shape, int), "Expected {} to be an int, but got {}".format(name, type(shape))
            assert shape > 0, "Expected {} > 0, {}".format(name, shape)

        # build mesh [dp, cp, tp]
        self.device_mesh = init_device_mesh(
            device_type="cuda" if self.backend == "nccl" else "cpu",
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_names,
        )
        # flatten dp+cp if cp>1
        if self.cp_size > 1:
            self.device_mesh[("dp", "cp")]._flatten(mesh_dim_name="dp_cp")

        # based on https://github.com/pytorch/torchtitan/blob/d282cf2ce9ca8049b4b8423c1d7578c80426576f/torchtitan/distributed/parallel_dims.py#L191
        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []

        # for dp_replicate:
        dp_mesh_dim_names.append("dp_replicate")
        dp_cp_mesh_dim_names.append("dp_replicate")
        # for dp_shard:
        dp_mesh_dim_names.append("dp_shard")
        dp_shard_cp_mesh_dim_names.append("dp_shard")
        dp_cp_mesh_dim_names.append("dp_shard")
        # for cp:
        dp_shard_cp_mesh_dim_names.append("cp")
        dp_cp_mesh_dim_names.append("cp")

        # submesh for dp
        self.device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        # submesh for dp_shard_cp
        self.device_mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        # submesh for dp_cp
        self.device_mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        return self.device_mesh

    def parallelize(self, model, use_hf_tp_plan=False):
        """
        Parallelizes the given model using FSDP2 and TP sharding strategies.

        This method must be called after the distributed environment has been set up.
        It selects a TP sharding plan (currently supporting Hugging Face
        TP plan via get_hf_tp_shard_plan) and applies the FSDP2 parallelization strategy.

        Args:
            model (nn.Module): The model to be parallelized.
            use_hf_tp_plan (bool): if true, will attempt to get the TP plan from the model.

        Returns:
            The parallelized model.

        Raises:
            NotImplemented: If the required TP sharding plan is not supported.
        """
        if self.device_mesh["tp"].size() > 1:
            if use_hf_tp_plan:
                tp_shard_plan = get_hf_tp_shard_plan(model)
            else:
                # Parallelize the first embedding and the last linear out projection
                base_model_tp_plan = {
                    "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
                    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                    "model.layers.*.mlp.up_proj": ColwiseParallel(),
                    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                    "model.layers.*.mlp.down_proj": RowwiseParallel(),
                    "lm_head": ColwiseParallel(output_layouts=Replicate()),
                }

                base_model_sp_plan = {
                    "model.embed_tokens": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                    "model.norm": SequenceParallel(),
                    "model.layers.*.input_layernorm": SequenceParallel(),
                    "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "model.layers.*.post_attention_layernorm": SequenceParallel(),
                    "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
                }

                if self.sequence_parallel:
                    # Enable sequence parallelism only if TP size > 1
                    base_model_tp_plan.update(base_model_sp_plan)

                tp_shard_plan = base_model_tp_plan

                # TODO(boxiangw): Change this to a log
                if self.device_mesh.get_rank() == 0:
                    print(
                        "Using default TP plan for parallelization. "
                        "It is compatible with huggingface llama3-style models."
                    )
        else:
            tp_shard_plan = None

        fsdp2_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            mp_policy=self.mp_policy,
            tp_shard_plan=tp_shard_plan,
            offload_policy=self.offload_policy,
        )
        return model
