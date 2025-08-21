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
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.parallel_dims import DimNames, ParallelDims
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
        parallel_dims (ParallelDims): Parallel dimensions for the model.
        sequence_parallel (bool): Enable sequence parallelism in TP plan if True.
        mp_policy (MixedPrecisionPolicy): Defines the mixed precision policy for parameters,
            reductions, and outputs.
        offload_policy (CPUOffloadPolicy): Policy to offload parameters or optimizer states
            to CPU, if specified.

    Methods:
        __post_init__():
            Automatically sets up the distributed environment after initialization.
        _setup_distributed():
            Initializes the torch.distributed process group, infers parallel sizes,
            builds the device mesh, and registers a destroy handler.
        parallelize(model):
            Applies FSDP2 and Tensor-Parallel sharding strategies to the given model.
    """

    parallel_dims: ParallelDims = field(default_factory=ParallelDims)

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

    def __post_init__(self):
        """
        Post-initialization hook that sets up the distributed environment.
        """
        self.device_mesh = self.parallel_dims.mesh
        assert self.device_mesh is not None, "Device mesh is not initialized1"

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
        if self.device_mesh is None:
            raise ValueError("Device mesh is not initialized")

        if DimNames.TP in self.device_mesh.mesh_dim_names and self.device_mesh[DimNames.TP].size() > 1:
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
