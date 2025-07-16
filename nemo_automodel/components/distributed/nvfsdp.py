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
from typing import List, Optional

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)

from nemo_automodel.components.distributed.parallelizer import (
    get_hf_tp_shard_plan,
    nvfsdp_strategy_parallelize,
)


@dataclass
class NVFSDPManager:
    """
    Manager for setting up and parallelizing models using nvFSDP with TP, DP, CP sharding.

    This manager initializes the torch.distributed process group, infers the group sizes
    for data parallelism (DP) and tensor parallelism (TP), builds the device mesh for
    distributed operations, and applies parallelization to the model using a prescribed
    TP sharding plan. It also supports mixed precision and CPU offloading options.

    Attributes:
        dp_size (Optional[int]): Data-parallel group size. If None or non-positive, it is
            inferred from WORLD_SIZE.
        tp_size (Optional[int]): Tensor-parallel group size. Defaults to 1 if zero/None.
        cp_size (int): Context-parallel group size for pipeline-like sharding.
        sequence_parallel (bool): Enables sequence parallelism in the TP plan when True.
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
    tp_size: Optional[int] = field(
        default=1,
        metadata={"help": "Tensor-parallel group size; if None, defaults to 1."},
    )
    cp_size: Optional[int] = field(
        default=1,
        metadata={"help": "Context-parallel group size (for pipeline-like sharding)."},
    )
    sequence_parallel: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable sequence parallelism in TP plan if True. Not supported with nvFSDP right now."},
    )
    backend: Optional[str] = field(default="nccl", metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."})
    world_size: Optional[int] = field(
        default=None,
        # init=False,
        metadata={"help": "Total number of processes."},
    )
    nvfsdp_unit_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "transformers.models.llama.modeling_llama.LlamaDecoderLayer",
        ],
        metadata={"help": "List of unit modules to be wrapped with nvFSDP."},
    )

    # nvFSDP config
    data_parallel_sharding_strategy: Optional[str] = field(
        default="optim_grads_params",
        metadata={"help": "Data parallel sharding strategy."},
    )
    init_nvfsdp_with_meta_device: Optional[bool] = field(
        default=False, metadata={"help": "Initialize nvFSDP with meta device if True."}
    )
    grad_reduce_in_fp32: Optional[bool] = field(default=False, metadata={"help": "Reduce gradients in fp32 if True."})
    preserve_fp32_weights: Optional[bool] = field(default=False, metadata={"help": "Preserve fp32 weights if True."})
    overlap_grad_reduce: Optional[bool] = field(default=True, metadata={"help": "Overlap gradient reduction if True."})
    overlap_param_gather: Optional[bool] = field(
        default=True, metadata={"help": "Overlap parameter gathering if True."}
    )
    check_for_nan_in_grad: Optional[bool] = field(
        default=True, metadata={"help": "Check for NaN in gradients if True."}
    )
    average_in_collective: Optional[bool] = field(default=False, metadata={"help": "Average in collective if True."})
    disable_bucketing: Optional[bool] = field(default=False, metadata={"help": "Disable bucketing if True."})
    calculate_per_token_loss: Optional[bool] = field(
        default=False, metadata={"help": "Calculate per token loss if True."}
    )
    keep_fp8_transpose_cache_when_using_custom_fsdp: Optional[bool] = field(
        default=False, metadata={"help": "Keep fp8 transpose cache when using custom FSDP if True."}
    )
    nccl_ub: Optional[bool] = field(default=False, metadata={"help": "Use NCCL UBs if True."})
    fsdp_double_buffer: Optional[bool] = field(default=False, metadata={"help": "Use double buffer if True."})

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

        # infer if not provided
        self.dp_size = self.dp_size
        if self.dp_size is None or self.dp_size <= 0:
            self.dp_size = self.world_size
        self.tp_size = self.tp_size or 1

        mesh_shape = (self.dp_size, self.cp_size, self.tp_size)
        mesh_names = ("data_parallel", "context_parallel", "tensor_parallel")
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
            self.device_mesh[("data_parallel", "context_parallel")]._flatten(mesh_dim_name="dp_cp")
        return self

    def parallelize(self, model, optimizer=None, use_hf_tp_plan=False):
        """
        Parallelizes the given model using FSDP2 and TP sharding strategies.

        This method must be called after the distributed environment has been set up.
        It selects a TP sharding plan (currently supporting Hugging Face
        TP plan via get_hf_tp_shard_plan) and applies the FSDP2 parallelization strategy.

        Args:
            model: The model to be parallelized.
            optimizer: The optimizer for the model. If None, user need to call model.finish_grad_sync()
                before optimizer.step(), model.install_optimized_model_weights() and model.zero_grad_buffer()
                after optimizer.zero_grad()
            use_hf_tp_plan (bool): if true, will query the model for the TP plan.

        Returns:
            The parallelized model.

        Raises:
            NotImplemented: If the required TP sharding plan is not supported.
        """
        if self.data_parallel_sharding_strategy != "optim_grads_params":
            if self.device_mesh.get_rank() == 0:
                print(
                    "Warning: nvFSDP data_parallel_sharding_strategy is not optim_grads_params. "
                    "Parameters will not be sharded."
                )

        if self.device_mesh["tensor_parallel"].size() > 1:
            if use_hf_tp_plan:
                tp_shard_plan = get_hf_tp_shard_plan(model)
            else:
                # Parallelize the first embedding and the last linear out projection
                base_model_tp_plan = {
                    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                    "model.layers.*.mlp.up_proj": ColwiseParallel(),
                    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                    "model.layers.*.mlp.down_proj": RowwiseParallel(),
                }

                # TODO(boxiangw): investigate SP
                if self.sequence_parallel and self.device_mesh.get_rank() == 0:
                    # TODO(boxiangw): Change this to a log
                    print("Sequence parallelism is disabled. It is not compatible with nvFSDP.")

                tp_shard_plan = base_model_tp_plan
                # TODO(boxiangw): Change this to a log
                if self.device_mesh.get_rank() == 0:
                    print(
                        "Using default TP plan for parallelization. "
                        "It is compatible with huggingface llama3-style models."
                    )
        else:
            tp_shard_plan = None

        model = nvfsdp_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            optimizer=optimizer,
            nvfsdp_unit_modules=self.nvfsdp_unit_modules,
            tp_shard_plan=tp_shard_plan,
            data_parallel_sharding_strategy=self.data_parallel_sharding_strategy,
            init_nvfsdp_with_meta_device=self.init_nvfsdp_with_meta_device,
            grad_reduce_in_fp32=self.grad_reduce_in_fp32,
            preserve_fp32_weights=self.preserve_fp32_weights,
            overlap_grad_reduce=self.overlap_grad_reduce,
            overlap_param_gather=self.overlap_param_gather,
            check_for_nan_in_grad=self.check_for_nan_in_grad,
            average_in_collective=self.average_in_collective,
            disable_bucketing=self.disable_bucketing,
            calculate_per_token_loss=self.calculate_per_token_loss,
            keep_fp8_transpose_cache_when_using_custom_fsdp=self.keep_fp8_transpose_cache_when_using_custom_fsdp,
            nccl_ub=self.nccl_ub,
            fsdp_double_buffer=self.fsdp_double_buffer,
        )

        return model
