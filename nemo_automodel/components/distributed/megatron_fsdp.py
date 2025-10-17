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

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.parallelizer import (
    _get_parallel_plan,
    megatron_fsdp_strategy_parallelize,
)

logger = logging.getLogger(__name__)


@dataclass
class MegatronFSDPManager:
    """
    Manager for setting up and parallelizing models using MegatronFSDP with TP, DP, CP sharding.

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
        use_hf_tp_plan (bool): Use Hugging Face TP plan if True.
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
        metadata={"help": "Enable sequence parallelism in TP plan if True. Not supported with MegatronFSDP right now."},
    )
    use_hf_tp_plan: Optional[bool] = field(
        default=False,
        metadata={"help": "Use Hugging Face TP plan if True."},
    )
    backend: Optional[str] = field(default="nccl", metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."})
    world_size: Optional[int] = field(
        default=None,
        # init=False,
        metadata={"help": "Total number of processes."},
    )
    megatron_fsdp_unit_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "transformers.models.llama.modeling_llama.LlamaDecoderLayer",
        ],
        metadata={"help": "List of unit modules to be wrapped with MegatronFSDP."},
    )

    # MegatronFSDP config
    zero_dp_strategy: Optional[int] = field(
        default=3,
        metadata={"help": "Data parallel sharding strategy."},
    )
    init_fsdp_with_meta_device: Optional[bool] = field(
        default=False, metadata={"help": "Initialize MegatronFSDP with meta device if True."}
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
    keep_fp8_transpose_cache: Optional[bool] = field(
        default=False, metadata={"help": "Keep fp8 transpose cache when using custom FSDP if True."}
    )
    nccl_ub: Optional[bool] = field(default=False, metadata={"help": "Use NCCL UBs if True."})
    fsdp_double_buffer: Optional[bool] = field(default=False, metadata={"help": "Use double buffer if True."})

    # Gradient / Activation checkpointing
    activation_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable activation checkpointing for transformer MLP layers to save memory."},
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

        # infer if not provided
        self.tp_size = self.tp_size or 1
        self.cp_size = self.cp_size or 1

        if self.dp_size is None or self.dp_size <= 0:
            # Calculate dp_size to ensure dp_size * tp_size * cp_size == world_size
            total_parallel_ranks = self.tp_size * self.cp_size
            if self.world_size % total_parallel_ranks != 0:
                raise ValueError(
                    f"world_size ({self.world_size}) must be divisible by (tp_size * cp_size) "
                    f"({self.tp_size} * {self.cp_size} = {total_parallel_ranks})"
                )
            self.dp_size = self.world_size // total_parallel_ranks

        mesh_shape = (self.dp_size, self.cp_size, self.tp_size)
        mesh_names = ("dp", "cp", "tp")
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
        return self

    def parallelize(self, model, optimizer=None):
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

        Returns:
            The parallelized model.

        Raises:
            NotImplemented: If the required TP sharding plan is not supported.
        """
        if dist.get_world_size() == 1:
            logger.info("World size is 1, skipping parallelization.")
            model = model.to("cuda").to(torch.bfloat16)
            if self.activation_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                else:
                    logger.error("Model does not support gradient checkpointing. Skipping.")
            return model, optimizer

        if self.activation_checkpointing:
            logger.error("Activation checkpointing is not yet supported with MegatronFSDP. Skipping.")

        if self.zero_dp_strategy != 3:
            if self.device_mesh.get_rank() == 0:
                print("Warning: MegatronFSDP zero_dp_strategy is not 3. Parameters will not be sharded.")

        if self.device_mesh["tp"].size() > 1:
            # Delegate plan selection to central helper. MegatronFSDP currently does not support SP.
            tp_shard_plan = _get_parallel_plan(
                model,
                sequence_parallel=False,  # explicit: SP not supported here
                tp_shard_plan=None,
                use_hf_tp_plan=self.use_hf_tp_plan,
            )
        else:
            tp_shard_plan = None

        if self.cp_size > 1:
            dp_shard_dim = "dp_cp"
        else:
            dp_shard_dim = "dp"
        tp_dim = "tp"

        model = megatron_fsdp_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            optimizer=optimizer,
            megatron_fsdp_unit_modules=self.megatron_fsdp_unit_modules,
            tp_shard_plan=tp_shard_plan,
            zero_dp_strategy=self.zero_dp_strategy,
            init_fsdp_with_meta_device=self.init_fsdp_with_meta_device,
            grad_reduce_in_fp32=self.grad_reduce_in_fp32,
            preserve_fp32_weights=self.preserve_fp32_weights,
            overlap_grad_reduce=self.overlap_grad_reduce,
            overlap_param_gather=self.overlap_param_gather,
            check_for_nan_in_grad=self.check_for_nan_in_grad,
            average_in_collective=self.average_in_collective,
            disable_bucketing=self.disable_bucketing,
            calculate_per_token_loss=self.calculate_per_token_loss,
            keep_fp8_transpose_cache=self.keep_fp8_transpose_cache,
            nccl_ub=self.nccl_ub,
            fsdp_double_buffer=self.fsdp_double_buffer,
            dp_shard_dim=dp_shard_dim,
            tp_dim=tp_dim,
        )

        return model
