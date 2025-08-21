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

import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nemo_automodel.components.distributed.parallel_dims import ParallelDims


@dataclass
class DDPManager:
    """
    Manages setting up distributed training using PyTorch's DDP.

    Attributes:
        backend (str): The distributed backend to use (e.g. "nccl" or "gloo"). Defaults to "nccl".
        rank (int): Global rank of this process. This is set during distributed setup.
        world_size (int): Total number of processes in the distributed group. Set at distributed setup.
    """

    backend: str = field(default="nccl", metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."})

    parallel_dims: ParallelDims = field(default_factory=ParallelDims)

    def setup_distributed(self):
        """
        Initialize the torch.distributed process group and set up device configuration.

        This method requires the following environment variables to be set:
            - RANK: Global rank of the process.
            - WORLD_SIZE: Total number of processes.
            - MASTER_ADDR: Address of the master node.
            - MASTER_PORT: Port on which the master node is listening.

        The method sets the `rank` and `world_size` of the DDPManager,
        configures the device (GPU for 'nccl' backend, CPU otherwise), and initializes the process group.
        """
        if not dist.is_initialized():
            rank = int(os.environ["RANK"])
            world = int(os.environ["WORLD_SIZE"])
            os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
            os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
            dist.init_process_group(self.backend, rank=rank, world_size=world)

        self.rank = self.parallel_dims.rank
        self.world_size = self.parallel_dims.world_size

        # Pin GPU if using NCCL
        if self.backend == "nccl":
            local_gpu = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_gpu)
            self.device = torch.device("cuda", index=local_gpu)
        else:
            self.device = torch.device("cpu")

    def wrap_model(self, model):
        """
        Wraps the given model with DistributedDataParallel (DDP).

        Moves the model to the initialized device before wrapping. For CUDA devices,
        the device id is passed to DDP as device_ids; for CPU, no device ids are provided.

        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.

        Returns:
            torch.nn.parallel.DistributedDataParallel: The DDP-wrapped model.
        """
        return DDP(model.to(self.device), device_ids=[self.device] if self.device.type == "cuda" else None)

    # @contextmanager
    # def no_sync(self):
    #     """
    #     Context manager to temporarily disable gradient synchronization during backpropagation.
    #
    #     This can be used for gradient accumulation:
    #         with manager.no_sync():
    #             loss.backward()
    #
    #     When used within a DDP-wrapped model, it skips the gradient all‐reduce.
    #     """
    #     if isinstance(self.model, DDP):
    #         with self.model.no_sync():
    #             yield
    #     else:
    #         yield
