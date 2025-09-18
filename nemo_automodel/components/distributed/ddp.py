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

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from nemo_automodel.components.distributed.parallelizer import _extract_model_layers

logger = logging.getLogger(__name__)


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

    world_size: int = field(default_factory=lambda: int, metadata={"help": "Total number of distributed processes."})

    # This is populated in setup_distributed(), not by user:
    rank: int = field(init=False, default_factory=lambda: int, metadata={"help": "Global rank of this process."})

    activation_checkpointing: bool = field(default=False, metadata={"help": "Enable activation checkpointing if True."})

    def __post_init__(self):
        """
        Post-initialization hook that sets up the distributed environment.
        """
        return self._setup_distributed()

    def _setup_distributed(self):
        """
        Initialize the torch.distributed process group and set up device configuration.

        The method sets the `rank` and `world_size` of the DDPManager,
        configures the device (GPU for 'nccl' backend, CPU otherwise), and initializes the process group.
        """
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")

        if not dist.is_initialized():
            raise RuntimeError("expected torch.distributed to be initialized")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Pin GPU if using NCCL
        if self.backend == "nccl":
            local_gpu = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_gpu)
            self.device = torch.device("cuda", index=local_gpu)
        else:
            self.device = torch.device("cpu")

    def parallelize(self, model):
        """
        Wraps the given model with DistributedDataParallel (DDP).

        Moves the model to the initialized device before wrapping. For CUDA devices,
        the device id is passed to DDP as device_ids; for CPU, no device ids are provided.

        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.

        Returns:
            torch.nn.parallel.DistributedDataParallel: The DDP-wrapped model.
        """
        if dist.get_world_size() == 1:
            logger.info("World size is 1, skipping parallelization.")
            model = model.to("cuda").to(torch.bfloat16)
            if self.activation_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                else:
                    logger.error("Model does not support gradient checkpointing. Skipping.")
            return model

        if self.activation_checkpointing:
            # Disable KV caching during training to ensure deterministic
            # shapes between forward and checkpoint recomputation.
            if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not False:
                try:
                    model.config.use_cache = False
                except Exception:
                    pass

            layers = _extract_model_layers(model)
            for i, layer in enumerate(layers):
                if hasattr(layer, "mlp"):
                    layers[i].mlp = checkpoint_wrapper(layer.mlp)
                if hasattr(layer, "self_attn"):
                    layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)

                if hasattr(layer, "input_layernorm"):
                    layers[i].input_layernorm = checkpoint_wrapper(layers[i].input_layernorm)

                if hasattr(layer, "post_attention_layernorm"):
                    layers[i].post_attention_layernorm = checkpoint_wrapper(layers[i].post_attention_layernorm)

        return DDP(model.to(self.device), device_ids=[self.device] if self.device.type == "cuda" else None)
