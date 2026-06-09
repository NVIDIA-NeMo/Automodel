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

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from nemo_automodel.components.distributed.config import DDPConfig
from nemo_automodel.components.distributed.parallelizer import _extract_model_layers

logger = logging.getLogger(__name__)


class DDPManager:
    """
    Manager for distributed training using PyTorch's DDP.

    This manager wraps models with DistributedDataParallel for data-parallel
    distributed training.

    Args:
        config (DDPConfig): Configuration for DDP distributed training.

    Example:
        from nemo_automodel.components.distributed.config import DDPConfig

        config = DDPConfig(activation_checkpointing=True)
        manager = DDPManager(config)
        model = manager.parallelize(model)
    """

    def __init__(self, config: DDPConfig):
        self.config = config

        # Extract config fields for easy access
        self.activation_checkpointing = config.activation_checkpointing
        self.broadcast_buffers = config.broadcast_buffers
        self.find_unused_parameters = config.find_unused_parameters
        self.static_graph = config.static_graph
        self.bucket_cap_mb = config.bucket_cap_mb
        self.gradient_as_bucket_view = config.gradient_as_bucket_view

        # Setup distributed environment
        self._setup_distributed()

    def _setup_distributed(self):
        """
        Initialize device configuration for DDP.

        Sets the rank, world_size, and device based on the process group backend.
        """
        if not dist.is_available():
            raise RuntimeError("torch.distributed not available")

        if not dist.is_initialized():
            raise RuntimeError("expected torch.distributed to be initialized")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        backend = str(dist.get_backend()).lower()
        if "nccl" in backend and torch.cuda.is_available():
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
            model = model.to(self.device)
            if self.device.type == "cuda":
                model = model.to(torch.bfloat16)
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

        ddp_kwargs = {
            "device_ids": [self.device] if self.device.type == "cuda" else None,
            "broadcast_buffers": self.broadcast_buffers,
            "find_unused_parameters": self.find_unused_parameters,
            "static_graph": self.static_graph,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
        }
        if self.bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = self.bucket_cap_mb

        return DDP(model.to(self.device), **ddp_kwargs)
