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
import warnings
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nemo_automodel.components.distributed.activation_checkpointing import (
    apply_submodule_checkpointing,
    detect_kv_sharing_and_maybe_disable_cache,
    is_selective_activation_checkpointing,
)
from nemo_automodel.components.distributed.config import DDPConfig
from nemo_automodel.components.distributed.parallelizer import (
    _extract_model_layer_groups,
    _filter_layer_groups_for_activation_checkpointing,
    _should_use_hf_native_gradient_checkpointing,
    apply_selective_activation_checkpointing,
)

logger = logging.getLogger(__name__)


def _resolve_ddp_device() -> torch.device:
    """Validate distributed state and return the device owned by this rank."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        raise RuntimeError("expected torch.distributed to be initialized")

    rank = dist.get_rank()
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend and torch.cuda.is_available():
        local_gpu = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_gpu)
        return torch.device("cuda", index=local_gpu)
    return torch.device("cpu")


def parallelize_ddp(
    model: torch.nn.Module,
    config: DDPConfig,
    *,
    device: torch.device | None = None,
    reapply_trainability: Callable[[torch.nn.Module], None] | None = None,
) -> torch.nn.Module:
    """Apply activation checkpointing and PyTorch DDP from a typed config."""
    if device is None:
        device = _resolve_ddp_device()

    if dist.get_world_size() == 1:
        logger.info("World size is 1, skipping parallelization.")
        model = model.to(device)
        if device.type == "cuda":
            model = model.to(torch.bfloat16)
        if config.activation_checkpointing:
            if is_selective_activation_checkpointing(config.activation_checkpointing):
                apply_selective_activation_checkpointing(
                    model,
                    activation_checkpointing_scope=config.activation_checkpointing_scope,
                )
            else:
                layer_groups = _extract_model_layer_groups(model)
                layers, ac_scopes = _filter_layer_groups_for_activation_checkpointing(
                    layer_groups,
                    config.activation_checkpointing_scope,
                )
                if _should_use_hf_native_gradient_checkpointing(model, layer_groups, ac_scopes):
                    model.gradient_checkpointing_enable()
                else:
                    apply_submodule_checkpointing(layers, detect_kv_sharing_and_maybe_disable_cache(model))
        if reapply_trainability is not None:
            reapply_trainability(model)
        return model

    if config.activation_checkpointing:
        has_kv_sharing = detect_kv_sharing_and_maybe_disable_cache(model)

        if is_selective_activation_checkpointing(config.activation_checkpointing):
            apply_selective_activation_checkpointing(
                model,
                activation_checkpointing_scope=config.activation_checkpointing_scope,
            )
        else:
            layer_groups = _extract_model_layer_groups(model)
            layers, _ = _filter_layer_groups_for_activation_checkpointing(
                layer_groups,
                config.activation_checkpointing_scope,
            )
            apply_submodule_checkpointing(layers, has_kv_sharing)

    ddp_kwargs = {
        "device_ids": [device] if device.type == "cuda" else None,
        "broadcast_buffers": config.broadcast_buffers,
        "find_unused_parameters": config.find_unused_parameters,
        "static_graph": config.static_graph,
        "gradient_as_bucket_view": config.gradient_as_bucket_view,
    }
    if config.bucket_cap_mb is not None:
        ddp_kwargs["bucket_cap_mb"] = config.bucket_cap_mb

    model = model.to(device)
    if reapply_trainability is not None:
        reapply_trainability(model)
    return DDP(model, **ddp_kwargs)


class DDPManager:
    """Deprecated compatibility wrapper for PyTorch DDP.

    .. deprecated:: 0.7
        Pass :class:`DDPConfig` through the config-driven infrastructure and
        provide model-specific behavior with :class:`ModelParallelizer`. This
        compatibility class is scheduled for removal in 0.8.

    This manager wraps models with DistributedDataParallel for data-parallel
    distributed training.

    Args:
        config (DDPConfig): Configuration for DDP distributed training.

    """

    def __init__(self, config: DDPConfig):
        warnings.warn(
            "DDPManager is deprecated and will be removed in 0.8; pass DDPConfig through the "
            "config-driven infrastructure and use ModelParallelizer for model-owned behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self._setup_distributed()

    def __getattr__(self, name):
        return getattr(self.config, name)

    def _setup_distributed(self):
        """
        Initialize device configuration for DDP.

        Sets the rank, world_size, and device based on the process group backend.
        """
        self.device = _resolve_ddp_device()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def parallelize(
        self,
        model: torch.nn.Module,
        reapply_trainability: Callable[[torch.nn.Module], None] | None = None,
    ) -> torch.nn.Module:
        """
        Wraps the given model with DistributedDataParallel (DDP).

        Moves the model to the initialized device before wrapping. For CUDA devices,
        the device id is passed to DDP as device_ids; for CPU, no device ids are provided.

        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.
            reapply_trainability: Optional callback that re-resolves parameter
                trainability after model surgery and before DDP construction.

        Returns:
            torch.nn.parallel.DistributedDataParallel: The DDP-wrapped model.
        """
        return parallelize_ddp(
            model,
            self.config,
            device=self.device,
            reapply_trainability=reapply_trainability,
        )
