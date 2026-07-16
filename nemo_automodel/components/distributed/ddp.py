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


class _DDPFusedGradNormState:
    """Accumulate the norm of DDP-reduced gradient buckets."""

    def __init__(self, process_group, world_size: int, device: torch.device):
        self.process_group = process_group
        self.world_size = world_size
        self.device = device
        self.norm_sq = torch.zeros((), dtype=torch.float32, device=device)
        self.events = []

    def reset(self) -> None:
        self.norm_sq.zero_()
        self.events.clear()

    def wait(self) -> None:
        if not self.events:
            return
        current_stream = torch.cuda.current_stream(device=self.device)
        for event in self.events:
            current_stream.wait_event(event)
        self.events.clear()


def _ddp_fused_grad_norm_hook(state: _DDPFusedGradNormState, bucket):
    """Average a DDP bucket and accumulate its post-reduction squared norm."""
    buffer = bucket.buffer()
    buffer.div_(state.world_size)
    work = dist.all_reduce(buffer, group=state.process_group, async_op=True)

    def _accumulate_norm(future):
        reduced = future.value()[0]
        stream = torch.cuda.current_stream(device=reduced.device)
        with torch.cuda.stream(stream):
            state.norm_sq.add_(reduced.float().square().sum())
            event = torch.cuda.Event()
            event.record(stream)
        state.events.append(event)
        return reduced

    return work.get_future().then(_accumulate_norm)


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
        self.activation_checkpointing_scope = config.activation_checkpointing_scope
        self.broadcast_buffers = config.broadcast_buffers
        self.find_unused_parameters = config.find_unused_parameters
        self.static_graph = config.static_graph
        self.bucket_cap_mb = config.bucket_cap_mb
        self.gradient_as_bucket_view = config.gradient_as_bucket_view
        self.fused_grad_norm = config.fused_grad_norm

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
                if is_selective_activation_checkpointing(self.activation_checkpointing):
                    apply_selective_activation_checkpointing(
                        model,
                        activation_checkpointing_scope=self.activation_checkpointing_scope,
                    )
                else:
                    layer_groups = _extract_model_layer_groups(model)
                    layers, ac_scopes = _filter_layer_groups_for_activation_checkpointing(
                        layer_groups,
                        self.activation_checkpointing_scope,
                    )
                    if _should_use_hf_native_gradient_checkpointing(model, layer_groups, ac_scopes):
                        model.gradient_checkpointing_enable()
                    else:
                        apply_submodule_checkpointing(layers, detect_kv_sharing_and_maybe_disable_cache(model))
            return model

        if self.activation_checkpointing:
            has_kv_sharing = detect_kv_sharing_and_maybe_disable_cache(model)

            if is_selective_activation_checkpointing(self.activation_checkpointing):
                apply_selective_activation_checkpointing(
                    model,
                    activation_checkpointing_scope=self.activation_checkpointing_scope,
                )
            else:
                layer_groups = _extract_model_layer_groups(model)
                layers, _ = _filter_layer_groups_for_activation_checkpointing(
                    layer_groups,
                    self.activation_checkpointing_scope,
                )
                apply_submodule_checkpointing(layers, has_kv_sharing)

        ddp_kwargs = {
            "device_ids": [self.device] if self.device.type == "cuda" else None,
            "broadcast_buffers": self.broadcast_buffers,
            "find_unused_parameters": self.find_unused_parameters,
            "static_graph": self.static_graph,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
        }
        if self.bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = self.bucket_cap_mb

        ddp_model = DDP(model.to(self.device), **ddp_kwargs)
        if self.fused_grad_norm:
            if self.device.type != "cuda":
                raise ValueError("distributed.fused_grad_norm requires CUDA DDP")
            norm_state = _DDPFusedGradNormState(
                ddp_model.process_group,
                ddp_model.process_group.size() if ddp_model.process_group is not None else dist.get_world_size(),
                self.device,
            )
            ddp_model.register_comm_hook(norm_state, _ddp_fused_grad_norm_hook)
            ddp_model._nemo_fused_grad_norm_state = norm_state
            logger.info("Enabled opt-in fused DDP gradient norm")
        return ddp_model
