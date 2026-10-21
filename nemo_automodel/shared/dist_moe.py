# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Typed configuration for the optional fused Dist-MoE expert backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from dist_moe import Context


@dataclass(kw_only=True)
class DistMoEConfig:
    """Configure a shared, sequential Dist-MoE context for one model.

    The initial integration supports plain SwiGLU, BF16 or MXFP8 computation,
    and one forward/backward schedule at a time. Pipeline parallelism requires
    an activation-slot assignment and is not supported by this integration.
    """

    max_tokens_per_rank: int | None = None
    precision: Literal["bf16", "mxfp8"] = "bf16"
    pipeline: Literal["staged", "mega"] = "staged"
    activation_slot_capacity_factor: float | None = None
    device_scratch_capacity_factor: float = 8.0

    def __post_init__(self) -> None:
        """Validate capacities without importing the optional CUDA package."""
        if self.max_tokens_per_rank is not None and (
            isinstance(self.max_tokens_per_rank, bool)
            or not isinstance(self.max_tokens_per_rank, int)
            or self.max_tokens_per_rank <= 0
        ):
            raise ValueError("dist_moe.max_tokens_per_rank must be a positive integer")
        if self.precision not in ("bf16", "mxfp8"):
            raise ValueError("dist_moe.precision must be bf16 or mxfp8")
        if self.pipeline not in ("staged", "mega"):
            raise ValueError("dist_moe.pipeline must be staged or mega")
        for name, value in (
            ("device_scratch_capacity_factor", self.device_scratch_capacity_factor),
            ("activation_slot_capacity_factor", self.activation_slot_capacity_factor),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"dist_moe.{name} must be finite and nonnegative")
        if self.device_scratch_capacity_factor <= 0:
            raise ValueError("dist_moe.device_scratch_capacity_factor must be positive")

    def build(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        top_k: int,
        num_layers: int,
    ) -> Context:
        """Build the context collectively after selecting each rank's GPU.

        Args:
            group: Expert-parallel group with equal contiguous expert partitions.
            device: Selected local CUDA device, with SM100 or newer architecture.
            hidden_dim: Expert input/output width.
            intermediate_dim: Expert hidden width.
            num_experts: Global expert count, divisible by the group size.
            top_k: Route slots per physical token row.
            num_layers: Local layers sharing one activation stack and scratch area.

        Returns:
            Context shared sequentially across local expert modules. Its owner
            must call close after all pending forward/backward work finishes.
        """
        available, package = safe_import("dist_moe", msg="Install the optional dist-moe package for this backend.")
        if not available:
            raise ImportError("dispatcher='dist_moe' requires the dist-moe package")
        if self.max_tokens_per_rank is None:
            raise ValueError("dist_moe.max_tokens_per_rank is required")
        if torch.cuda.get_device_capability(device)[0] < 10:
            raise ValueError("Dist-MoE requires SM100 or newer; H100 is unsupported")
        policy = None
        if self.precision == "mxfp8":
            policy = package.BlockScaledConfig(format=package.BlockScaledFormat.MXFP8_E4M3, pipeline=self.pipeline)
        config = package.Config(
            max_num_local_input_tokens=self.max_tokens_per_rank,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            max_moe_layers_per_activation_slot=num_layers,
            activation_slot_capacity_factor=self.activation_slot_capacity_factor,
            device_scratch_capacity_factor=self.device_scratch_capacity_factor,
            block_scaled=policy,
            wgrad_dtype=torch.bfloat16,
        )
        return package.create_context(group=group, config=config, device=device)
