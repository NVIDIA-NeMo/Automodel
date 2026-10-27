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

"""Dist-MoE execution with AutoModel's canonical expert parameter ownership."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from dist_moe import Context


class GroupedExpertsDistMoE(GroupedExperts):
    """Execute fused dispatch, expert computation, and combine.

    Parameters retain AutoModel's global layouts [experts, hidden, 2*intermediate]
    and [experts, intermediate, hidden]. EP shards dimension zero; FSDP may shard
    dimension one. Forward converts the unsharded local weights to the native
    Dist-MoE layout, preserving autograd and checkpoint names. Those conversions
    are part of end-to-end timing.
    """

    def __init__(self, config: MoEConfig, backend: BackendConfig) -> None:
        """Validate supported expert semantics and allocate canonical parameters."""
        if config.expert_activation != "swiglu":
            raise ValueError("Dist-MoE integration requires SwiGLU")
        if config.swiglu_clamp_after_activation:
            raise ValueError("Dist-MoE does not support clamping after the activation")
        if not math.isfinite(config.swiglu_limit) or config.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be finite and nonnegative")
        if config.expert_bias:
            raise ValueError("Dist-MoE does not support biased experts")
        super().__init__(config, backend)
        self.dist_moe_config = backend.dist_moe
        self.ep_mesh: DeviceMesh | None = None
        self.context: Context | None = None

    def init_token_dispatcher(self, ep_mesh: DeviceMesh) -> None:
        """Attach the EP mesh; the model creates one shared context after sharding."""
        self.ep_mesh = ep_mesh

    def initialize_shared_context(self, modules: list[GroupedExpertsDistMoE]) -> None:
        """Create one scratch/activation owner for homogeneous local expert layers."""
        if self.ep_mesh is None:
            raise RuntimeError("Dist-MoE requires expert-parallel initialization")
        shape = (
            self.config.expert_dim,
            self.config.moe_inter_dim,
            self.config.n_routed_experts,
            self.config.n_activated_experts,
        )
        for module in modules:
            other = (
                module.config.expert_dim,
                module.config.moe_inter_dim,
                module.config.n_routed_experts,
                module.config.n_activated_experts,
            )
            if (
                other != shape
                or module.dist_moe_config != self.dist_moe_config
                or module.config.swiglu_limit != self.config.swiglu_limit
            ):
                raise ValueError("One Dist-MoE context requires homogeneous local expert shapes and configuration")
            if module.context is not None:
                raise RuntimeError("Dist-MoE context is already initialized")
        context = self.dist_moe_config.build(
            group=self.ep_mesh.get_group(),
            device=torch.device("cuda", torch.cuda.current_device()),
            hidden_dim=shape[0],
            intermediate_dim=shape[1],
            num_experts=shape[2],
            top_k=shape[3],
            num_layers=len(modules),
            swiglu_limit=self.config.swiglu_limit,
        )
        for module in modules:
            module.context = context

    def close(self) -> None:
        """Release the shared context after every layer's backward has finished."""
        if self.context is not None:
            self.context.close()
            self.context = None

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute routed experts on equal padded physical extents across EP ranks.

        Args:
            x: BF16 CUDA tensor of shape [local_tokens, hidden].
            token_mask: Boolean CUDA tensor of shape [local_tokens]; true rows are valid.
            weights: CUDA routing scores [local_tokens, top_k], differentiable.
            indices: Integer CUDA global expert IDs [local_tokens, top_k].

        Returns:
            BF16 CUDA tensor [local_tokens, hidden], with zero inactive rows.
            Padding, weight casts, and layout conversions preserve autograd.
        """
        if self.context is None:
            raise RuntimeError("Dist-MoE context must be initialized before forward")
        capacity = self.dist_moe_config.max_tokens_per_rank
        if capacity is None or x.size(0) > capacity:
            raise ValueError("Input exceeds dist_moe.max_tokens_per_rank")
        if x.dtype != torch.bfloat16:
            raise ValueError("Dist-MoE requires BF16 expert inputs")
        available, package = safe_import("dist_moe", msg="Install dist-moe for this backend.")
        if not available:
            raise ImportError("Dist-MoE is unavailable")
        masked = ~token_mask.unsqueeze(-1)
        ids = indices.masked_fill(masked, -1)
        scores = weights.float().masked_fill(masked, 0.0)
        padding = capacity - x.size(0)
        inputs = F.pad(x, (0, 0, 0, padding)).contiguous()
        ids = F.pad(ids, (0, 0, 0, padding), value=-1).contiguous()
        scores = F.pad(scores, (0, 0, 0, padding)).contiguous()
        w13 = self.gate_and_up_projs
        w2 = self.down_projs
        if isinstance(w13, DTensor):
            w13 = w13.to_local()
        if isinstance(w2, DTensor):
            w2 = w2.to_local()
        w13 = w13.to(torch.bfloat16).transpose(1, 2).contiguous()
        w2 = w2.to(torch.bfloat16).transpose(1, 2).contiguous()
        output = package.routed_experts(inputs, ids, scores, w13, w2, self.context)
        return output[: x.size(0)]
