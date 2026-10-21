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

"""Dist-MoE execution with native-layout parameters and canonical checkpoints."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from dist_moe import Context


class GroupedExpertsDistMoE(torch.nn.Module):
    """Execute fused dispatch, expert computation, and combine.

    Optimizer-owned parameters use contiguous [experts, 2*intermediate, hidden]
    and [experts, hidden, intermediate] storage. EP shards dimension zero; FSDP
    may shard dimension one and gathers before execution. Canonical checkpoint
    tensors are transposed views of these parameters, with no persistent cache
    or per-forward layout copy.
    """

    def __init__(self, config: MoEConfig, backend: BackendConfig) -> None:
        """Validate expert semantics and allocate native-layout trainable parameters."""
        if config.expert_activation != "swiglu":
            raise ValueError("Dist-MoE integration requires SwiGLU")
        if config.swiglu_clamp_after_activation:
            raise ValueError("Dist-MoE does not support clamping after the activation")
        if not math.isfinite(config.swiglu_limit) or config.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be finite and nonnegative")
        if config.expert_bias:
            raise ValueError("Dist-MoE does not support biased experts")
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.expert_bias = False
        self.gate_up_proj_bias = None
        self.down_proj_bias = None
        self.routed_gate_up_weights = torch.nn.Parameter(
            torch.empty(config.n_routed_experts, 2 * config.moe_inter_dim, config.expert_dim, dtype=config.dtype)
        )
        self.routed_down_weights = torch.nn.Parameter(
            torch.empty(config.n_routed_experts, config.expert_dim, config.moe_inter_dim, dtype=config.dtype)
        )
        self.dist_moe_config = backend.dist_moe
        self.ep_mesh: DeviceMesh | None = None
        self.context: Context | None = None

    @property
    def gate_and_up_projs(self) -> torch.Tensor:
        """Return a canonical [experts, hidden, 2*intermediate] alias for checkpoints.

        DTensor mesh and placements are preserved, with feature shard dimensions
        transposed along with the tensor. This view is not an optimizer parameter.
        """
        return self.routed_gate_up_weights.transpose(-1, -2)

    @property
    def down_projs(self) -> torch.Tensor:
        """Return a canonical [experts, intermediate, hidden] checkpoint alias.

        DTensor placements are transposed with the view. Writing the returned
        tensor updates the native parameter storage.
        """
        return self.routed_down_weights.transpose(-1, -2)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize native storage using the canonical expert RNG order.

        Args:
            buffer_device: Device on which canonical random draws execute.
            init_std: Standard deviation of the zero-mean normal initialization.

        Draw full feature widths for each local EP expert, then select the FSDP
        feature slice. Parameters and their storage are never replaced.
        """
        with torch.device(buffer_device):
            for parameter, features in (
                (self.routed_gate_up_weights, (self.config.expert_dim, 2 * self.config.moe_inter_dim)),
                (self.routed_down_weights, (self.config.moe_inter_dim, self.config.expert_dim)),
            ):
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                canonical = local.new_empty((local.size(0), *features)).normal_(mean=0.0, std=init_std)
                native = canonical.transpose(-1, -2)
                if isinstance(parameter, DTensor):
                    _, offsets = compute_local_shape_and_global_offset(
                        parameter.shape, parameter.device_mesh, parameter.placements
                    )
                    native = native.narrow(1, offsets[1], local.size(1)).narrow(2, offsets[2], local.size(2))
                local.copy_(native)

    def _save_to_state_dict(self, destination: OrderedDict[str, torch.Tensor], prefix: str, keep_vars: bool) -> None:
        """Expose canonical checkpoint keys as views of native parameter storage."""
        for name in ("gate_and_up_projs", "down_projs"):
            value = self.gate_and_up_projs if name == "gate_and_up_projs" else self.down_projs
            destination[prefix + name] = value if keep_vars else value.detach()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Copy canonical checkpoint tensors into the existing native parameters.

        Canonical tensors may be DTensors or rank-local tensors. DCP can fill
        either alias in place and omit that key from its later rebuild pass, so
        the two keys are loaded independently without replacing parameters.
        """
        del local_metadata
        projections = {
            "gate_and_up_projs": self.routed_gate_up_weights,
            "down_projs": self.routed_down_weights,
        }
        with torch.no_grad():
            for name, parameter in projections.items():
                key = prefix + name
                if key not in state_dict:
                    if strict:
                        missing_keys.append(key)
                    continue
                source = state_dict[key]
                source = source.to_local() if isinstance(source, DTensor) else source
                target = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                source = source.transpose(-1, -2)
                if source.shape != target.shape:
                    error_msgs.append(
                        f"size mismatch for {key}: transposed checkpoint {tuple(source.shape)}, "
                        f"native parameter {tuple(target.shape)}"
                    )
                    continue
                target.copy_(source)
        if strict:
            unexpected_keys.extend(
                key for key in state_dict if key.startswith(prefix) and key[len(prefix) :] not in projections
            )

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
            Padding and weight casts preserve autograd; weights already have the kernel layout.
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
        w13 = self.routed_gate_up_weights
        w2 = self.routed_down_weights
        if isinstance(w13, DTensor):
            w13 = w13.to_local()
        if isinstance(w2, DTensor):
            w2 = w2.to_local()
        w13 = w13.to(torch.bfloat16)
        w2 = w2.to(torch.bfloat16)
        output = package.routed_experts(inputs, ids, scores, w13, w2, self.context)
        return output[: x.size(0)]
