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

"""Mixture-of-Kittens expert backend for AutoModel's shared MoE component."""

from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import create_dtensor_from_local
from nemo_automodel.shared.import_utils import safe_import

_MOK_IMPORT_MESSAGE = (
    "dispatcher='mok' requires a built Mixture-of-Kittens installation or an importable mixture-of-kittens checkout."
)
_MOK_AVAILABLE, _mok_functional = safe_import("mok.functional", msg=_MOK_IMPORT_MESSAGE)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local tensor while preserving its autograd connection.

    Args:
        tensor: Tensor or DTensor of arbitrary shape. A DTensor must be sharded or
            replicated on the current rank with a materialized local value.

    Returns:
        Tensor with the DTensor's rank-local shape, or the original plain tensor.
        The returned tensor aliases the input's local storage.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class _MoKRuntime:
    """Own one layer's MoK configuration and expert-parallel process group."""

    def __init__(self, backend: BackendConfig) -> None:
        self.backend = backend
        self.config: object | None = None
        self.ep_group: dist.ProcessGroup | None = None

    def initialize(self, ep_mesh: DeviceMesh, *, n_routed_experts: int) -> None:
        """Attach the runtime to the expert-parallel group.

        Args:
            ep_mesh: One-dimensional device mesh named ``ep``.
            n_routed_experts: Global number of routed experts, divided evenly over
                the EP mesh.
        """
        if ep_mesh.ndim != 1 or ep_mesh.mesh_dim_names != ("ep",):
            raise ValueError(f"MoK requires a one-dimensional 'ep' mesh, got {ep_mesh.mesh_dim_names}")
        if not _MOK_AVAILABLE:
            raise ImportError(_MOK_IMPORT_MESSAGE)
        ep_size = ep_mesh.size()
        if ep_size not in (4, 8, 16, 32, 64):
            raise ValueError(f"MoK EP size must be one of 4, 8, 16, 32, 64; got {ep_size}")
        if n_routed_experts % ep_size != 0:
            raise ValueError(
                f"MoK requires n_routed_experts ({n_routed_experts}) to be divisible by ep_size ({ep_size})"
            )
        self.config = self.backend.mok.build()
        self.ep_group = ep_mesh.get_group()

    def forward(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        shared_gate_weights: torch.Tensor,
        shared_up_weights: torch.Tensor,
        shared_down_weights: torch.Tensor,
        routed_gate_weights: torch.Tensor,
        routed_up_weights: torch.Tensor,
        routed_down_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, object, object]:
        """Run MoK's manual forward and retain its backward context.

        Args:
            x: Contiguous BF16 tensor of shape [tokens, hidden].
            router_weights: Contiguous FP32 tensor of shape [tokens, activated_experts].
            top_experts: Contiguous int64 tensor of shape [tokens, activated_experts].
            shared_gate_weights: Contiguous BF16 tensor of shape [expert_intermediate, hidden].
            shared_up_weights: Contiguous BF16 tensor of shape [expert_intermediate, hidden].
            shared_down_weights: Contiguous BF16 tensor of shape [hidden, expert_intermediate].
            routed_gate_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_up_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_down_weights: Contiguous BF16 tensor of shape
                [local_experts, hidden, expert_intermediate].

        Returns:
            Tuple containing output of shape [tokens, hidden], an opaque MoK schedule,
            and an opaque MoK forward context.
        """
        if self.ep_group is None or self.config is None:
            raise RuntimeError("MoK expert runtime was used before expert-parallel initialization")
        workspace = _mok_functional.get_workspace(
            self.config,
            self.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=top_experts.shape[1],
        )
        schedule = _mok_functional.build_schedule(
            workspace,
            self.config,
            top_experts,
            num_local_experts=routed_gate_weights.shape[0],
        )
        output, forward_context = _mok_functional.forward(
            self.config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        return output, schedule, forward_context

    def backward(
        self,
        schedule: object,
        forward_context: object,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        shared_gate_weights: torch.Tensor,
        shared_up_weights: torch.Tensor,
        shared_down_weights: torch.Tensor,
        routed_gate_weights: torch.Tensor,
        routed_up_weights: torch.Tensor,
        routed_down_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run MoK's manual backward.

        Args:
            schedule: Opaque schedule returned by :meth:`forward`.
            forward_context: Opaque activation context returned by :meth:`forward`.
            grad_output: Contiguous BF16 tensor of shape [tokens, hidden].
            x: Contiguous BF16 tensor of shape [tokens, hidden].
            router_weights: Contiguous FP32 tensor of shape [tokens, activated_experts].
            shared_gate_weights: Contiguous BF16 tensor of shape [expert_intermediate, hidden].
            shared_up_weights: Contiguous BF16 tensor of shape [expert_intermediate, hidden].
            shared_down_weights: Contiguous BF16 tensor of shape [hidden, expert_intermediate].
            routed_gate_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_up_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_down_weights: Contiguous BF16 tensor of shape
                [local_experts, hidden, expert_intermediate].

        Returns:
            Tuple of gradients for ``x``, router weights, three routed weights,
            and three shared weights, with shapes matching their corresponding inputs.
        """
        if self.ep_group is None or self.config is None:
            raise RuntimeError("MoK expert runtime was used before expert-parallel initialization")
        workspace = _mok_functional.get_workspace(
            self.config,
            self.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=router_weights.shape[1],
        )
        return _mok_functional.backward(
            self.config,
            workspace,
            schedule,
            forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )


class _MoKAutogradFunction(torch.autograd.Function):
    """Connect MoK's explicit backward API to PyTorch autograd."""

    @staticmethod
    def forward(
        ctx: object,
        runtime: _MoKRuntime,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        shared_gate_weights: torch.Tensor,
        shared_up_weights: torch.Tensor,
        shared_down_weights: torch.Tensor,
        routed_gate_weights: torch.Tensor,
        routed_up_weights: torch.Tensor,
        routed_down_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run a fused MoK forward pass.

        Args:
            ctx: Autograd-owned context.
            runtime: Initialized MoK runtime for this layer's EP group.
            x: Contiguous BF16 tensor of shape [tokens, hidden].
            router_weights: Contiguous FP32 tensor of shape [tokens, activated_experts].
            top_experts: Contiguous int64 tensor of shape [tokens, activated_experts].
            shared_gate_weights: Contiguous BF16 tensor of shape [expert_intermediate, hidden].
            shared_up_weights: Contiguous BF16 tensor of shape [expert_intermediate, hidden].
            shared_down_weights: Contiguous BF16 tensor of shape [hidden, expert_intermediate].
            routed_gate_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_up_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_down_weights: Contiguous BF16 tensor of shape
                [local_experts, hidden, expert_intermediate].

        Returns:
            BF16 tensor of shape [tokens, hidden].
        """
        output, schedule, forward_context = runtime.forward(
            x,
            router_weights,
            top_experts,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        ctx.runtime = runtime
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.save_for_backward(
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        return output

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Run the fused MoK backward pass.

        Args:
            ctx: Autograd-owned context containing the forward schedule and tensors.
            grad_output: BF16 tensor of shape [tokens, hidden].

        Returns:
            Gradients matching every :meth:`forward` input. The runtime and integer
            expert-index inputs have ``None`` gradients.
        """
        (
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        ) = ctx.saved_tensors
        (
            grad_x,
            grad_router_weights,
            grad_routed_gate,
            grad_routed_up,
            grad_routed_down,
            grad_shared_gate,
            grad_shared_up,
            grad_shared_down,
        ) = ctx.runtime.backward(
            ctx.schedule,
            ctx.forward_context,
            grad_output,
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        return (
            None,
            grad_x,
            grad_router_weights,
            None,
            grad_shared_gate,
            grad_shared_up,
            grad_shared_down,
            grad_routed_gate,
            grad_routed_up,
            grad_routed_down,
        )


class GroupedExpertsMoK(nn.Module):
    """AutoModel expert parameters executed by Mixture-of-Kittens.

    Routed parameters use MoK-native contiguous layouts during training while
    state-dict hooks preserve AutoModel's established combined expert keys.
    """

    def __init__(self, config: MoEConfig, backend: BackendConfig) -> None:
        """Construct MoK-native routed expert parameters.

        Args:
            config: AutoModel MoE configuration.
            backend: Backend configuration whose ``mok`` field owns runtime tuning.
        """
        super().__init__()
        self._validate_model_config(config)
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.expert_bias = False
        self.routed_gate_weights = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim, dtype=config.dtype)
        )
        self.routed_up_weights = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim, dtype=config.dtype)
        )
        self.routed_down_weights = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, config.moe_inter_dim, dtype=config.dtype)
        )
        self.runtime = _MoKRuntime(backend)
        self.ep_mesh: DeviceMesh | None = None
        self.ep_rank = 0

    @staticmethod
    def _validate_model_config(config: MoEConfig) -> None:
        """Reject MoE variants outside MoK's current fused-kernel contract."""
        unsupported: list[str] = []
        if config.dtype != torch.bfloat16:
            unsupported.append(f"dtype={config.dtype}")
        if config.dim % 256 != 0:
            unsupported.append(f"dim={config.dim} (must be divisible by 256)")
        if config.moe_inter_dim % 256 != 0:
            unsupported.append(f"moe_inter_dim={config.moe_inter_dim} (must be divisible by 256)")
        if config.n_shared_experts != 1:
            unsupported.append(f"n_shared_experts={config.n_shared_experts} (must be 1)")
        if config.shared_expert_inter_dim not in (None, config.moe_inter_dim):
            unsupported.append("shared_expert_inter_dim must equal moe_inter_dim")
        if config.expert_activation != "swiglu" or config.shared_expert_activation != "swiglu":
            unsupported.append("routed and shared expert activations must both be swiglu")
        if config.expert_bias:
            unsupported.append("expert_bias=True")
        if config.shared_expert_gate:
            unsupported.append("shared_expert_gate=True")
        if config.moe_latent_size is not None:
            unsupported.append(f"moe_latent_size={config.moe_latent_size}")
        if config.swiglu_limit != 0.0:
            unsupported.append(f"swiglu_limit={config.swiglu_limit}")
        if unsupported:
            raise ValueError("dispatcher='mok' does not support this MoE config: " + "; ".join(unsupported))

    def init_token_dispatcher(self, ep_mesh: DeviceMesh) -> None:
        """Initialize MoK on an expert-parallel device mesh.

        Args:
            ep_mesh: One-dimensional device mesh named ``ep``.
        """
        self.ep_mesh = ep_mesh
        self.ep_rank = ep_mesh.get_local_rank()
        self.runtime.initialize(ep_mesh, n_routed_experts=self.n_routed_experts)

    @property
    def gate_and_up_projs(self) -> torch.Tensor:
        """Return AutoModel's virtual combined GateUp state-dict tensor.

        Returns:
            Tensor of shape [experts, hidden, 2 * expert_intermediate], sharded
            over the EP mesh on the expert axis when EP is initialized.
        """
        gate = _local_tensor(self.routed_gate_weights).transpose(-1, -2)
        up = _local_tensor(self.routed_up_weights).transpose(-1, -2)
        combined = torch.cat((gate, up), dim=-1)
        return create_dtensor_from_local(combined, self.ep_mesh, self.ep_rank if self.ep_mesh is not None else None)

    @property
    def down_projs(self) -> torch.Tensor:
        """Return AutoModel's virtual down-projection state-dict tensor.

        Returns:
            Tensor of shape [experts, expert_intermediate, hidden], sharded over
            the EP mesh on the expert axis when EP is initialized.
        """
        down = _local_tensor(self.routed_down_weights).transpose(-1, -2)
        return create_dtensor_from_local(down, self.ep_mesh, self.ep_rank if self.ep_mesh is not None else None)

    @property
    def gate_up_proj_bias(self) -> None:
        """Return ``None`` because MoK does not support expert bias."""
        return None

    @property
    def down_proj_bias(self) -> None:
        """Return ``None`` because MoK does not support expert bias."""
        return None

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        shared_gate_weights: torch.Tensor,
        shared_up_weights: torch.Tensor,
        shared_down_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Execute fused shared and routed SwiGLU experts.

        Args:
            x: BF16 tensor of shape [tokens, hidden].
            token_mask: Boolean tensor of shape [tokens]. All entries must be true;
                padding is rejected by the owning :class:`MoE` before this call.
            weights: Tensor of shape [tokens, activated_experts].
            indices: Int64 tensor of shape [tokens, activated_experts].
            shared_gate_weights: BF16 tensor of shape [expert_intermediate, hidden].
            shared_up_weights: BF16 tensor of shape [expert_intermediate, hidden].
            shared_down_weights: BF16 tensor of shape [hidden, expert_intermediate].

        Returns:
            BF16 tensor of shape [tokens, hidden] containing the fused shared and
            router-weighted routed expert output.
        """
        del token_mask
        if isinstance(x, DTensor):
            raise ValueError("MoK expects rank-local activations, not a DTensor")
        return _MoKAutogradFunction.apply(
            self.runtime,
            x.contiguous(),
            weights.float().contiguous(),
            indices.contiguous(),
            _local_tensor(shared_gate_weights).contiguous(),
            _local_tensor(shared_up_weights).contiguous(),
            _local_tensor(shared_down_weights).contiguous(),
            _local_tensor(self.routed_gate_weights),
            _local_tensor(self.routed_up_weights),
            _local_tensor(self.routed_down_weights),
        )

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize MoK-native expert weights.

        Args:
            buffer_device: Device on which initialization kernels execute.
            init_std: Standard deviation of the normal weight initialization.
        """
        with torch.device(buffer_device):
            _local_tensor(self.routed_gate_weights).normal_(mean=0.0, std=init_std)
            _local_tensor(self.routed_up_weights).normal_(mean=0.0, std=init_std)
            _local_tensor(self.routed_down_weights).normal_(mean=0.0, std=init_std)

    def _save_to_state_dict(
        self,
        destination: OrderedDict[str, torch.Tensor],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        """Save virtual AutoModel expert tensors instead of MoK-native parameters."""
        gate_up = self.gate_and_up_projs
        down = self.down_projs
        destination[f"{prefix}gate_and_up_projs"] = gate_up if keep_vars else gate_up.detach()
        destination[f"{prefix}down_projs"] = down if keep_vars else down.detach()

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
        """Load established AutoModel expert tensors into MoK-native parameters."""
        del local_metadata, strict, unexpected_keys, error_msgs
        gate_up_key = f"{prefix}gate_and_up_projs"
        down_key = f"{prefix}down_projs"
        gate_up = state_dict.get(gate_up_key)
        down = state_dict.get(down_key)
        if gate_up is None:
            missing_keys.append(gate_up_key)
        if down is None:
            missing_keys.append(down_key)
        if gate_up is None or down is None:
            return
        gate_up = _local_tensor(gate_up)
        down = _local_tensor(down)
        inter_dim = self.config.moe_inter_dim
        expected_gate_up_shape = (*_local_tensor(self.routed_gate_weights).shape[:1], self.config.dim, 2 * inter_dim)
        expected_down_shape = (*_local_tensor(self.routed_down_weights).shape[:1], inter_dim, self.config.dim)
        if tuple(gate_up.shape) != expected_gate_up_shape:
            raise RuntimeError(
                f"Cannot load {gate_up_key}: expected local shape {expected_gate_up_shape}, got {tuple(gate_up.shape)}"
            )
        if tuple(down.shape) != expected_down_shape:
            raise RuntimeError(
                f"Cannot load {down_key}: expected local shape {expected_down_shape}, got {tuple(down.shape)}"
            )
        with torch.no_grad():
            _local_tensor(self.routed_gate_weights).copy_(gate_up[..., :inter_dim].transpose(-1, -2))
            _local_tensor(self.routed_up_weights).copy_(gate_up[..., inter_dim:].transpose(-1, -2))
            _local_tensor(self.routed_down_weights).copy_(down.transpose(-1, -2))
