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

import math
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig, MoKBackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import create_dtensor_from_local
from nemo_automodel.shared.import_utils import safe_import

_MOK_IMPORT_MESSAGE = (
    "dispatcher='mok' requires a built Mixture-of-Kittens installation or an importable mixture-of-kittens checkout."
)
_mok_functional = None
_mok_ops = None


def _load_mok_functional():
    """Import MoK after distributed setup has selected this process's GPU.

    Importing MoK loads its CUDA extension.  Doing that at module-import time is
    too early for torchrun workers: every local worker still has CUDA device 0
    selected, leaving one stray CUDA context per worker on the first GPU.  MoK
    is only needed when expert parallelism is initialized, which happens after
    ``initialize_distributed`` calls ``torch.cuda.set_device``.
    """
    global _mok_functional
    if _mok_functional is None:
        available, functional = safe_import("mok.functional", msg=_MOK_IMPORT_MESSAGE)
        if not available:
            raise ImportError(_MOK_IMPORT_MESSAGE)
        _mok_functional = functional
    return _mok_functional


def _load_mok_ops():
    """Import MoK's weight-quantization ops after selecting this process's GPU."""
    global _mok_ops
    if _mok_ops is None:
        available, ops = safe_import("mok.ops", msg=_MOK_IMPORT_MESSAGE)
        if not available:
            raise ImportError(_MOK_IMPORT_MESSAGE)
        _mok_ops = ops
    return _mok_ops


def _mxfp8_weight_both(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prequantize both MoK MXFP8 layouts in one kernel launch.

    Args:
        weight: Contiguous BF16 tensor of shape
            [local_experts, output_features, input_features].

    Returns:
        Normal E4M3 weight, its E8M0 block scales, transposed E4M3 weight,
        and its E8M0 block scales in the opaque layouts consumed by MoK.
    """
    weight_fp8, weight_scale, weight_t_fp8, weight_t_scale = _load_mok_ops().mxfp8_quantize(weight, True, True)
    if weight_fp8 is None or weight_scale is None or weight_t_fp8 is None or weight_t_scale is None:
        raise RuntimeError("MoK MXFP8 quantization did not return both requested weight layouts")
    return weight_fp8, weight_scale, weight_t_fp8, weight_t_scale


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


def _flatten_mok_tensor_dataclass(
    value: object,
) -> tuple[tuple[torch.Tensor, ...], tuple[type, tuple[tuple[str, int], ...]]]:
    """Flatten a MoK schedule/context without retaining tensor references in metadata.

    MoK's dataclasses contain either tensors or tuples of tensors (the latter for
    MXFP8 values).  Saving the flattened tensors with ``ctx.save_for_backward``
    makes them visible to activation-checkpoint saved-tensor hooks.  Storing the
    original dataclass directly on ``ctx`` would keep every layer's macro-sized
    context alive until backward.

    Args:
        value: MoK schedule or forward-context dataclass.

    Returns:
        Flat tensors and tensor-free reconstruction metadata.
    """
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"Expected a MoK tensor dataclass, got {type(value).__name__}")

    tensors: list[torch.Tensor] = []
    field_layout: list[tuple[str, int]] = []
    for field in fields(value):
        field_value = getattr(value, field.name)
        if isinstance(field_value, torch.Tensor):
            field_tensors = (field_value,)
        elif (
            isinstance(field_value, tuple)
            and field_value
            and all(isinstance(item, torch.Tensor) for item in field_value)
        ):
            field_tensors = field_value
        else:
            raise TypeError(
                f"MoK field {type(value).__name__}.{field.name} must be a tensor or nonempty tuple of tensors"
            )
        field_layout.append((field.name, len(field_tensors)))
        tensors.extend(field_tensors)
    return tuple(tensors), (type(value), tuple(field_layout))


def _unflatten_mok_tensor_dataclass(
    spec: tuple[type, tuple[tuple[str, int], ...]],
    tensors: tuple[torch.Tensor, ...],
    offset: int,
) -> tuple[object, int]:
    """Reconstruct a MoK dataclass from checkpoint-managed saved tensors."""
    dataclass_type, field_layout = spec
    values: dict[str, Any] = {}
    for name, count in field_layout:
        end = offset + count
        if end > len(tensors):
            raise RuntimeError("MoK saved-tensor metadata exceeds the available tensors")
        field_tensors = tensors[offset:end]
        values[name] = field_tensors[0] if count == 1 else field_tensors
        offset = end
    return dataclass_type(**values), offset


class _MoKRuntime:
    """Own one layer's MoK configuration and expert-parallel process group."""

    def __init__(self, mok_config: MoKBackendConfig, *, swiglu_limit: float) -> None:
        self.mok_config = mok_config
        self.swiglu_limit = None if swiglu_limit == 0.0 else float(swiglu_limit)
        self.config: object | None = None
        self.ep_group: dist.ProcessGroup | None = None
        self._mxfp8_cache_generation = 0
        self._mxfp8_cached_generation: int | None = None
        self._mxfp8_weights: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...] | None = None
        self._retain_mxfp8_cache_until_optimizer_step = False

    def _get_workspace(self, x: torch.Tensor, topk: int) -> object:
        """Return MoK's workspace for one local activation shape.

        Args:
            x: Contiguous BF16 tensor of shape [tokens, hidden].
            topk: Number of activated experts per token.

        Returns:
            Opaque MoK workspace shared by schedule construction and kernels.
        """
        return _mok_functional.get_workspace(
            self.config,
            self.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=topk,
        )

    def _get_mxfp8_weights(
        self,
        routed_gate_weights: torch.Tensor,
        routed_up_weights: torch.Tensor,
        routed_down_weights: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Prequantize all routed-weight layouts once per optimizer generation.

        Args:
            routed_gate_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_up_weights: Contiguous BF16 tensor of shape
                [local_experts, expert_intermediate, hidden].
            routed_down_weights: Contiguous BF16 tensor of shape
                [local_experts, hidden, expert_intermediate].

        Returns:
            Gate, up, and down projection layouts. Each entry contains the normal
            E4M3 weight, its E8M0 block scales, the transposed E4M3 weight, and its
            E8M0 block scales in the opaque layouts consumed by MoK.
        """
        weights = (routed_gate_weights, routed_up_weights, routed_down_weights)
        if self._mxfp8_weights is None or self._mxfp8_cached_generation != self._mxfp8_cache_generation:
            self._mxfp8_weights = tuple(_mxfp8_weight_both(weight) for weight in weights)
            self._mxfp8_cached_generation = self._mxfp8_cache_generation
        return self._mxfp8_weights

    def _invalidate_mxfp8_cache(self) -> None:
        """Advance the BF16-weight generation and drop its quantized tensors."""
        self._mxfp8_cache_generation += 1
        self._mxfp8_cached_generation = None
        self._mxfp8_weights = None

    def initialize(self, ep_mesh: DeviceMesh, *, n_routed_experts: int) -> None:
        """Attach the runtime to the expert-parallel group.

        Args:
            ep_mesh: One-dimensional device mesh named ``ep``.
            n_routed_experts: Global number of routed experts, divided evenly over
                the EP mesh.
        """
        if ep_mesh.ndim != 1 or ep_mesh.mesh_dim_names != ("ep",):
            raise ValueError(f"MoK requires a one-dimensional 'ep' mesh, got {ep_mesh.mesh_dim_names}")
        ep_size = ep_mesh.size()
        if ep_size not in (4, 8, 16, 32, 64):
            raise ValueError(f"MoK EP size must be one of 4, 8, 16, 32, 64; got {ep_size}")
        if n_routed_experts % ep_size != 0:
            raise ValueError(
                f"MoK requires n_routed_experts ({n_routed_experts}) to be divisible by ep_size ({ep_size})"
            )
        # Load the CUDA extension only after torchrun has bound this worker to
        # its local device.  This avoids all workers creating a context on GPU 0.
        _load_mok_functional()
        if self.mok_config.precision == "mxfp8":
            _load_mok_ops()
        self.config = self.mok_config.build()
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
    ) -> tuple[
        torch.Tensor,
        object,
        object,
        tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...] | None,
    ]:
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
            Tuple containing the BF16 output of shape [tokens, hidden], an opaque
            MoK schedule, an opaque MoK forward context, and prequantized routed
            weights. The weights are ``None`` for BF16. For MXFP8 they contain the
            gate, up, and down projections, each represented by
            ``(weight_fp8, weight_scale, weight_t_fp8, weight_t_scale)``.
        """
        if self.ep_group is None or self.config is None:
            raise RuntimeError("MoK expert runtime was used before expert-parallel initialization")
        workspace = self._get_workspace(x, top_experts.shape[1])
        schedule = _mok_functional.build_schedule(
            workspace,
            self.config,
            top_experts,
            num_local_experts=routed_gate_weights.shape[0],
        )
        if self.mok_config.precision == "mxfp8":
            mxfp8_weights = self._get_mxfp8_weights(
                routed_gate_weights,
                routed_up_weights,
                routed_down_weights,
            )
            forward_routed_gate_weights = mxfp8_weights[0][:2]
            forward_routed_up_weights = mxfp8_weights[1][:2]
            forward_routed_down_weights = mxfp8_weights[2][:2]
        else:
            mxfp8_weights = None
            forward_routed_gate_weights = routed_gate_weights
            forward_routed_up_weights = routed_up_weights
            forward_routed_down_weights = routed_down_weights
        output, forward_context = _mok_functional.forward(
            self.config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            forward_routed_gate_weights,
            forward_routed_up_weights,
            forward_routed_down_weights,
            self.swiglu_limit,
        )
        return output, schedule, forward_context, mxfp8_weights

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
        mxfp8_weights: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...] | None,
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
            mxfp8_weights: Prequantized gate, up, and down projections. Each is a
                tuple ``(weight_fp8, weight_scale, weight_t_fp8, weight_t_scale)``
                in MoK's opaque E4M3/E8M0 layouts. Must be non-``None`` for MXFP8
                and ``None`` for BF16.

        Returns:
            Tuple of gradients for ``x``, router weights, three routed weights,
            and three shared weights, with shapes matching their corresponding inputs.
        """
        if self.ep_group is None or self.config is None:
            raise RuntimeError("MoK expert runtime was used before expert-parallel initialization")
        workspace = self._get_workspace(x, router_weights.shape[1])
        if self.mok_config.precision == "mxfp8":
            if mxfp8_weights is None:
                raise RuntimeError("MoK MXFP8 backward requires the prequantized weights")
            if len(mxfp8_weights) != 3:
                raise RuntimeError(f"MoK MXFP8 backward expected 3 prequantized weights, got {len(mxfp8_weights)}")
            if any(any(tensor is None for tensor in weight) for weight in mxfp8_weights):
                raise RuntimeError("MoK MXFP8 backward requires both layouts for every routed weight")
            backward_routed_gate_weights = mxfp8_weights[0]
            backward_routed_up_weights = mxfp8_weights[1]
            backward_routed_down_weights = mxfp8_weights[2][2:]
        else:
            if mxfp8_weights is not None:
                raise RuntimeError("MoK BF16 backward received unexpected MXFP8 weights")
            backward_routed_gate_weights = routed_gate_weights
            backward_routed_up_weights = routed_up_weights
            backward_routed_down_weights = routed_down_weights
        try:
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
                backward_routed_gate_weights,
                backward_routed_up_weights,
                backward_routed_down_weights,
                self.swiglu_limit,
            )
        finally:
            if not self._retain_mxfp8_cache_until_optimizer_step:
                self._invalidate_mxfp8_cache()


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
        output, schedule, forward_context, mxfp8_weights = runtime.forward(
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
        schedule_tensors, schedule_spec = _flatten_mok_tensor_dataclass(schedule)
        forward_context_tensors, forward_context_spec = _flatten_mok_tensor_dataclass(forward_context)
        mxfp8_weight_tensors = (
            tuple(tensor for weight in mxfp8_weights for tensor in weight) if mxfp8_weights is not None else ()
        )
        ctx.runtime = runtime
        ctx.mok_tensor_specs = (schedule_spec, forward_context_spec)
        ctx.mxfp8_weight_tensor_count = len(mxfp8_weight_tensors)
        ctx.save_for_backward(
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
            *mxfp8_weight_tensors,
            *schedule_tensors,
            *forward_context_tensors,
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
        saved_tensors = ctx.saved_tensors
        (
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        ) = saved_tensors[:8]
        offset = 8
        mxfp8_tensor_count = ctx.mxfp8_weight_tensor_count
        if mxfp8_tensor_count:
            if mxfp8_tensor_count != 12:
                raise RuntimeError(f"MoK MXFP8 backward expected 12 saved weight tensors, got {mxfp8_tensor_count}")
            mxfp8_tensors = saved_tensors[offset : offset + mxfp8_tensor_count]
            mxfp8_weights = tuple(tuple(mxfp8_tensors[index : index + 4]) for index in range(0, mxfp8_tensor_count, 4))
            offset += mxfp8_tensor_count
        else:
            mxfp8_weights = None
        schedule, offset = _unflatten_mok_tensor_dataclass(ctx.mok_tensor_specs[0], saved_tensors, offset)
        forward_context, offset = _unflatten_mok_tensor_dataclass(ctx.mok_tensor_specs[1], saved_tensors, offset)
        if offset != len(saved_tensors):
            raise RuntimeError(f"MoK saved-tensor metadata consumed {offset} of {len(saved_tensors)} tensors")
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
            schedule,
            forward_context,
            grad_output,
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
            mxfp8_weights,
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
        self._validate_model_mok_config(config)
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
        self.runtime = _MoKRuntime(backend.mok, swiglu_limit=config.swiglu_limit)
        self.ep_mesh: DeviceMesh | None = None
        self.ep_rank = 0

    @staticmethod
    def _validate_model_mok_config(config: MoEConfig) -> None:
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
        if (
            type(config.swiglu_limit) not in (int, float)
            or not math.isfinite(config.swiglu_limit)
            or config.swiglu_limit < 0
        ):
            unsupported.append(f"swiglu_limit={config.swiglu_limit} (must be non-negative and finite)")
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
        weights: torch.Tensor,
        indices: torch.Tensor,
        shared_gate_weights: torch.Tensor,
        shared_up_weights: torch.Tensor,
        shared_down_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Execute fused shared and routed SwiGLU experts.

        Args:
            x: BF16 tensor of shape [tokens, hidden].
            weights: Tensor of shape [tokens, activated_experts].
            indices: Int64 tensor of shape [tokens, activated_experts].
            shared_gate_weights: BF16 tensor of shape [expert_intermediate, hidden].
            shared_up_weights: BF16 tensor of shape [expert_intermediate, hidden].
            shared_down_weights: BF16 tensor of shape [hidden, expert_intermediate].

        Returns:
            BF16 tensor of shape [tokens, hidden] containing the fused shared and
            router-weighted routed expert output.
        """
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
        """Initialize MoK-native weights from the canonical expert layout.

        Draw random values in the same tensor shapes and order as
        :class:`GroupedExperts` before transposing them into MoK's native
        layouts.  Otherwise switching only the dispatcher changes a
        random-initialized model even when every process uses the same seed.

        Args:
            buffer_device: Device on which initialization kernels execute.
            init_std: Standard deviation of the normal weight initialization.
        """
        with torch.device(buffer_device):
            routed_gate = _local_tensor(self.routed_gate_weights)
            routed_up = _local_tensor(self.routed_up_weights)
            routed_down = _local_tensor(self.routed_down_weights)
            inter_dim = self.config.moe_inter_dim

            # GroupedExperts draws one contiguous [gate | up] tensor.  Preserve
            # that RNG stream and semantic layout, then copy into the two
            # contiguous tensors consumed by MoK.
            canonical_gate_up = routed_gate.new_empty((routed_gate.size(0), self.config.dim, 2 * inter_dim))
            canonical_gate_up.normal_(mean=0.0, std=init_std)
            routed_gate.copy_(canonical_gate_up[..., :inter_dim].transpose(-1, -2))
            routed_up.copy_(canonical_gate_up[..., inter_dim:].transpose(-1, -2))
            del canonical_gate_up

            # GroupedExperts draws down projection weights immediately after
            # the combined tensor, in [expert, intermediate, hidden] order.
            canonical_down = routed_down.new_empty((routed_down.size(0), inter_dim, self.config.dim))
            canonical_down.normal_(mean=0.0, std=init_std)
            routed_down.copy_(canonical_down.transpose(-1, -2))
        self.runtime._invalidate_mxfp8_cache()

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
        inter_dim = self.config.moe_inter_dim
        expected_gate_up_shape = (*_local_tensor(self.routed_gate_weights).shape[:1], self.config.dim, 2 * inter_dim)
        expected_down_shape = (*_local_tensor(self.routed_down_weights).shape[:1], inter_dim, self.config.dim)
        with torch.no_grad():
            # DCP may already have written one virtual tensor through an aliasing
            # state-dict view while the other takes the rebuild path. Load the two
            # keys independently so the absence of an in-place-loaded key does not
            # suppress copying the rebuilt key.
            if gate_up is not None:
                gate_up = _local_tensor(gate_up)
                if tuple(gate_up.shape) != expected_gate_up_shape:
                    raise RuntimeError(
                        f"Cannot load {gate_up_key}: expected local shape {expected_gate_up_shape}, "
                        f"got {tuple(gate_up.shape)}"
                    )
                _local_tensor(self.routed_gate_weights).copy_(gate_up[..., :inter_dim].transpose(-1, -2))
                _local_tensor(self.routed_up_weights).copy_(gate_up[..., inter_dim:].transpose(-1, -2))
            if down is not None:
                down = _local_tensor(down)
                if tuple(down.shape) != expected_down_shape:
                    raise RuntimeError(
                        f"Cannot load {down_key}: expected local shape {expected_down_shape}, got {tuple(down.shape)}"
                    )
                _local_tensor(self.routed_down_weights).copy_(down.transpose(-1, -2))
        self.runtime._invalidate_mxfp8_cache()


def enable_mok_mxfp8_optimizer_step_cache(
    model_parts: list[nn.Module], optimizers: list[torch.optim.Optimizer]
) -> None:
    """Retain MXFP8 weights within a step and invalidate them after optimizer updates.

    Args:
        model_parts: Pipeline model parts paired one-to-one with ``optimizers``.
        optimizers: Optimizers whose post-step hooks own cache invalidation.
    """
    runtimes_by_part = tuple(
        tuple(
            module.runtime
            for module in model_part.modules()
            if isinstance(module, GroupedExpertsMoK) and module.runtime.mok_config.precision == "mxfp8"
        )
        for model_part in model_parts
    )
    if not any(runtimes_by_part):
        return
    if len(runtimes_by_part) != len(optimizers):
        raise ValueError(
            f"MoK MXFP8 cache expected one optimizer per model part, got {len(optimizers)} optimizers "
            f"for {len(model_parts)} parts"
        )

    for runtimes, optimizer in zip(runtimes_by_part, optimizers, strict=True):
        if not runtimes:
            continue
        for runtime in runtimes:
            runtime._retain_mxfp8_cache_until_optimizer_step = True

        def invalidate_cache_after_step(
            _optimizer: torch.optim.Optimizer,
            _args: tuple[object, ...],
            _kwargs: dict[str, object],
            *,
            runtimes: tuple[_MoKRuntime, ...] = runtimes,
        ) -> None:
            for runtime in runtimes:
                runtime._invalidate_mxfp8_cache()

        optimizer.register_step_post_hook(invalidate_cache_after_step)
