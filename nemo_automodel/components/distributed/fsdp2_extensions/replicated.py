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

"""Bounded replication for small, precision-sensitive FSDP2 parameters."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

# Repository PEFT recipes use ranks 4-64 (79% use rank 8 or 16). For an FP32
# LoRA A+B pair, bytes = 4 * rank * (in_features + out_features), so 8 MiB
# covers rank 64 on a square 16K-wide projection while still bounding each
# independently managed module.
DEFAULT_MAX_REPLICATED_PARAM_BYTES_PER_MODULE = 8 * 1024 * 1024
_GRAD_SYNC_ATTR = "_nemo_fsdp2_replicated_grad_sync"
_ShardedReason = Literal["size_limit", "non_fp32_residency"]


@dataclass(frozen=True)
class ManagedModuleSelection:
    """Replication decision for one explicitly managed module.

    Attributes:
        name: Qualified module name.
        parameters: Parameter tensors of arbitrary shape. DTensors retain their
            global shape, device mesh, and placements.
        logical_bytes: Total bytes over global parameter shapes.
        replicated: Whether the parameters remain outside FSDP ownership.
        sharded_reason: Reason a managed module retained sharded ownership.
    """

    name: str
    parameters: tuple[nn.Parameter, ...]
    logical_bytes: int
    replicated: bool
    sharded_reason: _ShardedReason | None = None


@dataclass(frozen=True)
class ReplicatedParameterSelection:
    """Model-wide result of independent per-managed-module decisions.

    Attributes:
        modules: Decisions containing parameter tensors of arbitrary shape for
            every explicitly managed module.
    """

    modules: tuple[ManagedModuleSelection, ...]

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for module in self.modules if module.replicated for parameter in module.parameters)

    @property
    def replicated_bytes(self) -> int:
        return sum(module.logical_bytes for module in self.modules if module.replicated)

    @property
    def oversized_modules(self) -> tuple[ManagedModuleSelection, ...]:
        return tuple(module for module in self.modules if module.sharded_reason == "size_limit")


def select_small_fp32_parameters(
    module: nn.Module,
    *,
    name_fragments: tuple[str, ...],
    max_bytes_per_module: int = DEFAULT_MAX_REPLICATED_PARAM_BYTES_PER_MODULE,
) -> ReplicatedParameterSelection:
    """Select FP32 parameters using an independent limit per managed module.

    A managed module is a maximal named submodule whose qualified name matches
    ``name_fragments``. The byte limit is evaluated independently for each such
    module over global parameter shapes, so TP sharding cannot make a large
    logical module accidentally qualify for replication. Eligible modules are
    still coalesced into one model-part gradient synchronization buffer.

    Args:
        module: Model containing candidate parameter tensors of arbitrary shape.
        name_fragments: Qualified-name fragments identifying managed modules.
        max_bytes_per_module: Maximum logical bytes allowed for each managed module.

    Returns:
        Per-module selection decisions and the flattened eligible parameter tensors.
        DTensors preserve global shapes, meshes, and placements.

    Raises:
        ValueError: If ``max_bytes_per_module`` is negative.
    """
    if max_bytes_per_module < 0:
        raise ValueError(f"max_bytes_per_module must be non-negative, got {max_bytes_per_module}")

    matched_modules: list[tuple[str, nn.Module]] = []
    for name, candidate in module.named_modules():
        if not name or not any(fragment in name for fragment in name_fragments):
            continue
        if any(name.startswith(parent_name + ".") for parent_name, _ in matched_modules):
            continue
        matched_modules.append((name, candidate))

    decisions: list[ManagedModuleSelection] = []
    assigned_param_ids: set[int] = set()
    for module_name, candidate in matched_modules:
        parameters = tuple(parameter for parameter in candidate.parameters() if id(parameter) not in assigned_param_ids)
        if not parameters:
            continue
        assigned_param_ids.update(id(parameter) for parameter in parameters)
        logical_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
        resident_fp32 = all(
            not parameter.dtype.is_floating_point or parameter.dtype is torch.float32 for parameter in parameters
        )
        replicated = resident_fp32 and logical_bytes <= max_bytes_per_module
        sharded_reason = None
        if not resident_fp32:
            sharded_reason = "non_fp32_residency"
        elif not replicated:
            sharded_reason = "size_limit"
        decisions.append(
            ManagedModuleSelection(
                name=module_name,
                parameters=parameters,
                logical_bytes=logical_bytes,
                replicated=replicated,
                sharded_reason=sharded_reason,
            )
        )
    return ReplicatedParameterSelection(tuple(decisions))


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local tensor without copying storage.

    Args:
        tensor: Tensor of arbitrary shape. A DTensor may use any TP placement;
            its rank-local shard is returned.

    Returns:
        Tensor of arbitrary rank with the rank-local shape and storage. A plain
        tensor is returned unchanged and aliases the input.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@dataclass
class _ParameterSlot:
    """Stable module-owned location for a replaceable replicated parameter.

    Meta-device checkpoint initialization may replace a module's ``Parameter``
    object after FSDP ownership is established. Keeping the owning module and
    local attribute name lets gradient synchronization resolve the current,
    materialized parameter without retaining the stale meta tensor.

    Attributes:
        module: Module whose direct parameter mapping owns ``name``.
        name: Local parameter name in ``module._parameters``.
    """

    module: nn.Module
    name: str

    def resolve(self) -> nn.Parameter:
        """Return the current parameter object installed in this module slot."""
        parameter = self.module._parameters.get(self.name)
        if not isinstance(parameter, nn.Parameter):
            raise RuntimeError(f"replicated parameter slot {type(self.module).__name__}.{self.name} is missing")
        return parameter


@dataclass
class _ReplicatedGradSync:
    """Coalesce replicated gradients into one FP32 all-reduce per mesh dim.

    Attributes:
        parameter_slots: Stable module slots for replicated parameter tensors of
            arbitrary shape. Resolved DTensors may retain TP placements but must
            be replicated over ``mesh``.
        mesh: Data-parallel FSDP or HSDP mesh reduced dimension by dimension.
    """

    parameter_slots: tuple[_ParameterSlot, ...]
    mesh: DeviceMesh

    @torch.no_grad()
    def synchronize(self) -> None:
        if not self.parameter_slots or self.mesh.size() == 1:
            return

        parameters = tuple(slot.resolve() for slot in self.parameter_slots)
        local_parameters = tuple(_local_tensor(parameter) for parameter in parameters)
        device = local_parameters[0].device
        # One FP32 payload carries a rank-symmetric error flag, one local-use bit
        # per parameter, and every gradient. Build the complete header on the
        # host and transfer it once: assigning one scalar use bit at a time makes
        # every pageable H2D write synchronize the CUDA stream.
        header_size = 1 + len(parameters)
        local_grads: list[torch.Tensor | None] = []
        local_used: list[bool] = []
        invalid = False
        for parameter, local_parameter in zip(parameters, local_parameters):
            invalid |= local_parameter.device != device
            grad = parameter.grad
            local_used.append(grad is not None)
            local_grad = None if grad is None else _local_tensor(grad)
            if local_grad is not None:
                valid = (
                    local_grad.dtype is torch.float32
                    and local_grad.device == device
                    and local_grad.shape == local_parameter.shape
                )
                invalid |= not valid
                if not valid:
                    local_grad = None
            local_grads.append(local_grad)

        flat_grad = torch.empty(
            header_size + sum(parameter.numel() for parameter in local_parameters),
            dtype=torch.float32,
            device=device,
        )
        flat_grad.zero_()
        flat_grad[:header_size].copy_(torch.tensor([invalid, *local_used], dtype=torch.float32, device=device))
        offset = header_size
        for local_parameter, local_grad in zip(local_parameters, local_grads):
            if local_grad is not None:
                flat_grad[offset : offset + local_parameter.numel()].copy_(local_grad.reshape(-1))
            offset += local_parameter.numel()

        reduced_world_size = 1
        for group in self.mesh.get_all_groups():
            group_size = dist.get_world_size(group=group)
            if group_size > 1:
                dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=group)
                reduced_world_size *= group_size
        reduced_header = flat_grad[:header_size].cpu()
        if reduced_header[0].item() != 0:
            raise RuntimeError(
                "replicated parameter gradients must be FP32 tensors with the same local shape and device as their "
                "parameters on every data-parallel rank"
            )
        globally_used = reduced_header[1:].bool().tolist()
        if reduced_world_size > 1:
            flat_grad[header_size:].div_(reduced_world_size)

        offset = header_size
        for parameter, local_parameter, used in zip(parameters, local_parameters, globally_used):
            next_offset = offset + local_parameter.numel()
            if not used:
                # Globally unused parameters retain grad=None, so optimizers do
                # not apply weight decay or advance state for them.
                parameter.grad = None
                offset = next_offset
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            local_grad = _local_tensor(parameter.grad)
            local_grad.copy_(flat_grad[offset:next_offset].view_as(local_grad))
            offset = next_offset


def _fsdp_requires_gradient_sync(fsdp_state: object) -> bool:
    """Return whether the current FSDP backward is reducing gradients."""
    state_ctx = getattr(fsdp_state, "_state_ctx", None)
    all_states = getattr(state_ctx, "all_states", ())
    param_groups = [
        param_group for state in all_states if (param_group := getattr(state, "_fsdp_param_group", None)) is not None
    ]
    # A root containing only ignored parameters has no FSDP parameter group, but
    # its replicated parameters still need their normal data-parallel reduction.
    return not param_groups or any(bool(param_group.reduce_grads) for param_group in param_groups)


def _install_fsdp_post_backward_grad_sync(model: nn.Module, grad_sync: _ReplicatedGradSync) -> None:
    """Run replicated-gradient synchronization from FSDP's final callback.

    FSDP owns the accumulation lifecycle through ``set_requires_gradient_sync``.
    Wrapping its root post-backward callback means deferred backwards accumulate
    locally, and the first backward for which FSDP reduces gradients also reduces
    the full accumulated FP32 gradients. Re-reducing an already averaged prefix
    is idempotent, so this also follows configurations that reduce every
    microbatch.
    """
    get_fsdp_state = getattr(model, "_get_fsdp_state", None)
    if get_fsdp_state is None:
        raise RuntimeError("replicated FSDP2 gradient synchronization requires a PyTorch FSDPModule root")
    fsdp_state = get_fsdp_state()
    original_callback = fsdp_state._root_post_backward_final_callback
    if getattr(original_callback, "_nemo_replicated_grad_sync", False):
        raise RuntimeError("replicated FSDP2 gradient synchronization is already installed on this root")

    @wraps(original_callback)
    def post_backward_with_replicated_grad_sync() -> None:
        should_sync = _fsdp_requires_gradient_sync(fsdp_state)
        original_callback()
        if should_sync:
            grad_sync.synchronize()

    post_backward_with_replicated_grad_sync._nemo_replicated_grad_sync = True
    fsdp_state._root_post_backward_final_callback = post_backward_with_replicated_grad_sync
    setattr(model, _GRAD_SYNC_ATTR, grad_sync)


def make_fully_shard_with_replicated_parameter_grad_sync(
    root_module: nn.Module,
    parameters: tuple[nn.Parameter, ...],
    mesh: DeviceMesh,
    *,
    fully_shard_fn: Callable[..., nn.Module],
) -> Callable[..., nn.Module]:
    """Wrap ``fully_shard`` to own replicated FP32 gradient synchronization.

    The returned callable is suitable for the repository's recursive FSDP
    traversal. Child units delegate unchanged. When the traversal reaches
    ``root_module``, the wrapper installs one lifecycle hook on the resulting
    FSDP root. This keeps synchronization correct for ordinary custom loops as
    well as recipes using deferred FSDP gradient synchronization.

    Args:
        root_module: Model part that becomes the root FSDP unit.
        parameters: FP32 parameter tensors of arbitrary shape that remain replicated
            over ``mesh``. DTensors may retain independent TP placements.
        mesh: Data-parallel FSDP or HSDP mesh used for gradient synchronization.
        fully_shard_fn: FSDP callable used by the recursive traversal.

    Returns:
        A ``fully_shard``-compatible callable that installs synchronization when
        ``root_module`` is wrapped.
    """
    trainable_ids = {id(parameter) for parameter in parameters if parameter.requires_grad}
    parameter_slots: list[_ParameterSlot] = []
    found_ids: set[int] = set()
    for owner in root_module.modules():
        for name, parameter in owner.named_parameters(recurse=False):
            parameter_id = id(parameter)
            if parameter_id in trainable_ids and parameter_id not in found_ids:
                parameter_slots.append(_ParameterSlot(owner, name))
                found_ids.add(parameter_id)
    missing_ids = trainable_ids - found_ids
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} replicated trainable parameter(s) are not owned by root_module")

    grad_sync = _ReplicatedGradSync(tuple(parameter_slots), mesh)
    installed = False

    @wraps(fully_shard_fn)
    def fully_shard_with_grad_sync(module: nn.Module, **kwargs) -> nn.Module:
        nonlocal installed
        wrapped = fully_shard_fn(module, **kwargs)
        if module is root_module:
            if installed:
                raise RuntimeError("replicated FSDP2 gradient synchronization root was fully sharded more than once")
            _install_fsdp_post_backward_grad_sync(wrapped, grad_sync)
            installed = True
        return wrapped

    return fully_shard_with_grad_sync
