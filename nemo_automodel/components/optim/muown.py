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

"""Muown: Muon directions and Adam neuron magnitudes, using Dion's FSDP2 all-to-all ownership scheme.

Algorithm reference: https://arxiv.org/abs/2605.10797. Numerical conventions follow
the reference implementation's ordinary Muown / newtonschulz5_torch backend.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Generator
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from nemo_automodel.components.optim import muown_triton
from nemo_automodel.shared.import_utils import safe_import

_HAS_DION, _dion = safe_import("dion")
if _HAS_DION:
    from dion.opt_utils import AsyncTask, create_param_batches

_MuonBase = _dion.Muon if _HAS_DION else Optimizer


def _row_sum(value: Tensor, axis: int) -> Tensor:
    """Reduce an input-feature axis and resolve DTensor partial sums.

    Args:
        value: Tensor of shape [..., output, input] or [..., input, output],
            with arbitrary leading matrix-batch dimensions. DTensors may shard
            batch or matrix axes; shapes are global.
        axis: Physical input-feature axis, -1 or -2.

    Returns:
        Tensor with the input-feature axis reduced to size 1. Partial sums are
        replicated over the reduced axis's mesh; other placements are retained.
    """
    result = value.sum(dim=axis, keepdim=True)
    if isinstance(result, DTensor) and any(p.is_partial() for p in result.placements):
        result = result.redistribute(placements=[Replicate() if p.is_partial() else p for p in result.placements])
    return result


def _row_norm(value: Tensor, axis: int, epsilon: float) -> Tensor:
    """Compute per-output-neuron norms without gathering the weight matrix.

    Args:
        value: Tensor of shape [..., output, input] or [..., input, output],
            including globally shaped DTensors as described by _row_sum.
        axis: Physical input-feature axis, -1 or -2.
        epsilon: Minimum returned norm.

    Returns:
        Norm tensor in value's dtype, with the input-feature axis of size 1
        and resolved placements as described by _row_sum.
    """
    norm_input = value if value.dtype == torch.float64 else value.float()
    # A complete input-feature axis can use the reference norm kernel directly.
    # Input-sharded matrices need a global sum of squared local contributions.
    input_sharded = isinstance(value, DTensor) and any(
        isinstance(placement, Shard)
        and placement.dim % value.ndim == axis % value.ndim
        and value.device_mesh.size(i) > 1
        for i, placement in enumerate(value.placements)
    )
    norm = _row_sum(norm_input.square(), axis).sqrt() if input_sharded else norm_input.norm(dim=axis, keepdim=True)
    # A size-one input shard may still produce _NormPartial in DTensor. Resolve
    # it before callers mutate the norm in place, just as _row_sum does.
    if isinstance(norm, DTensor) and any(p.is_partial() for p in norm.placements):
        norm = norm.redistribute(placements=[Replicate() if p.is_partial() else p for p in norm.placements])
    return (norm.clamp_min(epsilon) if epsilon else norm).to(value.dtype)


def _nonzero_magnitude(value: Tensor, epsilon: float) -> Tensor:
    """Preserve signed neuron magnitudes while guarding division by zero.

    Args:
        value: Magnitude tensor of shape [..., output, 1] or [..., 1, output],
            with arbitrary leading batch axes; DTensor placements are preserved.
        epsilon: Minimum absolute magnitude.

    Returns:
        Independent tensor with the same shape, dtype, device and placements.
    """
    # Adam can change a magnitude's sign; do not clamp negative magnitudes to +eps.
    magnitude = value.abs().clamp_min(epsilon)
    return torch.where(value < 0, -magnitude, magnitude)


def _matrix_sharding(param: Tensor) -> tuple[int | None, dist.ProcessGroup | None]:
    """Resolve the communication group for the last two matrix axes.

    Args:
        param: Parameter of global shape [..., rows, columns]. Leading axes
            are matrix batches; DTensors may shard those axes and at most one
            matrix axis. Matrix shards must be even and Partial is unsupported.

    Returns:
        Matrix tensor axis and its process group, or (None, None) when each
        rank already owns complete matrices.
    """
    if not isinstance(param, DTensor):
        return None, None
    if any(p.is_partial() for p in param.placements):
        raise ValueError("Muown parameters must not have Partial placements.")
    matrix_axes = {param.ndim - 2, param.ndim - 1}
    shards = [
        (i, p.dim % param.ndim)
        for i, p in enumerate(param.placements)
        if isinstance(p, Shard) and p.dim % param.ndim in matrix_axes and param.device_mesh.size(i) > 1
    ]
    if len(shards) > 1:
        raise NotImplementedError(
            "Muown supports at most one sharded matrix axis; expert batch axes may also be sharded."
        )
    if not shards:
        return None, None
    mesh_axis, axis = shards[0]
    if param.shape[axis] % param.device_mesh.size(mesh_axis):
        raise ValueError("Muown requires an evenly sharded matrix axis for Dion's all-to-all update.")
    return axis, param.device_mesh.get_group(mesh_axis)


def _newton_schulz(update: Tensor, epsilon: float | Tensor, *, matrix_transposed: bool, steps: int) -> Tensor:
    """Orthogonalize complete local matrices with the reference quintic polynomial.

    Args:
        update: Local tensor of shape [..., output, input], or [..., input,
            output] when matrix_transposed is true. Leading axes are independent
            matrices; this function does not accept matrix-sharded DTensors.
        epsilon: Stabilizer for the matrix Frobenius norm. Dion supplies a
            zero-dimensional CPU tensor; Python scalars are also accepted.
        matrix_transposed: Whether the last axis contains output neurons.
        steps: Number of polynomial iterations.

    Returns:
        Independent contiguous BF16 tensor matching update's shape and axis
        order. A logical output dimension of 3 * input is split into three
        square Q/K/V matrices, following the reference's shape heuristic.
    """
    logical = update.mT if matrix_transposed else update
    shape = logical.shape
    # Match the reference's shape-based fused QKV rule, including its LR scaling.
    if shape[-2] == 3 * shape[-1]:
        logical = logical.unflatten(-2, (3, shape[-1]))
    matrix_shape = logical.shape
    value = logical.bfloat16()
    transpose = value.shape[-2] > value.shape[-1]
    if transpose:
        value = value.mT
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + epsilon)
    value = value.reshape(-1, value.shape[-2], value.shape[-1])
    for _ in range(steps):
        gram = value @ value.mT
        polynomial = torch.baddbmm(gram, gram, gram, beta=-4.7750, alpha=2.0315)
        value = torch.baddbmm(value, polynomial, value, beta=3.4445)
    if transpose:
        value = value.mT
    value = value.reshape(matrix_shape).reshape(shape)
    return (value.mT if matrix_transposed else value).contiguous()


@torch.compile
def _prepare_direction(
    weight: Tensor, magnitude: Tensor, norm: Tensor, gradient: Tensor, axis: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct a direction and apply the weight-normalization Jacobian.

    Args:
        weight: Effective weights [..., output, input], or [..., input, output]
            when axis is -2. DTensor shapes are global.
        magnitude: Signed neuron magnitudes with axis reduced to size 1.
        norm: Direction norms with the same shape and placements as magnitude.
        gradient: Weight gradients matching weight's shape and placements.
        axis: Physical input-feature axis; partial row sums are resolved.

    Returns:
        Tuple (direction, magnitude_gradient, direction_gradient). Direction
        and direction_gradient match weight's shape/placements; magnitude_gradient
        matches magnitude's shape/placements. All returned tensors are independent.
    """
    unit = weight / magnitude
    direction = unit * norm
    magnitude_gradient = _row_sum(gradient * unit, axis)
    direction_gradient = (magnitude / norm) * (gradient - unit * magnitude_gradient)
    return direction, magnitude_gradient, direction_gradient


@torch.compile
def _recompose(weight: Tensor, magnitude: Tensor, direction: Tensor, axis: int) -> Tensor:
    """Compose effective weights with reference-compatible fused arithmetic.

    Args:
        weight: Destination weights [..., output, input], or [..., input,
            output] for axis -2. Mutated in place; DTensor shapes are global.
        magnitude: Signed magnitudes with the input-feature axis reduced to 1.
        direction: Updated directions matching weight's shape and placements.
        axis: Physical input-feature axis.

    Returns:
        Independent row norms with magnitude's shape and placements. The caller
        guards zero rows after this ordinary normalization expression.
    """
    norm = _row_norm(direction, axis, 0.0)
    weight.copy_(magnitude * (direction / norm))
    return norm


class Muown(_MuonBase):
    """Learn per-output-neuron magnitudes with Adam and matrix directions with Muon.

    Uses the existing optional Dion dependency. Leading dimensions of 3D+
    weights are independent matrices (experts). Groups with
    matrix_transposed=True store matrices as [..., input, output]; the default
    is [..., output, input]. Embeddings, biases and heads should be placed in
    algorithm="adamw" or "lion" groups by MuownConfig.

    FSDP communication follows each parameter's DTensor mesh, so dense and expert
    parameters can share an optimizer despite different DP/EP meshes. Only one
    evenly sharded matrix axis is supported. EP-only and replicated matrices are
    orthogonalized locally. FP32 parameter storage is recommended.

    Args:
        params: Parameters or parameter-group dictionaries. Matrix weights have
            global shape [..., output, input], or [..., input, output] for
            matrix_transposed groups; leading axes are independent experts.
            AdamW/Lion groups accept arbitrary parameter shapes. DTensor matrix
            sharding follows _matrix_sharding. Gradients must use matching
            placements. Parameters and optimizer state are updated in-place;
            caller-owned gradients are preserved.
        distributed_mesh: Compatibility argument for Dion; matrix collectives
            use each parameter's actual DTensor mesh.
        lr: Base magnitude LR; direction LR is scaled by matrix dimensions.
        mu: Direction momentum coefficient.
        betas: Adam coefficients for learned magnitudes and scalar fallback.
        weight_decay: Decoupled decay applied to the original weight.
        epsilon: Adam denominator and zero-norm stabilizer.
        nesterov: Enable Nesterov direction momentum.
        ns_steps: Newton-Schulz iteration count.
        ns_epsilon: Newton-Schulz normalization stabilizer.
        muon_update_scale: Nonnegative multiplier for direction updates only.
            Magnitude Adam and auxiliary AdamW/Lion updates are unchanged.
            This run-level setting is read from config, including on resume.
        use_triton: Fuse local FP32 row updates on CUDA. Unsupported layouts,
            wide rows and input-feature shards retain the Torch implementation.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        distributed_mesh: DeviceMesh | dist.ProcessGroup | None = None,
        lr: float = 3e-4,
        mu: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        epsilon: float = 1e-8,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_epsilon: float = 1e-7,
        muon_update_scale: float = 1.0,
        use_triton: bool = False,
    ) -> None:
        if not _HAS_DION:
            raise ImportError(
                "Muown requires the optional Dion dependency. Install the existing optional dependency with uv sync --extra dev."
            )
        if use_triton and not muown_triton.HAVE_TRITON:
            raise ImportError("use_triton=True requires Triton.")
        if ns_steps < 1 or not isinstance(ns_steps, int) or ns_epsilon <= 0:
            raise ValueError("ns_steps must be a positive integer and ns_epsilon must be positive.")
        if not math.isfinite(muon_update_scale) or muon_update_scale < 0:
            raise ValueError("muon_update_scale must be finite and nonnegative.")
        self.muon_update_scale = muon_update_scale
        super().__init__(
            params,
            distributed_mesh=distributed_mesh,
            lr=lr,
            mu=mu,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov,
            adjust_lr=None,
            flatten=False,
        )
        for group in self.param_groups:
            group.setdefault("matrix_transposed", False)
            group.setdefault("ns_steps", ns_steps)
            group.setdefault("ns_epsilon", ns_epsilon)
            group.setdefault("use_triton", use_triton)
            if group["algorithm"] == "adamw":
                group["fused"] = True
            if not isinstance(group["ns_steps"], int) or group["ns_steps"] < 1 or group["ns_epsilon"] <= 0:
                raise ValueError("ns_steps must be a positive integer and ns_epsilon must be positive.")
            if (
                group["lr"] < 0
                or group["weight_decay"] < 0
                or group["epsilon"] <= 0
                or not 0 <= group["mu"] < 1
                or not 0 <= group["beta1"] < 1
                or not 0 <= group["beta2"] < 1
            ):
                raise ValueError("Muown requires nonnegative LR/decay, positive epsilon and momentum/betas in [0, 1).")
            if group["algorithm"] not in ("muon", "adamw", "lion"):
                raise ValueError(f"Unknown Muown parameter-group algorithm: {group['algorithm']}")
            for param in group["params"]:
                if param.is_meta or not param.is_floating_point():
                    raise ValueError("Muown requires materialized floating-point parameters.")
                if group["algorithm"] == "muon":
                    if param.ndim < 2:
                        raise ValueError("Muown matrix groups require parameters with at least two dimensions.")
                    _matrix_sharding(param)
        # Eager state initialization provides DCP with a complete load skeleton,
        # including parameters that have not received a gradient yet.
        with torch.no_grad():
            for group in self.param_groups:
                axis = -2 if group["matrix_transposed"] else -1
                for param in group["params"]:
                    state = self._get_or_initialize_state(param, group["algorithm"])
                    if group["algorithm"] == "adamw":
                        state["step"] = torch.zeros((), device=param.device, dtype=torch.float32)
                    if group["algorithm"] == "muon":
                        magnitude = _row_norm(param, axis, group["epsilon"])
                        state.update(
                            g=magnitude.clone(),
                            v_norm=magnitude.clone(),
                            m_g=torch.zeros_like(magnitude),
                            v_g=torch.zeros_like(magnitude),
                            muown_step=0,
                        )

    def _create_adamw_tasks(
        self, param_groups: list[dict[str, Any]], algo_name: str = "adamw"
    ) -> Generator[AsyncTask, None, None]:
        """Schedule PyTorch AdamW for the non-matrix parameter groups.

        Args:
            param_groups: Dion groups of arbitrary-shaped floating-point
                parameters. DTensors and their gradients/states share the same
                placements; each local shard is updated independently.
            algo_name: Dion's scalar-algorithm name.

        Yields:
            Tasks that update parameters and Adam moments in place.
        """
        for group in param_groups:
            if group["cautious_wd"]:
                yield from super()._create_adamw_tasks([group], algo_name)
            else:
                yield AsyncTask(self._update_adamw_group(group))

    def _update_adamw_group(self, group: dict[str, Any]) -> Generator[None, None, None]:
        """Apply AdamW to local shards using PyTorch's reference arithmetic.

        Args:
            group: AdamW group with arbitrary-shaped parameters; DTensor
                parameters, gradients and moments have matching placements.
                Parameters, moments and per-parameter steps are mutated in place.

        Yields:
            Control after the local AdamW update.
        """
        from torch.optim.adamw import adamw

        params = [param for param in group["params"] if param.grad is not None]
        if not params:
            return
        states = [self.state[param] for param in params]

        def local(value: Tensor) -> Tensor:
            """Return an arbitrary-shaped tensor's local storage view.

            Args:
                value: Tensor of any shape or DTensor with global shape.

            Returns:
                Aliasing local tensor; a DTensor's shape follows its placements.
            """
            return value.to_local() if isinstance(value, DTensor) else value

        adamw(
            params=[local(param) for param in params],
            grads=[local(param.grad) for param in params],
            exp_avgs=[local(state["momentum"]) for state in states],
            exp_avg_sqs=[local(state["variance"]) for state in states],
            max_exp_avg_sqs=[],
            state_steps=[state["step"] for state in states],
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=True,
            amsgrad=False,
            beta1=group["beta1"],
            beta2=group["beta2"],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["epsilon"],
            maximize=False,
        )
        yield

    def _create_muon_tasks(
        self, param_groups: list[dict[str, Any]], algo_name: str = "muon"
    ) -> Generator[AsyncTask, None, None]:
        """Schedule groups with matrix layouts and ownership documented by Muown.

        Args:
            param_groups: Dion group dictionaries containing matrix parameters
                of global shape [..., rows, columns] and Muown hyperparameters.
            algo_name: Dion's matrix-algorithm name.

        Yields:
            Async tasks that update parameters and their optimizer state.
        """
        for group in param_groups:
            # Dion's shape-based batching does not distinguish different meshes.
            # Separate topologies before using its batching and all-to-all code.
            topologies = defaultdict(list)
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise ValueError("Muown does not support sparse gradients.")
                axis, process_group = _matrix_sharding(param)
                mesh = param.device_mesh if isinstance(param, DTensor) else None
                topologies[(mesh, axis, process_group, param.device)].append(param)
            for (_, axis, process_group, _), params in topologies.items():
                batch_size = dist.get_world_size(process_group) if process_group is not None else 1
                for batch in create_param_batches(params, batch_size):
                    yield AsyncTask(self._update_batch(batch, group, axis, process_group, batch_size))

    def _update_batch(
        self,
        params: list[Tensor],
        group: dict[str, Any],
        shard_dim: int | None,
        process_group: dist.ProcessGroup | None,
        world_size: int,
    ) -> Generator[None, None, None]:
        """Update a batch while keeping temporary direction buffers batch-local.

        Args:
            params: Parameters sharing global shape [..., rows, columns],
                placements, dtype, device, and mesh. Last-two-axis ordering is
                given by group["matrix_transposed"]. Gradients are preserved.
            group: Dion hyperparameters and the Muown matrix layout.
            shard_dim: Evenly sharded matrix axis, or None for complete matrices.
            process_group: Group owning the matrix shards, or None.
            world_size: Size of process_group, or 1 for local matrices.

        Yields:
            Control while Dion's collectives are outstanding. Parameters and
            optimizer state are updated in-place before the generator finishes.
        """
        axis = -2 if group["matrix_transposed"] else -1
        states = [self.state[param] for param in params]
        directions, updates, magnitude_gradients, triton_states = [], [], [], []
        for param, state in zip(params, states):
            local_state = None
            if group.get("use_triton", False) and shard_dim != axis % param.ndim:
                tensors = (
                    param.detach(),
                    param.grad,
                    state["momentum"],
                    state["g"],
                    state["v_norm"],
                    state["m_g"],
                    state["v_g"],
                )
                local_tensors = tuple(t.to_local() if isinstance(t, DTensor) else t for t in tensors)
                if muown_triton._supported(local_tensors, group["matrix_transposed"]):
                    local_state = local_tensors
            triton_states.append(local_state)
            if local_state is not None:
                weight, gradient, momentum, magnitude, norm, _, _ = local_state
                direction, update, magnitude_grad = muown_triton._prepare(
                    weight,
                    gradient,
                    momentum,
                    magnitude,
                    norm,
                    transposed=group["matrix_transposed"],
                    mu=group["mu"],
                    epsilon=group["epsilon"],
                    nesterov=group["nesterov"],
                )
                torch.autograd.graph.increment_version(state["momentum"])
                directions.append(direction)
                updates.append(update)
                magnitude_gradients.append(magnitude_grad)
                continue
            direction, magnitude_grad, direction_grad = _prepare_direction(
                param.detach(), _nonzero_magnitude(state["g"], group["epsilon"]), state["v_norm"], param.grad, axis
            )
            directions.append(direction)
            momentum = state["momentum"]
            momentum.mul_(group["mu"]).add_(direction_grad)
            update = direction_grad.add(momentum, alpha=group["mu"]) if group["nesterov"] else momentum.clone()
            updates.append(update.to_local().bfloat16() if isinstance(update, DTensor) else update.bfloat16())
            magnitude_gradients.append(magnitude_grad)
        # Retain Dion's all-to-all ownership scheme, but keep Muown's arithmetic
        # here. Dion's ordinary Muon helper scales an update while still in BF16,
        # which rounds before the FP32 direction addition.
        if shard_dim is not None:
            updates.extend(torch.zeros_like(updates[0]) for _ in range(world_size - len(updates)))
            received = [torch.empty_like(update) for update in updates]
            work = dist.all_to_all(received, updates, group=process_group, async_op=True)
            yield
            work.wait()
            whole_update = torch.cat(received, dim=shard_dim)
            whole_update = _newton_schulz(
                whole_update,
                group["ns_epsilon"],
                matrix_transposed=group["matrix_transposed"],
                steps=group["ns_steps"],
            )
            send = [part.contiguous() for part in whole_update.tensor_split(world_size, dim=shard_dim)]
            work = dist.all_to_all(updates, send, group=process_group, async_op=True)
            yield
            work.wait()
        else:
            updates = [
                _newton_schulz(
                    update,
                    group["ns_epsilon"],
                    matrix_transposed=group["matrix_transposed"],
                    steps=group["ns_steps"],
                )
                for update in updates
            ]
        rows, cols = params[0].shape[-2:]
        if group["matrix_transposed"]:
            rows, cols = cols, rows
        scale = self.muon_update_scale * 0.2 * math.sqrt(cols if rows == 3 * cols else max(rows, cols))
        beta1, beta2 = group["beta1"], group["beta2"]
        for param, state, direction, update, magnitude_grad, local_state in zip(
            params, states, directions, updates, magnitude_gradients, triton_states
        ):
            if local_state is not None:
                weight, _, _, magnitude, norm, first_moment, second_moment = local_state
                state["muown_step"] += 1
                muown_triton._finish(
                    weight,
                    magnitude,
                    norm,
                    first_moment,
                    second_moment,
                    direction,
                    update,
                    magnitude_grad,
                    transposed=group["matrix_transposed"],
                    lr=group["lr"],
                    scale=scale,
                    beta1=beta1,
                    beta2=beta2,
                    step=state["muown_step"],
                    epsilon=group["epsilon"],
                    weight_decay=group["weight_decay"],
                )
                torch.autograd.graph.increment_version([param, state["g"], state["v_norm"], state["m_g"], state["v_g"]])
                continue
            local_direction = direction.to_local() if isinstance(direction, DTensor) else direction
            # add_ applies the scalar in the master-weight dtype. A separate
            # BF16 multiply would introduce another low-precision rounding.
            local_direction.add_(update, alpha=-group["lr"] * scale)
            state["muown_step"] += 1
            step = state["muown_step"]
            state["m_g"].mul_(beta1).add_(magnitude_grad, alpha=1 - beta1)
            state["v_g"].mul_(beta2).addcmul_(magnitude_grad, magnitude_grad, value=1 - beta2)
            denominator = (state["v_g"] / (1 - beta2**step)).sqrt().add_(group["epsilon"])
            state["g"].addcdiv_(state["m_g"] / (1 - beta1**step), denominator, value=-group["lr"])
            previous = param.clone() if group["weight_decay"] else None
            norm = _recompose(param.detach(), state["g"], direction, axis)
            # Guard only degenerate rows, preserving the fused arithmetic for
            # ordinary neurons instead of perturbing every norm's rounding.
            param.copy_(torch.where(norm >= group["epsilon"], param, state["g"] * (direction / group["epsilon"])))
            norm.clamp_min_(group["epsilon"])
            if previous is not None:
                param.add_(previous, alpha=-group["lr"] * group["weight_decay"])
            state["v_norm"].copy_(norm)
            if group["weight_decay"]:
                state["g"].copy_(_row_norm(param, axis, group["epsilon"]))
