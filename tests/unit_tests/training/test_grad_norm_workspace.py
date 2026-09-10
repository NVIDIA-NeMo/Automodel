# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Bounded gradient norm workspace, numerical behavior, and distributed ownership."""

import math
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from nemo_automodel.components.training import utils as training_utils


def _whole_tensor_norm(gradients: list[torch.Tensor], norm_type: float) -> torch.Tensor:
    """Evaluate the pre-chunking regular-tensor equations independently.

    Division by a zero-dimensional FP64 scale keeps nonscalar BF16/FP32
    gradients in their dtype; scalar gradients promote to FP64 as before.

    Args:
        gradients: CPU BF16, FP32, or FP64 gradient tensors of arbitrary shapes,
            including scalar, empty, and noncontiguous tensors. Inputs are read
            without modification using the original dtype promotion rules.
        norm_type: Positive finite norm exponent or positive infinity.

    Returns:
        Independently allocated CPU FP64 scalar containing the norm of the
        concatenated gradients, preserving zero and nonfinite results.
    """
    nonempty = [gradient for gradient in gradients if gradient.numel()]
    maximum = torch.zeros((), dtype=torch.float64)
    for gradient in nonempty:
        maximum = torch.maximum(maximum, gradient.abs().max().double())
    if math.isinf(norm_type):
        return maximum
    scale = torch.where(torch.isfinite(maximum) & maximum.ne(0), maximum, torch.ones_like(maximum))
    power_sum = torch.zeros((), dtype=torch.float64)
    for gradient in nonempty:
        normalized = gradient.abs() / scale
        assert normalized.dtype == (torch.float64 if gradient.ndim == 0 else gradient.dtype)
        powered = normalized.square() if norm_type == 2 else normalized.pow(norm_type)
        power_sum += powered.sum(dtype=torch.float64)
    return maximum * power_sum.pow(1 / norm_type)


def _check_clipping_and_sgd(gradients: list[torch.Tensor], norm_type: float, max_norm: float = 0.75) -> torch.Tensor:
    """Compare clipping and one SGD update against whole-tensor reference math.

    Args:
        gradients: CPU BF16, FP32, or FP64 tensors of arbitrary shapes and
            nonoverlapping strides, including scalar and empty tensors. Fresh
            parameter gradients copy their values and strides; inputs are not
            mutated or used as optimizer-owned storage.
        norm_type: Positive finite norm exponent or positive infinity.
        max_norm: Maximum gradient norm applied before the SGD update.

    Returns:
        CPU FP64 scalar containing the production pre-clipping norm, after
        asserting agreement of the norm, clipped gradients, and updated weights.
    """
    expected_norm = _whole_tensor_norm(gradients, norm_type)
    coefficient = (max_norm / (expected_norm + 1e-6)).clamp(max=1.0)
    expected_gradients = [(gradient * coefficient).to(gradient.dtype) for gradient in gradients]
    parameters = [nn.Parameter(torch.full_like(gradient, 0.5)) for gradient in gradients]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = torch.empty_strided(gradient.shape, gradient.stride(), dtype=gradient.dtype)
        parameter.grad.copy_(gradient)
        assert parameter.grad.stride() == gradient.stride()
    expected_parameters = [
        torch.add(parameter.detach(), gradient, alpha=-0.125)
        for parameter, gradient in zip(parameters, expected_gradients)
    ]
    actual_norm = training_utils._clip_grad_norm_impl(parameters, max_norm, norm_type, foreach=True)
    # Only FP64 reduction association changes; elementwise operations keep
    # their old dtypes. The tolerance is far below FP32/BF16 rounding precision.
    torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-12, atol=0, equal_nan=True)
    for parameter, expected in zip(parameters, expected_gradients):
        torch.testing.assert_close(
            parameter.grad, expected, rtol=max(1e-12, 2 * torch.finfo(expected.dtype).eps), atol=0, equal_nan=True
        )
    torch.optim.SGD(parameters, lr=0.125).step()
    for parameter, expected in zip(parameters, expected_parameters):
        torch.testing.assert_close(
            parameter, expected, rtol=max(1e-12, 2 * torch.finfo(expected.dtype).eps), atol=0, equal_nan=True
        )
    return actual_norm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_chunked_norm_clip_and_sgd_match_original_dtype_math(dtype, norm_type, monkeypatch):
    monkeypatch.setattr(training_utils, "_GRAD_NORM_CHUNK_NUMEL", 11)
    values = torch.linspace(-3.7, 5.3, 70, dtype=dtype).reshape(7, 10)
    gradients = [values.t(), values[:, ::2], torch.tensor(-0.375, dtype=dtype), torch.empty(0, 3, dtype=dtype)]
    assert not gradients[0].is_contiguous() and not gradients[1].is_contiguous()
    _check_clipping_and_sgd(gradients, norm_type)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_norm_crosses_actual_four_million_element_budget(dtype):
    assert training_utils._GRAD_NORM_CHUNK_NUMEL == 4 * 1024 * 1024
    # A nonuniform short tail detects lost/duplicated elements between chunks.
    size = training_utils._GRAD_NORM_CHUNK_NUMEL + 17
    gradient = torch.full((size,), 3.0, dtype=dtype)
    gradient[1::3] = -4.0
    gradient[-17:] = torch.arange(17, dtype=dtype) - 9
    _check_clipping_and_sgd([gradient], 2.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_extreme_finite_gradients_remain_finite(dtype, monkeypatch):
    monkeypatch.setattr(training_utils, "_GRAD_NORM_CHUNK_NUMEL", 3)
    large, small = torch.finfo(dtype).max / 8, torch.finfo(dtype).tiny * 4
    gradients = [torch.tensor([large, small, -large, 0, large / 2, -small, 2], dtype=dtype)]
    norm = _check_clipping_and_sgd(gradients, 3.0)
    assert torch.isfinite(norm)


@pytest.mark.parametrize("value", [0.0, float("inf"), float("nan")])
@pytest.mark.parametrize("norm_type", [2.0, float("inf")])
def test_zero_and_nonfinite_results_and_error_policy(value, norm_type, monkeypatch):
    monkeypatch.setattr(training_utils, "_GRAD_NORM_CHUNK_NUMEL", 2)
    gradient = torch.zeros(7)
    gradient[4] = value
    _check_clipping_and_sgd([gradient, torch.empty(0)], norm_type)
    parameter = nn.Parameter(torch.ones_like(gradient))
    parameter.grad = gradient.clone()
    if not math.isfinite(value):
        with pytest.raises(RuntimeError, match="non-finite"):
            training_utils._clip_grad_norm_impl([parameter], 1.0, norm_type, error_if_nonfinite=True)
        # Reject before clipping, including values beyond the first chunk.
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0, equal_nan=True)


class _AllocationAudit(TorchDispatchMode):
    """Record newly allocated operator outputs, excluding aliases of inputs."""

    def __init__(self):
        super().__init__()
        self.allocations = []

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        """Execute an operator and record outputs with newly allocated storage.

        Args:
            func: PyTorch operator overload being dispatched.
            types: Tensor subclasses participating in this dispatch.
            args: Nested positional arguments, which may contain tensors of
                arbitrary shapes, dtypes, devices, strides, and aliasing layouts.
            kwargs: Nested keyword arguments with the same tensor contract as
                args, or None when the operator has no keyword arguments.

        Returns:
            The unmodified operator result, including any tensors or nested
            containers of tensors. Output shapes, dtypes, devices, and storage
            aliases are those returned by func; the audit adds no tensor copies.
        """
        kwargs = kwargs or {}
        input_storage = {
            tensor.untyped_storage().data_ptr()
            for tensor in tree_leaves((args, kwargs))
            if isinstance(tensor, torch.Tensor)
        }
        output = func(*args, **kwargs)
        for tensor in tree_leaves(output):
            if isinstance(tensor, torch.Tensor) and tensor.numel():
                if tensor.untyped_storage().data_ptr() not in input_storage:
                    self.allocations.append((str(func), tensor.numel(), tensor.element_size(), tensor.dtype))
        return output


def test_real_strided_gradient_allocations_are_bounded(record_testsuite_property):
    budget = training_utils._GRAD_NORM_CHUNK_NUMEL
    gradient = torch.full((2, budget // 2 + 13), 3.0, dtype=torch.bfloat16).t()
    gradient[::3, 1] = 4
    assert gradient.numel() > budget and not gradient.is_contiguous()
    parameter = nn.Parameter(torch.zeros_like(gradient))
    parameter.grad = gradient
    original_storage = gradient.untyped_storage().data_ptr()
    expected = _whole_tensor_norm([gradient], 2.0)
    audit = _AllocationAudit()
    # The dispatcher observes full-size flatten/clone mistakes. The CPU
    # profiler also observes internal FP64 reduction-input conversions, which
    # need not appear as separate Python dispatch operations.
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as profiler:
        with audit:
            actual = training_utils._clip_grad_norm_impl([parameter], float("inf"), 2.0, foreach=False)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)
    assert parameter.grad is gradient
    assert gradient.untyped_storage().data_ptr() == original_storage
    assert all(elements <= budget for _, elements, _, _ in audit.allocations)
    assert max(elements for _, elements, _, _ in audit.allocations) == budget
    allocation_events = [event for event in profiler.events() if event.cpu_memory_usage > 0]
    largest_allocation = max(event.cpu_memory_usage for event in allocation_events)
    assert 0 < largest_allocation <= budget * 8
    record_testsuite_property("largest_cpu_operator_allocation_bytes", largest_allocation)
    record_testsuite_property("largest_python_visible_allocation_elements", max(row[1] for row in audit.allocations))
    record_testsuite_property("norm_chunk_elements", budget)


def _distributed_norm_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        shard_mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp_shard_cp",))
        owner_mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("ep",))
        shard_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        owner_values = torch.tensor([2.0, -1.0, 3.0, 0.0, 4.0], dtype=torch.float64)
        replicated_values = torch.tensor([6.0, -2.0, 1.0], dtype=torch.float64)
        partial_local = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64) * (rank + 1)
        singleton_local = torch.tensor([7.0], dtype=torch.float64) if rank == 0 else torch.empty(0, dtype=torch.float64)
        for norm_type in (2.0, float("inf")):
            model = nn.Module()
            expected_local = {}

            def register(
                name: str,
                values: torch.Tensor,
                shape: int,
                placement: Shard | Replicate | Partial,
                mesh: DeviceMesh,
                normalized: torch.Tensor | None = None,
            ) -> nn.Parameter:
                """Register a distributed parameter and its local gradient data.

                Args:
                    name: Parameter name in the enclosing test model.
                    values: CPU FP64 local gradient vector [local_elements].
                        Shards may be unequal or empty; replicated and partial
                        vectors have the full global length. Values are copied.
                    shape: Global one-dimensional parameter length.
                    placement: Placement of both parameter and gradient.
                    mesh: One-dimensional CPU device mesh for the DTensors.
                    normalized: Optional CPU FP64 vector matching values' shape
                        with the expected gradient after owner averaging, before
                        clipping. The expectation is retained without mutation;
                        omitting it uses values as the expected local gradient.

                Returns:
                    Registered zero-initialized DTensor Parameter with global
                    shape [shape], the requested placement, and an independently
                    stored gradient copied from values.
                """
                parameter = nn.Parameter(
                    DTensor.from_local(
                        torch.zeros_like(values), mesh, [placement], run_check=False, shape=(shape,), stride=(1,)
                    )
                )
                parameter.grad = DTensor.from_local(
                    values.clone(), mesh, [placement], run_check=False, shape=(shape,), stride=(1,)
                )
                model.register_parameter(name, parameter)
                expected_local[name] = values if normalized is None else normalized
                return parameter

            register("sharded", shard_values.chunk(2)[rank], 5, Shard(0), shard_mesh)
            register("singleton", singleton_local, 1, Shard(0), shard_mesh)
            register("replicated", replicated_values, 3, Replicate(), shard_mesh)
            register("partial", partial_local, 4, Partial(), shard_mesh)
            owner_local = owner_values.chunk(2)[rank]
            owner = register("owner", owner_local * 2, 5, Shard(0), owner_mesh, normalized=owner_local)
            owner._nemo_model_owned_grad_divisor = 2.0
            full_values = torch.cat(
                [
                    shard_values,
                    torch.tensor([7.0]),
                    replicated_values,
                    torch.tensor([3.0, 6.0, 9.0, 12.0]),
                    owner_values,
                ]
            ).double()
            expected_norm = torch.linalg.vector_norm(full_values, ord=norm_type)
            coefficient = (0.75 / (expected_norm + 1e-6)).clamp(max=1.0)
            # The small budget creates different chunk counts on the ranks;
            # the singleton additionally gives one rank no local elements.
            # The owner mesh also has the EP axis, so this checks that the
            # owner divisor takes precedence rather than being applied twice.
            with patch.object(training_utils, "_GRAD_NORM_CHUNK_NUMEL", 2):
                actual = training_utils.scale_grads_and_clip_grad_norm(
                    0.75,
                    [model],
                    norm_type=norm_type,
                    foreach=False,
                    moe_mesh=owner_mesh,
                    ep_axis_name="ep",
                    dp_group_size=2,
                )
            torch.testing.assert_close(actual, expected_norm, rtol=1e-12, atol=0)
            for name, parameter in model.named_parameters():
                assert isinstance(parameter.grad, DTensor)
                assert parameter.grad.placements == parameter.placements
                torch.testing.assert_close(
                    parameter.grad.to_local(), expected_local[name] * coefficient, rtol=1e-12, atol=0
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_gloo_shard_replicate_partial_and_owner_norms(tmp_path):
    mp.spawn(_distributed_norm_worker, args=(str(tmp_path / "norm-workspace-pg"),), nprocs=2, join=True)
