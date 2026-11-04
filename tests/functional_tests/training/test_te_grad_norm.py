# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""CUDA norm/clipping checks for TE and distributed TE/Triton backends."""

import copy
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from nemo_automodel.components.training import utils
from nemo_automodel.shared.import_utils import safe_import_te

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not safe_import_te()[0],
    reason="Requires CUDA and Transformer Engine",
)


def _reference_norm(gradients):
    return torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(g, dtype=torch.complex128 if g.is_complex() else torch.float64)
                for g in gradients
            ]
        )
    )


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided", "offset"])
@pytest.mark.parametrize("te_available", [True, False])
def test_norm_clipped_gradients_and_optimizer_step(
    layout: str, te_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare norm, clipping, and an SGD update to an independent FP64 reference."""
    if not te_available:
        monkeypatch.setattr(utils, "safe_import_te", lambda: (False, None))
    torch.manual_seed(23)
    parameters, references, gradients = [], [], []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128):
        gradient = torch.randn(33, 65, device="cuda", dtype=dtype)
        if layout == "transpose":
            gradient = gradient.t()
        elif layout == "strided":
            gradient = gradient[:, ::2]
        elif layout == "offset":
            gradient = gradient[1:]
        parameter = torch.nn.Parameter(torch.randn_like(gradient))
        reference = torch.nn.Parameter(parameter.detach().clone())
        parameter.grad = gradient
        parameters.append(parameter)
        references.append(reference)
        gradients.append(gradient.clone())
    # Empty gradients and parameters without a gradient must not affect the norm.
    empty = torch.nn.Parameter(torch.empty(0, device="cuda"))
    empty.grad = torch.empty_like(empty)
    parameters.extend([empty, torch.nn.Parameter(torch.ones(2, device="cuda"))])
    expected = _reference_norm(gradients)
    coefficient = (0.7 / (expected + 1e-6)).clamp(max=1.0)
    for parameter, gradient in zip(references, gradients):
        parameter.grad = gradient * coefficient
    # Strided/transposed gradients use the reference path without loading TE.
    if not te_available and layout in ("contiguous", "offset"):
        with pytest.raises(RuntimeError, match="backend requested but unavailable"):
            utils._clip_grad_norm_impl(parameters, 0.7, foreach=True, grad_norm_backend="te")
        return
    observed = utils._clip_grad_norm_impl(parameters, 0.7, foreach=True, grad_norm_backend="te")
    torch.testing.assert_close(observed, expected, rtol=2e-6, atol=0)
    for parameter, reference in zip(parameters, references):
        torch.testing.assert_close(
            parameter.grad,
            reference.grad,
            rtol=8e-3 if parameter.dtype == torch.bfloat16 else 1e-3 if parameter.dtype == torch.float16 else 2e-6,
            atol=1e-7,
        )
    torch.optim.SGD(parameters, lr=0.1).step()
    torch.optim.SGD(references, lr=0.1).step()
    for parameter, reference in zip(parameters, references):
        torch.testing.assert_close(
            parameter,
            reference,
            rtol=8e-3 if parameter.dtype == torch.bfloat16 else 1e-3 if parameter.dtype == torch.float16 else 2e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize("value", [0.0, 1e-30, 1e-22, 6e31])
def test_finite_extreme_gradients(value):
    """TE's FP32 squared sum must not turn a representable norm into zero/inf."""
    parameter = torch.nn.Parameter(torch.zeros(16, device="cuda"))
    gradient = torch.full_like(parameter, value)
    parameter.grad = gradient.clone()
    expected = torch.linalg.vector_norm(gradient, dtype=torch.float64)
    observed = utils._clip_grad_norm_impl(parameter, 1.0, error_if_nonfinite=True, grad_norm_backend="te")
    torch.testing.assert_close(observed, expected, rtol=2e-6, atol=0)
    torch.testing.assert_close(parameter.grad, gradient * (1.0 / (expected + 1e-6)).clamp(max=1.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_gradients_raise_before_mutation(value):
    parameter = torch.nn.Parameter(torch.zeros(16, device="cuda"))
    parameter.grad = torch.ones_like(parameter)
    parameter.grad[4] = value
    original = parameter.grad.clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        utils._clip_grad_norm_impl(parameter, 1.0, error_if_nonfinite=True, grad_norm_backend="te")
    torch.testing.assert_close(parameter.grad, original, equal_nan=True)


@pytest.mark.parametrize("value", [1e-200, 6e200])
def test_finite_float64_extremes(value):
    parameter = torch.nn.Parameter(torch.zeros(16, device="cuda", dtype=torch.float64))
    parameter.grad = torch.full_like(parameter, value)
    expected = torch.tensor(4.0 * abs(value), device="cuda", dtype=torch.float64)
    observed = utils._clip_grad_norm_impl(parameter, 1.0, error_if_nonfinite=True, grad_norm_backend="te")
    torch.testing.assert_close(observed, expected, rtol=1e-14, atol=0)
    torch.testing.assert_close(
        parameter.grad, torch.full_like(parameter, value) * (1.0 / (expected + 1e-6)).clamp(max=1.0)
    )


def test_small_gradient_mixture():
    """Subnormal squares can disappear even when TE returns a nonzero norm."""
    parameter = torch.nn.Parameter(torch.zeros(101, device="cuda"))
    gradient = torch.full_like(parameter, 1e-19)
    gradient[0] = 1e-17
    parameter.grad = gradient.clone()
    expected = torch.linalg.vector_norm(gradient, dtype=torch.float64)
    observed = utils._clip_grad_norm_impl(parameter, 1.0, error_if_nonfinite=True, grad_norm_backend="te")
    torch.testing.assert_close(observed, expected, rtol=2e-6, atol=0)
    torch.testing.assert_close(parameter.grad, gradient)


def _distributed_worker(rank: int, init_file: str, grad_norm_backend: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=f"file://{init_file}",
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh("cuda", (2,))
        for placement, shape in [
            (Shard(0), (7, 5)),
            (Shard(1), (7, 5)),
            (Shard(0), (1, 5)),
            (Replicate(), (7, 5)),
            (Partial(), (7, 5)),
        ]:
            full = torch.arange(1, 1 + shape[0] * shape[1], device="cuda", dtype=torch.float32).reshape(shape)
            if isinstance(placement, Partial):
                parameter = torch.nn.Parameter(DTensor.from_local(torch.zeros_like(full), mesh, [placement]))
                # Contributions 1/3 and 2/3 must be summed BEFORE their norm.
                parameter.grad = DTensor.from_local(full * ((rank + 1) / 3), mesh, [placement])
            else:
                parameter = torch.nn.Parameter(distribute_tensor(torch.zeros_like(full), mesh, [placement]))
                parameter.grad = distribute_tensor(full.clone(), mesh, [placement])
            expected = torch.linalg.vector_norm(full, dtype=torch.float64)
            observed = utils._clip_grad_norm_impl([parameter], 0.7, foreach=True, grad_norm_backend=grad_norm_backend)
            torch.testing.assert_close(observed, expected, rtol=2e-6, atol=0)
            torch.testing.assert_close(
                parameter.grad.full_tensor(), full * (0.7 / (expected + 1e-6)), rtol=2e-6, atol=1e-7
            )

        # Different local tensor counts and TE eligibility must keep collective order.
        parameters, local_gradients = [], []
        for _ in range(rank + 1):
            gradient = torch.arange(1, 13, device="cuda", dtype=torch.float32).reshape(3, 4)
            if rank == 1:
                gradient = gradient.t().contiguous().t()
            parameter = torch.nn.Parameter(
                DTensor.from_local(
                    torch.zeros_like(gradient),
                    mesh,
                    [Shard(0)],
                    shape=torch.Size((6, 4)),
                    stride=(4, 1),
                    run_check=False,
                )
            )
            parameter.grad = DTensor.from_local(
                gradient.clone(),
                mesh,
                [Shard(0)],
                shape=torch.Size((6, 4)),
                stride=(4, 1),
                run_check=False,
            )
            parameters.append(parameter)
            local_gradients.append(gradient)
        expected = _reference_norm(local_gradients).square()
        dist.all_reduce(expected)
        expected = expected.sqrt()
        observed = utils._clip_grad_norm_impl(parameters, 0.7, foreach=True, grad_norm_backend=grad_norm_backend)
        torch.testing.assert_close(observed, expected, rtol=2e-6, atol=0)
        for parameter, gradient in zip(parameters, local_gradients):
            torch.testing.assert_close(
                parameter.grad.to_local(), gradient * (0.7 / (expected + 1e-6)), rtol=2e-6, atol=1e-7
            )

        # Pipeline stages own distinct parameters, so their squared norms must add.
        parameter = torch.nn.Parameter(torch.zeros(5, device="cuda"))
        parameter.grad = torch.full_like(parameter, rank + 1.0)
        observed = utils._clip_grad_norm_impl([parameter], 0.7, pp_mesh=mesh, grad_norm_backend=grad_norm_backend)
        torch.testing.assert_close(observed, torch.tensor(5.0, device="cuda", dtype=torch.float64))

        # Real FSDP2 backward and optimizer update, with different examples per rank.
        from torch.distributed.fsdp import fully_shard

        torch.manual_seed(123)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)).cuda()
        reference = copy.deepcopy(model)
        fully_shard(model, mesh=mesh)
        inputs = torch.randn(6, 8, device="cuda")
        reference(inputs).square().mean().backward()
        model(inputs.chunk(2)[rank]).square().mean().backward()
        expected = _reference_norm([p.grad for p in reference.parameters()])
        observed = utils._clip_grad_norm_impl(
            model.parameters(), 0.1, foreach=True, grad_norm_backend=grad_norm_backend
        )
        torch.testing.assert_close(observed, expected, rtol=2e-6, atol=1e-8)
        coefficient = (0.1 / (expected + 1e-6)).clamp(max=1.0)
        for parameter in reference.parameters():
            parameter.grad.mul_(coefficient)
        for parameter, original in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad.full_tensor(), original.grad, rtol=3e-6, atol=1e-8)
        torch.optim.SGD(model.parameters(), lr=0.2).step()
        torch.optim.SGD(reference.parameters(), lr=0.2).step()
        for parameter, original in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.full_tensor(), original, rtol=2e-6, atol=1e-8)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("grad_norm_backend", ["te", "triton"])
def test_distributed_norm_and_fsdp_update(tmp_path: Path, grad_norm_backend: str) -> None:
    mp.spawn(_distributed_worker, args=(str(tmp_path / "init"), grad_norm_backend), nprocs=2, join=True)


@pytest.mark.parametrize("backend", [None, "triton", "te"])
@pytest.mark.parametrize("torch_fast_path", [False, True])
def test_grad_norm_backend_selection(
    backend: str | None, torch_fast_path: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TE is selected explicitly; default clipping uses main's Triton backend."""
    model = torch.nn.Linear(2, 1, bias=False, device="cuda")
    initial = model.weight.detach().clone()
    gradient = torch.tensor([[3.0, 4.0]], device="cuda")
    model.weight.grad = gradient.clone()
    local_te_norm = MagicMock(wraps=utils._local_te_l2_norm)
    monkeypatch.setattr(utils, "_local_te_l2_norm", local_te_norm)
    options = {} if backend is None else {"grad_norm_backend": backend}
    norm = utils.scale_grads_and_clip_grad_norm(1.0, [model], use_torch_clip_grad_norm=torch_fast_path, **options)
    assert bool(local_te_norm.call_count) is (backend == "te")
    torch.testing.assert_close(norm.double(), torch.tensor(5.0, dtype=torch.float64, device="cuda"))
    expected_gradient = gradient * (1.0 / (5.0 + 1e-6))
    torch.testing.assert_close(model.weight.grad, expected_gradient, rtol=2e-6, atol=1e-7)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.testing.assert_close(model.weight, initial - 0.1 * expected_gradient, rtol=2e-6, atol=1e-7)
