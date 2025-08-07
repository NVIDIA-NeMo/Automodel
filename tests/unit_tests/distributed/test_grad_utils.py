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

from typing import Iterable, List
from unittest.mock import Mock

import types
import functools
import sys
import threading
from functools import wraps, partial

import queue

import math

import pytest
import torch
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed import grad_utils
import torch.distributed as c10d


TIMEOUT_DEFAULT = 5
DEFAULT_WORLD_SIZE = 1


class _DummyWorld:
    """Placeholder world object to satisfy internal checks."""


class MultiThreadedTestCase:  # minimal shim
    exception_queue: "queue.Queue[tuple[int, tuple[type, BaseException, object]]]" = queue.Queue()

    @staticmethod
    def _join_threads(threads: list[threading.Thread], func):  # noqa: D401
        for t in threads:
            t.join(TIMEOUT_DEFAULT)
        if not MultiThreadedTestCase.exception_queue.empty():
            rank, exc_info = MultiThreadedTestCase.exception_queue.get()
            raise exc_info[1]


class ProcessLocalGroup:  # stub
    @staticmethod
    def exception_handle(_ex):  # noqa: D401
        pass


def _install_threaded_pg():
    """Return dummy world handle – avoids actual pg creation."""

    return _DummyWorld()


# modified from https://github.com/pytorch/pytorch/blob/3cf7b4024ef83e44e9ae223dbff7c7ab68240cb2/torch/testing/_internal/common_distributed.py#L1128
def spawn_threads_and_init_comms(func=None, *, timeout=5, world_size=1):
    """Decorator that initialises a (stubbed) threaded process group before test."""

    if func is None:
        return partial(spawn_threads_and_init_comms, timeout=timeout, world_size=world_size)

    def _run_test_method_with_multi_threads(world_size: int, callback):

        world = _install_threaded_pg()
        global_store = c10d.HashStore()

        def worker(rank):
            c10d.init_process_group(backend="threaded", rank=rank, world_size=world_size, store=global_store)
            try:
                callback()
            except BaseException as ex:  # noqa: B902
                MultiThreadedTestCase.exception_queue.put((rank, sys.exc_info()))
                ProcessLocalGroup.exception_handle(ex)
            finally:
                c10d.destroy_process_group()

        threads = []
        for r in range(world_size):
            t = threading.Thread(target=worker, args=(r,))
            t.start()
            threads.append(t)

        return threads

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: D401
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        try:
            threads = _run_test_method_with_multi_threads(
                world_size, lambda: func(self, *args, **kwargs)
            )
            # join and error handling
            MultiThreadedTestCase._join_threads(threads, func)
        finally:
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

    return wrapper

@pytest.fixture(autouse=True)
def _patch_distributed(monkeypatch):
    """Neutralise *torch.distributed* primitives that require initialisation.

    *grad_utils.get_grad_norm* invokes *torch.distributed.all_reduce* on the
    supplied process groups.  Calling this without having initialised a
    process-group backend raises an error.  For unit-testing the pure math we
    only require that the function is *called* – its result is ignored because
    we are in a single-process environment.  We patch it to a lightweight
    no-op.
    """

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    # Stub out process-group initialisation so *DeviceMesh* does not error.
    import importlib
    c10d = importlib.import_module("torch.distributed.distributed_c10d")

    monkeypatch.setattr(c10d, "init_process_group", lambda *a, **k: None, raising=False)

    # Ensure a default process group object exists so *_get_default_group* succeeds.
    if getattr(c10d, "_default_pg", None) is None:
        c10d._default_pg = object()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _patch_tensor_cuda(monkeypatch):
    """Ensure *.cuda()* on a tensor is a cheap no-op on CPU-only boxes.

    The implementation in *grad_utils.get_grad_norm* unconditionally moves the
    accumulator tensor to CUDA.  On systems where CUDA is unavailable this
    would raise an error.  We monkey-patch the method to just return *self* so
    that the remainder of the code path can be exercised.
    """

    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self, raising=False)

def _make_param(data: Iterable[float] | torch.Tensor, *, requires_grad: bool = True) -> torch.Tensor:
    """Helper constructing a parameter tensor and attaching matching gradient."""

    tensor = torch.tensor(list(data) if not isinstance(data, torch.Tensor) else data, dtype=torch.float32)
    tensor.requires_grad_(requires_grad)
    tensor.grad = tensor.clone().detach() if requires_grad else None  # type: ignore[attr-defined]
    return tensor

@pytest.mark.parametrize(
    "max_grad_norm,total_norm,scaling_expected", [(1.0, 5.0, 0.2), (10.0, 5.0, 1.0)],
)
def test_clip_grad_by_total_norm_scaling(max_grad_norm: float, total_norm: float, scaling_expected: float):
    """Gradients should be scaled **iff** *clip_coeff < 1*.

    Two parameters are constructed with known gradients to make it trivial to
    check the post-call values.
    """

    p1 = _make_param([3.0, 4.0])  # |grad| = 5
    p2 = _make_param([1.0, 2.0])  # |grad| = sqrt(5)

    # Keep copies for comparison after in-place modification.
    g1_before, g2_before = p1.grad.clone(), p2.grad.clone()

    grad_utils.clip_grad_by_total_norm_([p1, p2], max_grad_norm=max_grad_norm, total_norm=total_norm)

    assert torch.allclose(p1.grad, g1_before * scaling_expected)
    assert torch.allclose(p2.grad, g2_before * scaling_expected)


def test_clip_grad_by_total_norm_handles_none_gradients():
    """Parameters without *.grad* must be ignored without raising."""

    p1 = _make_param([1.0])
    p2 = _make_param([2.0])
    p2.grad = None  # type: ignore[assignment]

    # Should not raise.
    grad_utils.clip_grad_by_total_norm_([p1, p2], max_grad_norm=1.0, total_norm=1.0)


def test_clip_grad_by_total_norm_single_tensor_input():
    """Function accepts a lone tensor in place of an iterable."""

    param = _make_param([2.0, 2.0])
    original_grad = param.grad.clone()

    grad_utils.clip_grad_by_total_norm_(param, max_grad_norm=2.0, total_norm=4.0)

    scaling = 2.0 / (4.0 + 1e-6)
    assert torch.allclose(param.grad, original_grad * scaling)

# The threaded PG wrapper ensures *DeviceMesh* can operate without a real backend.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for *inf* norm path.")
@spawn_threads_and_init_comms(world_size=2)
def test_clip_grad_by_total_norm_with_dtensor():
    """Integration test exercising *clip_grad_by_total_norm_* with real DTensor."""

    if not torch.cuda.is_available():
        pytest.skip("DTensor path requires CUDA device.")

    # Parameter itself is irrelevant – only its gradient is used/modified.
    param = _make_param([0.0, 0.0])

    from torch.distributed._tensor import DeviceMesh, Replicate

    # Build a minimal 1-D mesh on CPU to avoid NCCL initialisation.
    mesh = DeviceMesh("cuda", torch.arange(2))  # size 1 mesh is communication-free.

    local_grad = torch.tensor([1.0, -1.0], dtype=torch.float32, device="cuda")
    from_local_dt = DTensor.from_local(local_grad, mesh, [Replicate()])
    param.grad = from_local_dt  # type: ignore[assignment]

    total_norm = torch.norm(local_grad).item()
    max_grad_norm = total_norm * 0.5  # Force scaling (< 1)
    expected_coeff = max_grad_norm / (total_norm + 1e-6)

    grad_utils.clip_grad_by_total_norm_(param, max_grad_norm=max_grad_norm, total_norm=total_norm)

    assert torch.allclose(param.grad._local_tensor, local_grad * expected_coeff)  # type: ignore[attr-defined]


def _expected_l2_norm(*grads: List[torch.Tensor | torch.Tensor]):  # noqa: D401 – helper
    """Compute reference L2-norm used for assertion."""

    squared_sum = sum(torch.norm(g, 2) ** 2 for g in grads)
    return math.sqrt(squared_sum)


@pytest.mark.parametrize("norm_type", [2, 2.0])
def test_get_grad_norm_l2(norm_type: int | float):
    """Numerical correctness for p-norm where *p = 2* (the common case)."""

    # Parameters with deterministic gradients.
    p1 = _make_param([3.0, 4.0])  # |grad| = 5
    p2 = _make_param([1.0, 2.0])  # |grad| = sqrt(5)

    expected = _expected_l2_norm(p1.grad, p2.grad)  # type: ignore[arg-type]

    # Dummy process groups – only identity semantics required.
    dp_group = Mock(name="dp_group")
    tp_group = Mock(name="tp_group")

    out = grad_utils.get_grad_norm([p1, p2], dp_group, tp_group, norm_type=norm_type)

    assert math.isclose(out, expected, rel_tol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for *inf* norm path.")
def test_get_grad_norm_inf():
    """Infinity-norm path allocates a CUDA tensor – run only when CUDA exists."""

    param = _make_param([-3.0, 7.0])
    expected = torch.abs(param.grad).max().item()  # type: ignore[arg-type]

    out = grad_utils.get_grad_norm(param, Mock(), Mock(), norm_type=math.inf)

    assert pytest.approx(out) == expected

