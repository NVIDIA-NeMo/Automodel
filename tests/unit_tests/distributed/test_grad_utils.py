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

import math

import pytest
import torch
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed import grad_utils


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


def test_clip_grad_by_total_norm_with_dtensor(monkeypatch):
    """Ensure DTensor gradients go through *to_local_if_dtensor* helper.

    We spy on the helper to assert it was invoked exactly once.  No numerical
    assertions are made because the helper returns a detached copy whose
    changes don't propagate back to the original mock.
    """

    param = _make_param([1.0, -1.0])
    dtensor_mock: DTensor = Mock(spec=DTensor)
    dtensor_mock.detach.return_value = dtensor_mock  # Used inside the util.
    param.grad = dtensor_mock  # type: ignore[assignment]

    call_count = 0

    def _spy(t):
        nonlocal call_count
        call_count += 1
        # Return a clone so mul_ doesn't error on a Mock instance.
        return torch.ones_like(param)

    monkeypatch.setattr(grad_utils, "to_local_if_dtensor", _spy)

    # Should execute without error and invoke the spy.
    grad_utils.clip_grad_by_total_norm_(param, max_grad_norm=1.0, total_norm=5.0)

    assert call_count == 1


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
