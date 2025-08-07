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
from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.tensor_utils import (
    get_cpu_state_dict,
    to_cpu,
    to_local_if_dtensor,
)


@pytest.fixture(autouse=True)
def _patch_cuda_synchronize(monkeypatch):
    """Disable *torch.cuda.synchronize* during unit-tests.

    The real call is unnecessary for correctness of these utilities and may fail
    on CPU-only build of PyTorch.  We patch it to a no-op for the duration of each
    test.
    """

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


class _FakeDevice:  # noqa: D401, pylint: disable=too-few-public-methods
    """Lightweight replacement for *torch.device* with arbitrary *type*."""

    def __init__(self, dev_type: str):
        self.type = dev_type

    # *torch.device* stringification is relied upon in error message paths.
    def __str__(self):  # noqa: D401
        return self.type

def test_to_local_if_dtensor_returns_local_tensor():
    """Ensure a *DTensor* is converted to its local shard."""

    local = torch.randn(2, 2)
    dtensor_mock: DTensor = Mock(spec=DTensor)
    dtensor_mock.to_local.return_value = local

    out = to_local_if_dtensor(dtensor_mock)

    assert out is local

def test_to_local_if_dtensor_noop_for_regular_tensor():
    """Regular tensors should be returned unmodified."""

    tensor = torch.randn(3, 3)
    assert to_local_if_dtensor(tensor) is tensor

@pytest.mark.parametrize("tensor_device", (
    ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
))
def test_to_cpu_tensor(tensor_device: str):
    """*to_cpu* should return a CPU copy of plain tensors regardless of device."""

    tensor = torch.ones(4, device=tensor_device)
    out = to_cpu(tensor)

    assert torch.allclose(out, tensor.cpu())
    assert out.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this path.")
def test_to_cpu_dtensor_cuda():
    """DTensor on CUDA device should be materialised and moved to CPU."""

    full_tensor_gpu = torch.arange(5, device="cuda")

    dtensor_mock: DTensor = Mock(spec=DTensor)
    dtensor_mock.device = torch.device("cuda:0")
    dtensor_mock.full_tensor.return_value = full_tensor_gpu

    out = to_cpu(dtensor_mock)

    expected = full_tensor_gpu.cpu()
    assert torch.allclose(out, expected)
    assert out.device.type == "cpu"


def test_to_cpu_dtensor_cpu():
    """DTensor already on CPU should return its local tensor unchanged."""

    local_tensor = torch.randn(3, 3)

    dtensor_mock: DTensor = Mock(spec=DTensor)
    dtensor_mock.device = torch.device("cpu")
    dtensor_mock._local_tensor = local_tensor

    assert to_cpu(dtensor_mock) is local_tensor


def test_to_cpu_dtensor_unknown_device():
    """Unknown device types must raise *ValueError*."""

    dtensor_mock: DTensor = Mock(spec=DTensor)
    dtensor_mock.device = _FakeDevice("mps")  # Unsupported in *to_cpu*

    with pytest.raises(ValueError):
        _ = to_cpu(dtensor_mock)


def test_to_cpu_passthrough_other_types():
    """Non-tensor inputs are returned unchanged."""

    sentinel = object()
    assert to_cpu(sentinel) is sentinel

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for *inf* norm path.")
@pytest.mark.parametrize("pin_memory", [True, False])
def test_get_cpu_state_dict_scalar(pin_memory):
    """Handles 0-dim tensors correctly."""

    bias = torch.tensor(7.0)
    state_gen = [("bias", bias)]

    out = get_cpu_state_dict(state_gen, pin_memory=pin_memory)

    assert list(out.keys()) == ["bias"]
    assert torch.allclose(out["bias"], bias)
    assert out["bias"].device.type == "cpu"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for *inf* norm path.")
@pytest.mark.parametrize("pin_memory", [True, False])
def test_get_cpu_state_dict_multi_dim(pin_memory):
    """Multi-dimensional tensors are copied element-wise to CPU."""

    weight = torch.randn(4, 6)
    state_gen = [("weight", weight)]

    out = get_cpu_state_dict(state_gen, pin_memory=pin_memory)

    assert out["weight"].shape == weight.shape
    assert torch.allclose(out["weight"], weight)
    assert out["weight"].device.type == "cpu"


def test_get_cpu_state_dict_with_dtensor(monkeypatch):
    """DTensor values should be unwrapped to their local shard before copying."""

    local_weight = torch.randn(2, 2)

    dtensor_mock: DTensor = Mock(spec=DTensor)
    dtensor_mock.to_local.return_value = local_weight

    state_gen = [("w", dtensor_mock)]

    out = get_cpu_state_dict(state_gen)

    assert torch.allclose(out["w"], local_weight)
    assert out["w"].device.type == "cpu"
