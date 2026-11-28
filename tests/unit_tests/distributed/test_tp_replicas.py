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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.distributed.tp_replicas import (
    _TP_REPLICA_GRAD_REDUCTION_ATTR,
    _get_tp_mesh,
    _gradient_chunks,
    _gradient_reduce_dtype,
    _iter_unique_buffers,
    _iter_unique_parameters,
    _local_tensor,
    broadcast_tp_replicas,
    exclude_from_tp_replica_sync,
    mark_tp_replica_gradient_reduction,
    synchronize_tp_replica_gradients,
)


def test_mark_tp_replica_gradient_reduction():
    module = nn.Linear(10, 10)
    mark_tp_replica_gradient_reduction(module, "mean")
    assert getattr(module, _TP_REPLICA_GRAD_REDUCTION_ATTR) == "mean"

    mark_tp_replica_gradient_reduction(module, "sum")
    assert getattr(module, _TP_REPLICA_GRAD_REDUCTION_ATTR) == "sum"


def test_exclude_from_tp_replica_sync():
    parent = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    exclude_from_tp_replica_sync(parent)

    for child in parent.modules():
        assert getattr(child, _TP_REPLICA_GRAD_REDUCTION_ATTR) == "skip"


def test_gradient_reduce_dtype():
    assert _gradient_reduce_dtype(torch.bfloat16) == torch.float32
    assert _gradient_reduce_dtype(torch.float16) == torch.float32
    assert _gradient_reduce_dtype(torch.float32) == torch.float32
    assert _gradient_reduce_dtype(torch.float64) == torch.float64


def test_local_tensor_standard_tensor():
    t = torch.randn(4, 4)
    assert _local_tensor(t) is t


def test_get_tp_mesh_none():
    assert _get_tp_mesh(None, "tp") is None


def test_broadcast_tp_replicas_noop_without_mesh():
    model = nn.Linear(10, 10)
    count = broadcast_tp_replicas([model], device_mesh=None, tp_axis_name="tp")
    assert count == 0


def test_synchronize_tp_replica_gradients_noop_without_mesh():
    model = nn.Linear(10, 10)
    count = synchronize_tp_replica_gradients([model], device_mesh=None, tp_axis_name="tp")
    assert count == 0


def test_gradient_chunks():
    g1 = torch.zeros(100, dtype=torch.float32)
    g2 = torch.zeros(200, dtype=torch.float32)
    chunks = list(_gradient_chunks([g1, g2]))
    assert len(chunks) == 1
    assert chunks[0] == [g1, g2]


def test_gradient_chunks_splits_large_buffers(monkeypatch):
    monkeypatch.setattr("nemo_automodel.components.distributed.tp_replicas._MAX_FLAT_BUFFER_BYTES", 400)
    # 100 float32 elements = 400 bytes
    g1 = torch.zeros(50, dtype=torch.float32)  # 200 bytes
    g2 = torch.zeros(60, dtype=torch.float32)  # 240 bytes -> exceeds 400 with g1
    chunks = list(_gradient_chunks([g1, g2]))
    assert len(chunks) == 2
    assert chunks[0] == [g1]
    assert chunks[1] == [g2]


def test_iter_unique_parameters_invalid_reduction():
    model = nn.Linear(5, 5)
    setattr(model, _TP_REPLICA_GRAD_REDUCTION_ATTR, "invalid_reduction")
    with pytest.raises(ValueError, match="Unsupported TP replica gradient reduction"):
        list(_iter_unique_parameters([model]))


def test_iter_unique_parameters_conflicting_reductions():
    shared_param = nn.Parameter(torch.randn(5, 5))
    m1 = nn.Module()
    m1.weight = shared_param
    mark_tp_replica_gradient_reduction(m1, "mean")

    m2 = nn.Module()
    m2.weight = shared_param
    mark_tp_replica_gradient_reduction(m2, "sum")

    with pytest.raises(RuntimeError, match="conflicting TP replica gradient reductions"):
        list(_iter_unique_parameters([m1, m2]))


def test_iter_unique_buffers():
    m = nn.BatchNorm1d(10)
    buffers = list(_iter_unique_buffers([m]))
    assert len(buffers) >= 2  # running_mean, running_var
