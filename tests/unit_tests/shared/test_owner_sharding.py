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

"""Tests for the model-agnostic owner-sharded parameter contract."""

import pytest
import torch

import nemo_automodel.shared.owner_sharding as owner_sharding
from nemo_automodel.shared.owner_sharding import (
    OwnerShardedParameterSpec,
    get_owner_sharded_parameter_spec,
)


def test_owner_sharded_parameter_spec_round_trip() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    spec = OwnerShardedParameterSpec(
        process_group=None,
        gradient_divisor=4.0,
        optimizer_state_namespace="__test_owner_v1",
    )
    parameter._nemo_owner_sharded_spec = spec

    assert get_owner_sharded_parameter_spec(parameter) is spec
    assert get_owner_sharded_parameter_spec(torch.ones(1)) is None


@pytest.mark.parametrize("divisor", [0.0, -1.0, float("inf"), float("nan")])
def test_owner_sharded_parameter_spec_rejects_invalid_gradient_divisor(divisor: float) -> None:
    with pytest.raises(ValueError, match="gradient_divisor"):
        OwnerShardedParameterSpec(
            process_group=None,
            gradient_divisor=divisor,
            optimizer_state_namespace="__test_owner_v1",
        )


@pytest.mark.parametrize("divisor", [True, "2"])
def test_owner_sharded_parameter_spec_rejects_non_real_gradient_divisor(divisor: object) -> None:
    with pytest.raises(TypeError, match="gradient_divisor"):
        OwnerShardedParameterSpec(
            process_group=None,
            gradient_divisor=divisor,
            optimizer_state_namespace="__test_owner_v1",
        )


@pytest.mark.parametrize("namespace", ["", " has_space", "contains.dot", "contains/slash"])
def test_owner_sharded_parameter_spec_rejects_invalid_namespace(namespace: str) -> None:
    with pytest.raises(ValueError, match="optimizer_state_namespace"):
        OwnerShardedParameterSpec(
            process_group=None,
            gradient_divisor=1.0,
            optimizer_state_namespace=namespace,
        )


def test_owner_sharded_parameter_spec_rejects_non_string_namespace() -> None:
    with pytest.raises(TypeError, match="optimizer_state_namespace"):
        OwnerShardedParameterSpec(
            process_group=None,
            gradient_divisor=1.0,
            optimizer_state_namespace=7,
        )


def test_get_owner_sharded_parameter_spec_rejects_untyped_marker() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter._nemo_owner_sharded_spec = object()

    with pytest.raises(TypeError, match="must be an OwnerShardedParameterSpec"):
        get_owner_sharded_parameter_spec(parameter)


def test_get_owner_sharded_parameter_spec_rejects_second_sharding_owner(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter._nemo_owner_sharded_spec = OwnerShardedParameterSpec(
        process_group=None,
        gradient_divisor=1.0,
        optimizer_state_namespace="__test_owner_v1",
    )
    monkeypatch.setattr(owner_sharding, "DTensor", torch.Tensor)

    with pytest.raises(RuntimeError, match="cannot also use a model-owned sharding"):
        get_owner_sharded_parameter_spec(parameter)
