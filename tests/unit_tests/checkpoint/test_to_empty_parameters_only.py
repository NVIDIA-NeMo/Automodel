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

"""Meta-parameter materialization must keep the model-owned sharding contract."""

import pytest
import torch
from torch import nn

from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
from nemo_automodel.shared.model_owned_sharding import (
    model_owned_grad_divisor,
    stamp_model_owned_grad_divisor,
)


def _meta_module() -> nn.Module:
    with torch.device("meta"):
        module = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))
    stamp_model_owned_grad_divisor(module[0].weight, 8.0)
    return module


@pytest.mark.parametrize("swap_params", [False, True])
def test_to_empty_parameters_only_keeps_model_owned_grad_divisor(swap_params: bool):
    """The owner-group divisor survives both the rebuild and the swap path."""
    previous = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(swap_params)
    try:
        module = _meta_module()
        to_empty_parameters_only(module, device=torch.device("cpu"))
    finally:
        torch.__future__.set_swap_module_params_on_conversion(previous)

    assert module[0].weight.device.type == "cpu"
    assert module[0].weight.shape == (3, 4)
    assert model_owned_grad_divisor(module[0].weight) == 8.0
    assert model_owned_grad_divisor(module[1].weight) is None


def test_to_empty_parameters_only_does_not_invent_a_contract():
    """Parameters without a contract stay unmarked after materialization."""
    with torch.device("meta"):
        module = nn.Linear(4, 3, bias=True)
    to_empty_parameters_only(module, device=torch.device("cpu"))
    assert model_owned_grad_divisor(module.weight) is None
    assert model_owned_grad_divisor(module.bias) is None
