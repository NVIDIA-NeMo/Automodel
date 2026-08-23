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
from torch import nn

from nemo_automodel.shared.task_heads import (
    exclude_task_heads_from_tp_plan,
    is_task_head_parameter,
    register_task_head_module,
    task_head_module,
)


class _ProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)


class _ModelWithTaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ProjectionBlock()


def _model_with_registered_task_head() -> _ModelWithTaskHead:
    model = _ModelWithTaskHead()
    pre_hook_module_ids = {id(module) for module in model.modules()}
    pre_hook_parameter_ids = {id(parameter) for parameter in model.parameters()}
    model.task_head = _ProjectionBlock()
    register_task_head_module(
        model,
        model.task_head,
        pre_hook_module_ids=pre_hook_module_ids,
        pre_hook_parameter_ids=pre_hook_parameter_ids,
    )
    return model


def test_exclude_task_heads_from_tp_plan_keeps_only_backbone_rules() -> None:
    model = _model_with_registered_task_head()
    backbone_style = object()

    filtered = exclude_task_heads_from_tp_plan(
        model,
        {
            "backbone.projection": backbone_style,
            "task_head.projection": object(),
            "task_head.*": object(),
        },
    )

    assert filtered == {"backbone.projection": backbone_style}


def test_exclude_task_heads_from_tp_plan_rejects_mixed_wildcard() -> None:
    model = _model_with_registered_task_head()

    with pytest.raises(ValueError, match="matches both the managed task module and backbone modules"):
        exclude_task_heads_from_tp_plan(model, {"*.projection": object()})


def test_task_head_module_resolves_below_transparent_wrapper() -> None:
    model = _model_with_registered_task_head()
    wrapper = nn.Module()
    wrapper.module = model

    assert task_head_module(wrapper) is model.task_head

    compiled_wrapper = nn.Module()
    compiled_wrapper._orig_mod = model

    assert task_head_module(compiled_wrapper) is model.task_head
    assert is_task_head_parameter(compiled_wrapper, "_orig_mod.task_head.projection.weight")
