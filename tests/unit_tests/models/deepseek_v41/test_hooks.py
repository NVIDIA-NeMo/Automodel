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

"""Model-owned indexer freezing and generic MoE parallelization selection."""

from types import SimpleNamespace

from nemo_automodel._transformers.capabilities import _is_deepseek_v4
from nemo_automodel.components.distributed.parallelizer import (
    DefaultParallelizationStrategy,
    get_parallelization_strategy,
)
from nemo_automodel.components.models.deepseek_v4 import fsdp as dsv4_fsdp
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.moe.parallelizer import _is_deepseek_v4_model
from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config


def test_indexers_are_frozen_by_the_model_constructor() -> None:
    model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    expected = {name for name, _ in model.named_parameters() if ".attn.indexer." in name}
    assert expected and frozen == expected


def test_v41_uses_generic_moe_parallelization() -> None:
    model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    assert not _is_deepseek_v4_model(model)
    assert type(get_parallelization_strategy(model)) is DefaultParallelizationStrategy
    assert not dsv4_fsdp._is_deepseek_v4_module(model)
    assert not _is_deepseek_v4(model)
    assert _is_deepseek_v4(SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4")))
    for layer in model.model.layers.values():
        assert layer.mlp is layer.ffn
        assert all(".mlp." not in name for name in model.state_dict())
