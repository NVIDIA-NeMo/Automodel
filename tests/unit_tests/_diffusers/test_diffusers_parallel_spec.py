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

"""The diffusion pipeline binds model-owned ParallelSpec declarations onto diffusers transformers."""

import pytest
import torch.nn as nn

from nemo_automodel._diffusers.parallelization import attach_parallel_spec, diffusers_family
from nemo_automodel.components.distributed.parallelizer import get_parallelization_strategy, query_parallel_spec
from nemo_automodel.components.models.hunyuan_video15.parallelization import HunyuanParallelizationStrategy
from nemo_automodel.components.models.ltx2_video.parallelization import LTX2ParallelizationStrategy
from nemo_automodel.components.models.qwen_image.parallelization import QwenImageEditParallelizationStrategy
from nemo_automodel.components.models.wan.parallelization import WanParallelizationStrategy


@pytest.mark.parametrize(
    ("class_name", "family"),
    [
        ("WanTransformer3DModel", "wan"),
        ("HunyuanVideo15Transformer3DModel", "hunyuan_video15"),
        ("LTX2VideoTransformer3DModel", "ltx2_video"),
        ("QwenImageTransformer2DModel", "qwen_image"),
        ("Transformer2DModel", ""),
    ],
)
def test_diffusers_family_is_the_snake_case_stem_before_transformer(class_name, family):
    assert diffusers_family(class_name) == family


@pytest.mark.parametrize(
    ("class_name", "strategy_cls"),
    [
        ("WanTransformer3DModel", WanParallelizationStrategy),
        ("HunyuanVideo15Transformer3DModel", HunyuanParallelizationStrategy),
        ("LTX2VideoTransformer3DModel", LTX2ParallelizationStrategy),
        ("QwenImageTransformer2DModel", QwenImageEditParallelizationStrategy),
    ],
)
def test_attach_binds_the_declared_strategy_onto_the_instance_class(class_name, strategy_cls):
    upstream = type(class_name, (nn.Module,), {})
    module = attach_parallel_spec(upstream())
    assert isinstance(module, upstream)
    assert type(module).__name__ == class_name
    assert type(get_parallelization_strategy(module)) is strategy_cls


def test_subclass_inherits_its_base_declaration():
    base = type("WanTransformer3DModel", (nn.Module,), {})
    module = attach_parallel_spec(type("WanVariant", (base,), {})())
    assert type(get_parallelization_strategy(module)) is WanParallelizationStrategy


def test_modules_without_a_declaration_are_returned_unchanged():
    module = nn.Linear(2, 2)
    assert attach_parallel_spec(module) is module
    assert type(module) is nn.Linear
    assert query_parallel_spec(module).strategy is None
