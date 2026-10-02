# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen3.5 owns its model-class parallelization sidecar."""

from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from nemo_automodel.components.models.qwen3_5.parallelization import PARALLELIZER, Qwen3_5ModelParallelizer


def test_qwen3_5_models_resolve_the_model_owned_sidecar():
    assert isinstance(PARALLELIZER, Qwen3_5ModelParallelizer)
    for cls in (Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration):
        model = cls.__new__(cls)
        assert get_model_parallelizer(model) is PARALLELIZER
