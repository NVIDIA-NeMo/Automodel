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

"""Qwen3.5 owns registration of its distributed precision policy."""

from nemo_automodel.components.distributed.parallelizer import PARALLELIZATION_STRATEGIES
from nemo_automodel.components.models.qwen3_5.parallelization import (
    Qwen3_5ParallelizationStrategy,
    register_qwen3_5_parallel_strategy,
)


def test_qwen3_5_registers_both_native_model_classes_idempotently():
    register_qwen3_5_parallel_strategy()
    first = {
        name: PARALLELIZATION_STRATEGIES[name] for name in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration")
    }

    register_qwen3_5_parallel_strategy()

    assert all(isinstance(strategy, Qwen3_5ParallelizationStrategy) for strategy in first.values())
    assert all(PARALLELIZATION_STRATEGIES[name] is strategy for name, strategy in first.items())
