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

"""Model-owned distributed parallelization for DeepSeek-V4."""

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy
from nemo_automodel.components.models.deepseek_v4.fsdp import fully_shard_deepseek_v4


class DeepseekV4ParallelizationStrategy(DefaultParallelizationStrategy):
    """Keep DeepSeek-V4 reference-sensitive parameters in fp32 FSDP units."""

    def _fully_shard_module(self, module, **kwargs):
        return fully_shard_deepseek_v4(module, **kwargs)


_STRATEGY = DeepseekV4ParallelizationStrategy()
PARALLELIZER = ModelParallelizer(_STRATEGY, _STRATEGY)

__all__ = ["PARALLELIZER"]
