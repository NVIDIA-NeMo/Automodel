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

"""Parallelization contract for the transformers ``Qwen2_5_VLForConditionalGeneration``."""

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.models.qwen2_vl.parallelization import QWEN2_VL_PARALLEL_SPEC


class Qwen2_5_VLForConditionalGeneration:
    """Contract for the transformers ``Qwen2_5_VLForConditionalGeneration``; same module tree as Qwen2-VL."""

    parallel_spec: ParallelSpec = QWEN2_VL_PARALLEL_SPEC
