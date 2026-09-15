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

"""Parallelization contract for the transformers ``Llama4ForConditionalGeneration``."""

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec


class Llama4ForConditionalGeneration:
    """Contract for the transformers ``Llama4ForConditionalGeneration``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = ParallelSpec(
        layer_groups={"language": ("language_model.model.layers",), "vision": ("vision_model.model.layers",)},
    )
