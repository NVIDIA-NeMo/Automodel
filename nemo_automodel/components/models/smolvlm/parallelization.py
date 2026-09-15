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

"""Parallelization contract for the transformers ``SmolVLMForConditionalGeneration``.

The layer groups cannot be derived: transformers lists ``SmolVLMVisionAttention`` (a block's attention child) in
``_no_split_modules`` rather than the ``SmolVLMEncoderLayer`` blocks the vision tower stacks, so the vision group
would be missed. Both containers are therefore declared.
"""

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec


class SmolVLMForConditionalGeneration:
    """Contract for the transformers ``SmolVLMForConditionalGeneration``; bound by the loader onto its wrapper."""

    parallel_spec: ParallelSpec = ParallelSpec(
        layer_groups={"language": ("model.text_model.layers",), "vision": ("model.vision_model.encoder.layers",)},
    )
