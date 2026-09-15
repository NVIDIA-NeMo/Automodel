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

"""Parallelization contract for the transformers ``Qwen2VLForConditionalGeneration``."""

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# Layer containers list every known location across transformers releases; the first candidate
# that resolves wins. Canonical paths come first so the deprecated top-level ``visual`` alias of
# standardized 4.x releases is never picked over ``model.visual``.
QWEN2_VL_LAYERS = {
    "language": ("model.language_model.layers", "model.layers"),
    "vision": ("model.visual.blocks", "visual.blocks"),
}

# Shared with Qwen2.5-VL, whose module tree is identical.
QWEN2_VL_PARALLEL_SPEC = ParallelSpec(
    layer_groups=QWEN2_VL_LAYERS,
)


class Qwen2VLForConditionalGeneration:
    """Contract for the transformers ``Qwen2VLForConditionalGeneration``; bound by the loader onto its wrapper."""

    parallel_spec: ParallelSpec = QWEN2_VL_PARALLEL_SPEC
