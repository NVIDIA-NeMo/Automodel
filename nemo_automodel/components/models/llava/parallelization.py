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

"""Parallelization contract for the transformers ``LlavaForConditionalGeneration``.

The LLaVA-Next, LLaVA-Next-Video and LLaVA-OneVision heads share this module tree and import the
spec from here.
"""

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# Same tree history as Gemma3 (CLIP instead of SigLIP tower): canonical paths first, then the
# pre-standardization and deprecated-alias locations; the first candidate that resolves wins.
LLAVA_LAYERS = {
    "language": ("model.language_model.layers", "language_model.model.layers"),
    "vision": (
        "model.vision_tower.vision_model.encoder.layers",
        "model.vision_tower.encoder.layers",
        "vision_tower.vision_model.encoder.layers",
    ),
}

LLAVA_PARALLEL_SPEC = ParallelSpec(
    layer_groups=LLAVA_LAYERS,
    text_config_path="language_model.config",
    hf_tp_plan_prefix=("model.language_model",),
)


class LlavaForConditionalGeneration:
    """Contract for the transformers ``LlavaForConditionalGeneration``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = LLAVA_PARALLEL_SPEC
