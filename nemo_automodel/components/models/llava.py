# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model-owned layer layout for llava."""

from nemo_automodel.components.distributed import ModelParallelizer

PARALLELIZER = ModelParallelizer(
    layer_group_paths={
        "language": ("model.language_model.layers", "language_model.model.layers"),
        "vision": (
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ),
    }
)
