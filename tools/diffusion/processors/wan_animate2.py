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

"""Registry entry for Wan-Animate-2 triplet cache preprocessing.

The encoding logic is owned by the model package
(``nemo_automodel.components.models.wan_animate2.preprocessing``); this module
only registers it under the ``wan_animate2`` processor name so a preprocessing
CLI selects it the same way the image, image-edit, and video CLIs select theirs.
"""

from nemo_automodel.components.models.wan_animate2.preprocessing import WanAnimate2CacheEncoder

from .registry import ProcessorRegistry


@ProcessorRegistry.register("wan_animate2")
class WanAnimate2Processor(WanAnimate2CacheEncoder):
    """Wan-Animate-2 triplet manifest encoder registered for ``--processor wan_animate2``."""
