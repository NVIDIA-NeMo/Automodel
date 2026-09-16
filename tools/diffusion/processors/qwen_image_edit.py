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

"""Registry entry for Qwen-Image-Edit cache preprocessing.

The encoding logic is owned by the model package
(``nemo_automodel.components.models.qwen_image_edit.preprocessing``); this
module only registers it under the ``qwen_image_edit`` processor name so the
``image-edit`` CLI selects it the same way the image and video CLIs select
their processors.
"""

from nemo_automodel.components.models.qwen_image_edit.preprocessing import QwenImageEditCacheEncoder

from .registry import ProcessorRegistry


@ProcessorRegistry.register("qwen_image_edit")
class QwenImageEditProcessor(QwenImageEditCacheEncoder):
    """Qwen-Image-Edit manifest encoder registered for ``--processor qwen_image_edit``."""
