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

"""NeMo Automodel support for the ``diffusion_gemma`` block-diffusion model.

The config classes are the released ``transformers`` ``DiffusionGemmaConfig`` /
``DiffusionGemmaTextConfig`` (transformers defines and ``AutoConfig``-registers
``model_type="diffusion_gemma"`` natively), re-exported here for convenience.
AM owns only the compute (model/layers/adapter).
"""

from nemo_automodel.shared.import_utils import UnavailableMeta

try:
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaConfig as DiffusionGemmaConfig,
    )
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaTextConfig as DiffusionGemmaTextConfig,
    )
except (ModuleNotFoundError, ImportError):
    _MSG = "transformers.models.diffusion_gemma is not available."
    DiffusionGemmaConfig = UnavailableMeta("DiffusionGemmaConfig", (), {"_msg": _MSG})
    DiffusionGemmaTextConfig = UnavailableMeta("DiffusionGemmaTextConfig", (), {"_msg": _MSG})

__all__ = ["DiffusionGemmaConfig", "DiffusionGemmaTextConfig"]
