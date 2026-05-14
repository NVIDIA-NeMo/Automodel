# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""EAGLE-3 training components.

Only the EAGLE-3-native model and trainer classes live here. Generic
helpers were moved to the project's canonical locations:

- ``masked_soft_cross_entropy`` -> ``nemo_automodel.components.loss.soft_ce``
- ``build_eagle3_dataloader`` / ``build_eagle3_token_mapping``
  -> ``nemo_automodel.components.datasets.llm.eagle3``

Import them directly from those modules.
"""

from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel

__all__ = [
    "Eagle3TrainerModule",
    "HFEagle3TargetModel",
    "LlamaEagle3DraftModel",
]
