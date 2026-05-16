# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Gemma4 drafter (assistant) NeMo Automodel support.

Adds a thin NeMo wrapper around the HuggingFace ``Gemma4AssistantForCausalLM``
plus a ``Gemma4WithDrafter`` composite for joint base + drafter fine-tuning.
"""

from nemo_automodel.components.models.gemma4_drafter.composite import (
    Gemma4JointOutput,
    Gemma4WithDrafter,
)
from nemo_automodel.components.models.gemma4_drafter.model import Gemma4DrafterForCausalLM

ModelClass = Gemma4DrafterForCausalLM

__all__ = [
    "Gemma4DrafterForCausalLM",
    "Gemma4JointOutput",
    "Gemma4WithDrafter",
]
