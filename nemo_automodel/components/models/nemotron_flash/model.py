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

"""Nemotron Flash model entry points for NeMo Automodel."""

from transformers import AutoModelForCausalLM

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

from .configuration import NemotronFlashConfig
from .modeling_nemotron_flash import (
    NemotronFlashForCausalLM as _HFNemotronFlashForCausalLM,
)
from .modeling_nemotron_flash import (
    NemotronFlashModel,
    NemotronFlashPreTrainedModel,
)

__all__ = [
    "NemotronFlashConfig",
    "NemotronFlashPreTrainedModel",
    "NemotronFlashModel",
    "NemotronFlashForCausalLM",
    "ModelClass",
]


class NemotronFlashForCausalLM(HFCheckpointingMixin, _HFNemotronFlashForCausalLM):
    """NeMo AutoModel wrapper around the packaged Nemotron Flash implementation."""


try:
    AutoModelForCausalLM.register(NemotronFlashConfig, NemotronFlashForCausalLM)
except ValueError:
    pass


ModelClass = NemotronFlashForCausalLM
