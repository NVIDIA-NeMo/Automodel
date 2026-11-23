# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Typed-decision post-training support for Nemotron 3.5 Decision checkpoints."""

from nemo_automodel.components.models.deciffusion.model import NemotronForDecision
from nemo_automodel.components.models.deciffusion.records import encode_question
from nemo_automodel.components.models.deciffusion.scorer import require_decision_config
from nemo_automodel.components.models.deciffusion.serving_prompt import (
    DecisionContract,
    PreparedQuestion,
    prepare_question,
)

__all__ = [
    "DecisionContract",
    "NemotronForDecision",
    "PreparedQuestion",
    "encode_question",
    "prepare_question",
    "require_decision_config",
]
