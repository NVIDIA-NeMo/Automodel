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

"""Model-agnostic post-training components for System One / Jev-style decision models.

Decision models read a ``state`` plus a schema of typed questions (``noul`` / ``choice`` / ``score``) and return a
probability for every allowed option in a single forward pass.  This package holds the parts shared by every such
model: SystemOne records and targets, SFT / RL objectives, and evaluation metrics.
"""

from nemo_automodel.components.decision.losses import DecisionLoss, DecisionLossConfig, DecisionTargets
from nemo_automodel.components.decision.metrics import DecisionEvaluator

__all__ = [
    "DecisionEvaluator",
    "DecisionLoss",
    "DecisionLossConfig",
    "DecisionTargets",
]
