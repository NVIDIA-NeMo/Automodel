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

"""DSpark speculative-decoding draft model and training objective.

A semi-autoregressive parallel drafter: a parallel backbone produces every
position of a block in one pass, a lightweight serial Markov head injects
intra-block token dependency, and a confidence head predicts per-position
acceptance probability for scheduled verification.
"""

from nemo_automodel.components.speculative.dspark.common import DSparkForwardOutput
from nemo_automodel.components.speculative.dspark.config import build_draft_config
from nemo_automodel.components.speculative.dspark.draft_qwen3 import Qwen3DSparkModel
from nemo_automodel.components.speculative.dspark.loss import compute_dspark_loss

__all__ = [
    "DSparkForwardOutput",
    "Qwen3DSparkModel",
    "build_draft_config",
    "compute_dspark_loss",
]
