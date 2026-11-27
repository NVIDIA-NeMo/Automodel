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

"""DFly2 speculative-decoding draft training (arXiv:2611.34776).

A block-parallel drafter for sparse-attention targets: each draft layer is paired with
one target sparse-attention layer and reads that layer's keys (KV sharing) at the
positions it attends from the row before the anchor (index sharing), so the served
drafter keeps no KV cache and runs no indexer of its own.
"""

from nemo_automodel.components.speculative.dfly2.common import DFly2TargetFeatures, build_block_key_indices
from nemo_automodel.components.speculative.dfly2.core import DFly2TrainerModule
from nemo_automodel.components.speculative.dfly2.target import (
    DFly2TargetFeatureProvider,
    DFly2TargetModel,
    ModelOwnedDFly2Provider,
)

__all__ = [
    "DFly2TargetFeatureProvider",
    "DFly2TargetFeatures",
    "DFly2TargetModel",
    "DFly2TrainerModule",
    "ModelOwnedDFly2Provider",
    "build_block_key_indices",
]
