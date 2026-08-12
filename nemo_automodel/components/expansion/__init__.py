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

"""Dual-stream model expansion.

Grows a pretrained model by giving selected linear layers a second, trainable weight while
the pretrained weights stay frozen. The expanded model reproduces its parent exactly until
the new weights learn, which makes it a continuation of pretraining rather than a restart.

See :mod:`nemo_automodel.components.expansion.dual_stream` for how the second hidden-state
stream is carried, and :mod:`nemo_automodel.components.expansion.expanded_linear` for the
lateral connection between the streams.
"""

from nemo_automodel.components.expansion.apply import (
    ExpansionConfig,
    apply_expansion,
    expanded_linears,
    expansion_parameters,
    freeze_non_expansion_parameters,
    initialize_expansion,
    is_expansion_parameter,
)
from nemo_automodel.components.expansion.expanded_linear import (
    ExpandedLinear,
    LateralBus,
    LateralBusMode,
    patch_linear_for_expansion,
)

__all__ = [
    "ExpandedLinear",
    "ExpansionConfig",
    "LateralBus",
    "LateralBusMode",
    "apply_expansion",
    "expanded_linears",
    "expansion_parameters",
    "freeze_non_expansion_parameters",
    "initialize_expansion",
    "is_expansion_parameter",
    "patch_linear_for_expansion",
]
