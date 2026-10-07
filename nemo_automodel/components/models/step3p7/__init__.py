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


from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "Step3p7Config",
    "Step3p7ForConditionalGeneration",
    "Step3p7TextConfig",
    "StepRoboticsVisionEncoderConfig",
]


if TYPE_CHECKING:
    from nemo_automodel.components.models.step3p7.configuration_step3p7 import (
        Step3p7Config,
        Step3p7TextConfig,
        StepRoboticsVisionEncoderConfig,
    )
    from nemo_automodel.components.models.step3p7.model import Step3p7ForConditionalGeneration


def __getattr__(name: str) -> Any:
    """Load public model exports only when requested, keeping sidecars lightweight."""
    if name in ("Step3p7Config", "Step3p7TextConfig", "StepRoboticsVisionEncoderConfig"):
        return getattr(import_module("nemo_automodel.components.models.step3p7.configuration_step3p7"), name)
    if name in ("Step3p7ForConditionalGeneration",):
        return getattr(import_module("nemo_automodel.components.models.step3p7.model"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
