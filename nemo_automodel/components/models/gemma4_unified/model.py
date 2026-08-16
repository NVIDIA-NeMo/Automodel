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

"""Model-specific initialization for Hugging Face-native Gemma4 Unified models."""

from typing import Any

from nemo_automodel.components.models.gemma4_unified.state_dict_adapter import (
    GEMMA4_UNIFIED_MODEL_TYPE,
    Gemma4UnifiedStateDictAdapter,
)


def maybe_attach_state_dict_adapter(model: Any, *, is_custom_model: bool) -> None:
    """Attach the Gemma4 Unified adapter without replacing an existing adapter."""
    if (
        not is_custom_model
        and getattr(getattr(model, "config", None), "model_type", None) == GEMMA4_UNIFIED_MODEL_TYPE
        and getattr(model, "state_dict_adapter", None) is None
    ):
        model.state_dict_adapter = Gemma4UnifiedStateDictAdapter()
