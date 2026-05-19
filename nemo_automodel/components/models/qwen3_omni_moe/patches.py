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

"""Compatibility patches for Qwen3 Omni MoE models."""

import logging

logger = logging.getLogger(__name__)


def apply_qwen3_omni_config_patch() -> None:
    """Fix Qwen3OmniMoeTalkerCodePredictorConfig accessing use_sliding_window."""
    try:
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )
    except ImportError:
        logger.debug("Qwen3 Omni MoE config class is unavailable; skipping config patch.")
        return

    if not hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window"):
        Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = False


def apply_global_patches() -> None:
    """Apply import-time compatibility patches for Qwen3 Omni MoE models."""
    apply_qwen3_omni_config_patch()


def apply_pre_config_patches() -> None:
    """Apply patches required before HuggingFace config construction."""
    apply_qwen3_omni_config_patch()
