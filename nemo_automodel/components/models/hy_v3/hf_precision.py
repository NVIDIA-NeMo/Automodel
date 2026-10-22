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

"""Hugging Face hy_v3 routing precision compatibility."""

from __future__ import annotations

from torch import nn

from nemo_automodel.components.models.common.precision import preserve_fp32_module_compute


def apply_hf_precision(model: nn.Module, *, device_type: str) -> nn.Module:
    """Keep the upstream HF router's FP32 projection and scoring outside autocast.

    Args:
        model: Loaded model, including HF modules retained by force_hf or fallback.
        device_type: Device selected by the single-rank compute policy.

    Returns:
        Model with its HF router computation protected and original state intact.
    """
    from transformers.models.hy_v3.modeling_hy_v3 import HYV3TopKRouter

    return preserve_fp32_module_compute(model, module_type=HYV3TopKRouter, device_type=device_type)
