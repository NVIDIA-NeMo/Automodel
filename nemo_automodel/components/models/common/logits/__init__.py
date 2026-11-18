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

"""Language-model head projections."""

from nemo_automodel.components.models.common.logits.logits_processor import LogitsProcessor
from nemo_automodel.components.models.common.logits.projection import (
    compute_lm_head_logits,
    enable_fp32_logits,
    fp32_linear_forward,
    head_compute_dtype,
)

__all__ = [
    "LogitsProcessor",
    "compute_lm_head_logits",
    "enable_fp32_logits",
    "fp32_linear_forward",
    "head_compute_dtype",
]
