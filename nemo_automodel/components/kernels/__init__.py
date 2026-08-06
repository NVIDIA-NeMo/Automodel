# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Hub-backed kernel loading via the Hugging Face ``kernels`` library."""

from nemo_automodel.components.kernels.config import HubKernelConfig
from nemo_automodel.components.kernels.hub import (
    HUB_FLASH_ATTN2,
    apply_hub_kernels_to_model,
    extract_hub_kernels_config,
    get_flash_attn_func,
    get_flash_attn_varlen_func,
    get_hub_kernel,
    has_flash_attn_available,
    has_hub_kernel,
    is_hub_attn_implementation,
    resolve_attn_implementation,
)

__all__ = [
    "HUB_FLASH_ATTN2",
    "HubKernelConfig",
    "apply_hub_kernels_to_model",
    "extract_hub_kernels_config",
    "get_flash_attn_func",
    "get_flash_attn_varlen_func",
    "get_hub_kernel",
    "has_flash_attn_available",
    "has_hub_kernel",
    "is_hub_attn_implementation",
    "resolve_attn_implementation",
]
