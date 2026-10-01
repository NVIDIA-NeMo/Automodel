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

"""Deprecated compatibility imports for the FSDP2 extension helpers."""

import warnings

from nemo_automodel.components.distributed.fsdp2_extensions.compat import (
    patch_fsdp_accumulated_grad_guard,
    patch_fsdp_uniform_reduce_dtype,
    patch_fsdp_unused_param_reduction,
)

warnings.warn(
    "fsdp_patches moved to distributed.fsdp2_extensions.compat; update imports before the next major release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["patch_fsdp_accumulated_grad_guard", "patch_fsdp_uniform_reduce_dtype", "patch_fsdp_unused_param_reduction"]
