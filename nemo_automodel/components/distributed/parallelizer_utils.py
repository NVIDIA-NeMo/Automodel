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

from nemo_automodel.components.distributed.fsdp2_extensions.utils import (
    UniformSubtreeItem as UniformSubtreeItem,
)
from nemo_automodel.components.distributed.fsdp2_extensions.utils import (
    configure_fsdp_unused_param_reduction as configure_fsdp_unused_param_reduction,
)
from nemo_automodel.components.distributed.fsdp2_extensions.utils import (
    fully_shard_by_dtype,
    reject_unsupported_mtp_cp,
    reject_unsupported_mtp_cp_pp,
)
from nemo_automodel.components.distributed.fsdp2_extensions.utils import (
    get_internal_fsdp_mp_policy as get_internal_fsdp_mp_policy,
)
from nemo_automodel.components.distributed.fsdp2_extensions.utils import (
    iter_maximal_uniform_dtype_subtrees as iter_maximal_uniform_dtype_subtrees,
)

warnings.warn(
    "parallelizer_utils moved to distributed.fsdp2_extensions.utils; update imports before the next major release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["fully_shard_by_dtype", "reject_unsupported_mtp_cp", "reject_unsupported_mtp_cp_pp"]
