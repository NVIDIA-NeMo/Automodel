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

"""UPipe (Untied Ulysses) context-parallel primitives.

Head-chunked Ulysses context parallelism: attention is walked one head-chunk at a time
so full-size QKV (and, for Inkling, the relative-position logit bank) are never
materialized. Ring degree is fixed at 1, so this is pure sequence-to-head all-to-all
with no ring rotation.
"""

from nemo_automodel.components.distributed.context_parallel.upipe.all_to_all import (
    all_to_all_single,
    cp2hp,
    hp2cp,
)
from nemo_automodel.components.distributed.context_parallel.upipe.geometry import (
    UPipeHeadGeometry,
    geometry_for_attention,
)

__all__ = [
    "UPipeHeadGeometry",
    "all_to_all_single",
    "cp2hp",
    "geometry_for_attention",
    "hp2cp",
]
