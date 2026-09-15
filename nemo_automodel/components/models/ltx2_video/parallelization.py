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

"""Parallelization contract for the diffusers ``LTX2VideoTransformer3DModel``."""

from __future__ import annotations

from nemo_automodel.components.distributed.activation_checkpointing import ActivationCheckpointingSpec
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec


class LTX2VideoTransformer3DModel:
    """Contract for the diffusers ``LTX2VideoTransformer3DModel``; bound by the diffusion pipeline before sharding."""

    # diffusers declares no ``_no_split_modules`` for this class, so the block container is named here.
    parallel_spec: ParallelSpec = ParallelSpec(layer_groups={"backbone": ("transformer_blocks",)})
    # The blocks name their attention/FFN children ``attn1``/``attn2``/``ff``, which the submodule wrappers do
    # not cover; wrapping each whole block keeps the long combined video+audio token sequence in memory budget.
    activation_checkpointing_spec: ActivationCheckpointingSpec = ActivationCheckpointingSpec(granularity="layer")
