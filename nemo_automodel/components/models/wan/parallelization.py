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

"""Parallelization contract for the diffusers ``WanTransformer3DModel``.

Its blocks (``blocks``, declared in diffusers' ``_no_split_modules``) form the ``backbone`` layer group the
parallelizer derives, so only the TP plan and the whole-block activation checkpointing are declared here.
"""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel

from nemo_automodel.components.distributed.activation_checkpointing import ActivationCheckpointingSpec
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# Condition embedders, the FFN of every block and the final projection; attention stays replicated.
WAN_TP_PLAN: dict[str, ParallelStyle] = {
    "condition_embedder.text_embedder.linear_1": ColwiseParallel(),
    "condition_embedder.text_embedder.linear_2": RowwiseParallel(),
    "condition_embedder.time_embedder.linear_1": ColwiseParallel(),
    "condition_embedder.time_embedder.linear_2": RowwiseParallel(),
    "condition_embedder.time_proj": ColwiseParallel(),
    "blocks.*.ffn.net.0.proj": ColwiseParallel(),
    "blocks.*.ffn.net.2": RowwiseParallel(),
    "proj_out": RowwiseParallel(),
}


class WanTransformer3DModel:
    """Contract for the diffusers ``WanTransformer3DModel``; bound by the diffusion pipeline before sharding."""

    parallel_spec: ParallelSpec = ParallelSpec(tp_plan=WAN_TP_PLAN)
    # Every block is recomputed on backward as one unit: with ~30k video tokens (Wan2.2-A14B) the fp32
    # layer-norm casts inside a block OOM even on 8x80GB when only its submodules are checkpointed.
    activation_checkpointing_spec: ActivationCheckpointingSpec = ActivationCheckpointingSpec(granularity="layer")
