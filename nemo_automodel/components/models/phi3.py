# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model-owned parallelization for phi3."""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.tp_styles import (
    VocabParallelEmbedding,
)


def _parallelize_phi3(
    model: nn.Module | None,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        # Fused Attention can not be sharded
        "model.layers.*.self_attn.qkv_proj": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "model.layers.*.self_attn.o_proj": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        # Shard MLP layers
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
        "model.layers.*.mlp.down_proj": RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=Replicate(),
        ),
        "lm_head": ColwiseParallel(
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    }

    return cast(
        dict[str, ParallelStyle],
        base_model_tp_plan,
    )


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_phi3)

__all__ = ["PARALLELIZER"]
