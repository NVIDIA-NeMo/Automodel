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

"""Model-owned parallelization for nemotron_labs_diffusion."""

from __future__ import annotations

from typing import cast

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.tp_styles import (
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)


def _parallelize_nemotron_labs_diffusion(
    model,  # NemotronLabsDiffusionModel — loaded via trust_remote_code, not importable.
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """TP plan for ``NemotronLabsDiffusionModel`` (Nemotron-Labs-Diffusion).

    Same shape as :func:`_parallelize_ministral3` but the model uses
    ``encoder.*`` (not ``model.*``) and the output head is ``diffusion_head``
    (not ``lm_head``).
    """
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "encoder.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "encoder.layers.*.self_attn.q_proj": ColwiseParallel(),
        "encoder.layers.*.self_attn.k_proj": ColwiseParallel(),
        "encoder.layers.*.self_attn.v_proj": ColwiseParallel(),
        "encoder.layers.*.self_attn.o_proj": RowwiseParallel(),
        "encoder.layers.*.mlp.up_proj": ColwiseParallel(),
        "encoder.layers.*.mlp.gate_proj": ColwiseParallel(),
        "encoder.layers.*.mlp.down_proj": RowwiseParallel(),
        "diffusion_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        "encoder.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "encoder.norm": SequenceParallel(),
        "encoder.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "encoder.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "encoder.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
        "encoder.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "diffusion_head": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1),
            use_local_output=False,
        ),
    }

    if sequence_parallel:
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_nemotron_labs_diffusion)

__all__ = ["PARALLELIZER"]
