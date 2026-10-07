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

"""Model-owned parallelization for phi."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce
from nemo_automodel.components.distributed.tp_styles import (
    VocabParallelEmbedding,
)

if TYPE_CHECKING:
    from transformers.models.phi.modeling_phi import PhiForCausalLM


def _parallelize_phi(
    model: PhiForCausalLM,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a PhiForCausalLM (Phi-2) model across tensor parallel dimensions.

    Phi-2 uses ``self_attn.dense`` instead of ``self_attn.o_proj`` and
    ``mlp.fc1``/``mlp.fc2`` instead of ``mlp.gate_proj``/``mlp.up_proj``/``mlp.down_proj``.
    """
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.dense": RowwiseParallel(),
        "model.layers.*.mlp.fc1": ColwiseParallel(),
        "model.layers.*.mlp.fc2": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    if model.config.qk_layernorm:
        base_model_tp_plan.update(
            {
                "model.layers.*.self_attn.q_layernorm": ReplicatedWithGradAllReduce(),
                "model.layers.*.self_attn.k_layernorm": ReplicatedWithGradAllReduce(),
            }
        )

    if sequence_parallel:
        base_model_sp_plan: dict[str, ParallelStyle] = {
            "model.embed_tokens": VocabParallelEmbedding(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "model.final_layernorm": SequenceParallel(),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.self_attn.dense": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            "model.layers.*.mlp.fc2": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
        }
        base_model_tp_plan.update(base_model_sp_plan)

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_phi)

__all__ = ["PARALLELIZER"]
