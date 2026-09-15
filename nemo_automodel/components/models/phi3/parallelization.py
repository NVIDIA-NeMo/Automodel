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

"""Parallelization contract for the transformers ``Phi3ForCausalLM``."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# The fused ``qkv_proj`` attention stays replicated; MLP and ``lm_head`` are sharded. No sequence-parallel variant.
PHI3_TP_PLAN: dict[str, ParallelStyle] = {
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


class Phi3ForCausalLM:
    """Contract for the transformers ``Phi3ForCausalLM``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = ParallelSpec(tp_plan=PHI3_TP_PLAN)
