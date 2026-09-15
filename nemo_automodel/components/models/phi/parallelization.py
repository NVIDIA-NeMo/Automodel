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

"""Parallelization contract for the transformers ``PhiForCausalLM`` (Phi-2)."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# Phi-2 uses ``self_attn.dense`` instead of ``self_attn.o_proj`` and ``mlp.fc1`` / ``mlp.fc2`` instead of
# ``mlp.gate_proj`` / ``mlp.up_proj`` / ``mlp.down_proj``.
PHI_TP_PLAN: dict[str, ParallelStyle] = {
    "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.dense": RowwiseParallel(),
    "model.layers.*.mlp.fc1": ColwiseParallel(),
    "model.layers.*.mlp.fc2": RowwiseParallel(),
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
}

PHI_SEQUENCE_PARALLEL_PLAN: dict[str, ParallelStyle] = {
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


class PhiForCausalLM:
    """Contract for the transformers ``PhiForCausalLM``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = ParallelSpec(tp_plan=PHI_TP_PLAN, sequence_parallel_plan=PHI_SEQUENCE_PARALLEL_PLAN)
