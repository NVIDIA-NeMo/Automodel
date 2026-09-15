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

"""Parallelization contract for the remote-code ``NemotronLabsDiffusionModel`` (Nemotron-Labs-Diffusion dLLM)."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import (
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# Same shape as the Ministral3 plan, but the model uses ``encoder.*`` (not ``model.*``) and the output head is
# ``diffusion_head`` (not ``lm_head``). ``NemotronLabsDiffusionModel`` is loaded via trust_remote_code.
NEMOTRON_LABS_DIFFUSION_TP_PLAN: dict[str, ParallelStyle] = {
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

NEMOTRON_LABS_DIFFUSION_SEQUENCE_PARALLEL_PLAN: dict[str, ParallelStyle] = {
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


class NemotronLabsDiffusionModel:
    """Contract for the remote-code ``NemotronLabsDiffusionModel``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=NEMOTRON_LABS_DIFFUSION_TP_PLAN,
        sequence_parallel_plan=NEMOTRON_LABS_DIFFUSION_SEQUENCE_PARALLEL_PLAN,
    )
