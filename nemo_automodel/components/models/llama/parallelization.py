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

"""Tensor-parallel plan for llama-style dense decoders (shared by the transformers class)."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import (
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# Llama-style decoder; the fused ``qkv_proj`` / ``gate_up_proj`` keys cover the native combined projections
# and are simply unmatched on separate-projection checkpoints.
LLAMA_TP_PLAN: dict[str, ParallelStyle] = {
    "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Combined QKV projection
    "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate and up projection
    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.down_proj": RowwiseParallel(),
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
}

# Overlaid on ``LLAMA_TP_PLAN`` when sequence parallelism is requested.
LLAMA_SEQUENCE_PARALLEL_PLAN: dict[str, ParallelStyle] = {
    "model.embed_tokens": VocabParallelEmbedding(
        input_layouts=Replicate(),
        output_layouts=Shard(1),
        use_local_output=False,
    ),
    "model.norm": SequenceParallel(),
    "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
    "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
    "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(use_local_output=False),
    "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
    "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
}


# Also the topology of Llama-3.3-Nemotron Super (fused QKV and gate+up); the legacy ``llama_nemotron_super_tp_plan``
# alias resolves to the model's own declaration.
LLAMA_PARALLEL_SPEC = ParallelSpec(tp_plan=LLAMA_TP_PLAN, sequence_parallel_plan=LLAMA_SEQUENCE_PARALLEL_PLAN)
