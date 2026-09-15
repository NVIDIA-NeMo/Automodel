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

"""Tensor-parallel plan shared by Qwen2 / Qwen3 dense decoders (native and transformers classes)."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import (
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# Qwen2 / Qwen3 causal LMs (the native combined ``qkv_proj`` / ``gate_up_proj`` keys are unmatched on HF checkpoints).
QWEN_TP_PLAN: dict[str, ParallelStyle] = {
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    "model.embed_tokens": VocabParallelEmbedding(
        input_layouts=Replicate(),
    ),
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
    "model.layers.*.self_attn.o_proj": RowwiseParallel(),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
    "model.layers.*.mlp.down_proj": RowwiseParallel(),
}

# Overlaid on ``QWEN_TP_PLAN`` when sequence parallelism is requested. Qwen3 has ``q_norm`` / ``k_norm`` inside
# attention operating on head-sharded activations: they must NOT get ``SequenceParallel``, which would tag them
# as sequence-sharded.
QWEN_SEQUENCE_PARALLEL_PLAN: dict[str, ParallelStyle] = {
    "lm_head": ColwiseParallel(
        input_layouts=Shard(1),
        output_layouts=Shard(-1),
        use_local_output=False,
    ),
    "model.embed_tokens": VocabParallelEmbedding(
        input_layouts=Replicate(),
        output_layouts=Shard(1),
        # Keep DTensor outputs so HF modeling code (e.g. cache_position) can
        # observe the *global* sequence length via DTensor.shape.
        use_local_output=False,
    ),
    "model.norm": SequenceParallel(),
    "model.layers.*.input_layernorm": SequenceParallelAllGatherActivation(),
    # Rowwise projections reduce-scatter back to sequence-sharded activations.
    "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
    "model.layers.*.post_attention_layernorm": SequenceParallelAllGatherActivation(),
    "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
}

QWEN_PARALLEL_SPEC = ParallelSpec(tp_plan=QWEN_TP_PLAN, sequence_parallel_plan=QWEN_SEQUENCE_PARALLEL_PLAN)
