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

"""Shared TP topology for gated decoder blocks; sidecars supply model exceptions."""

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.tp_styles import SequenceParallelAllGatherActivation, VocabParallelEmbedding


def gated_decoder_tp_plan(
    sequence_parallel: bool = False, *, prefix: str = "model", head: str = "lm_head"
) -> dict[str, ParallelStyle]:
    """Return fresh styles for separate Q/K/V projections and a gated MLP.

    Sequence-parallel block norms gather full activations and retain DTensors;
    rowwise outputs scatter back to the sequence mesh. The vocabulary projection
    retains sharded DTensors for loss parallelism.
    """
    activation = Shard(1) if sequence_parallel else Replicate()
    plan: dict[str, ParallelStyle] = {
        f"{prefix}.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(), output_layouts=activation, use_local_output=not sequence_parallel
        ),
        head: ColwiseParallel(input_layouts=activation, output_layouts=Shard(-1), use_local_output=False),
    }
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj", "mlp.gate_proj"):
        plan[f"{prefix}.layers.*.{name}"] = ColwiseParallel()
    for name in ("self_attn.o_proj", "mlp.down_proj"):
        plan[f"{prefix}.layers.*.{name}"] = RowwiseParallel(
            output_layouts=activation, use_local_output=not sequence_parallel
        )
    if sequence_parallel:
        plan[f"{prefix}.norm"] = SequenceParallel()
        for name in ("input_layernorm", "post_attention_layernorm"):
            plan[f"{prefix}.layers.*.{name}"] = SequenceParallelAllGatherActivation(use_local_output=False)
    return plan
