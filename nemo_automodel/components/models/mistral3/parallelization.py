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

"""Parallelization contract for the transformers ``Mistral3ForConditionalGeneration`` (Pixtral + Ministral3).

The FP8 subclass in ``components/models/mistral3_vlm`` reuses :data:`MISTRAL3_VLM_PARALLEL_SPEC`.
"""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

# The Ministral3 text backbone lives at ``model.language_model.{embed_tokens, layers.*}``; vision_tower and
# multi_modal_projector stay replicated across TP ranks. No sequence-parallel variant.
MISTRAL3_VLM_TP_PLAN: dict[str, ParallelStyle] = {
    "model.language_model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
    "model.language_model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.language_model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.language_model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.language_model.layers.*.self_attn.o_proj": RowwiseParallel(),
    "model.language_model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.language_model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.language_model.layers.*.mlp.down_proj": RowwiseParallel(),
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
}


MISTRAL3_VLM_PARALLEL_SPEC = ParallelSpec(
    tp_plan=MISTRAL3_VLM_TP_PLAN,
    layer_groups={
        "language": ("model.language_model.layers",),
        "vision": (
            "model.vision_tower.encoder.layers",
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.transformer.layers",
        ),
    },
)


class Mistral3ForConditionalGeneration:
    """Contract for the transformers ``Mistral3ForConditionalGeneration``; bound by the loader onto its wrapper."""

    parallel_spec: ParallelSpec = MISTRAL3_VLM_PARALLEL_SPEC
