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

"""Model-owned parallelization for mistral3_vlm."""

from __future__ import annotations

from typing import cast

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.tp_styles import (
    VocabParallelEmbedding,
)


def _parallelize_mistral3_vlm(
    model,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """TP plan for Mistral3ForConditionalGeneration (and subclasses like
    Mistral3FP8VLMForConditionalGeneration). The Ministral3 text backbone
    lives at ``model.language_model.{embed_tokens, layers.*}``; vision_tower
    and multi_modal_projector stay replicated across TP ranks.
    """
    model_prefix = "model.language_model"
    base_model_tp_plan: dict[str, ParallelStyle] = {
        f"{model_prefix}.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        f"{model_prefix}.layers.*.self_attn.q_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.k_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.v_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(),
        f"{model_prefix}.layers.*.mlp.up_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.gate_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }
    return cast(dict[str, ParallelStyle], base_model_tp_plan)


PARALLELIZER = ModelParallelizer(
    tp_plan=_parallelize_mistral3_vlm,
    layer_group_paths={
        "language": ("model.language_model.layers",),
        "vision": (
            "model.vision_tower.encoder.layers",
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.transformer.layers",
        ),
    },
)

__all__ = ["PARALLELIZER"]

LAYOUT_PARALLELIZER = ModelParallelizer(layer_group_paths=PARALLELIZER.layer_group_paths)
