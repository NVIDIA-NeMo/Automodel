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

"""Model-owned parallelization for gemma3."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor.parallel import ParallelStyle, SequenceParallel

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce
from nemo_automodel.components.distributed.tp_styles import (
    RotaryEmbedParallel,
)
from nemo_automodel.components.models.common.tp_plan import gated_decoder_tp_plan


def _parallelize_gemma3(model: nn.Module | None, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Gemma3 uses four block norms and both global and local rotary embeddings."""
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

    prefix = "model.language_model" if isinstance(model, Gemma3ForConditionalGeneration) else "model"
    plan = gated_decoder_tp_plan(sequence_parallel, prefix=prefix)
    for name in ("q_norm", "k_norm"):
        plan[f"{prefix}.layers.*.self_attn.{name}"] = ReplicatedWithGradAllReduce()
    if sequence_parallel:
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            plan[f"{prefix}.layers.*.{name}"] = SequenceParallel()
        for name in ("rotary_emb", "rotary_emb_local"):
            plan[f"{prefix}.{name}"] = RotaryEmbedParallel(use_local_output=True)
    return plan


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_gemma3)

__all__ = ["PARALLELIZER"]


VLM_PARALLELIZER = ModelParallelizer(
    tp_plan=_parallelize_gemma3,
    layer_group_paths={
        "language": ("model.language_model.layers", "language_model.model.layers"),
        "vision": (
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ),
    },
)

LAYOUT_PARALLELIZER = ModelParallelizer(layer_group_paths=VLM_PARALLELIZER.layer_group_paths)
