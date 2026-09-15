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

"""Parallelization contracts for the transformers ``Gemma3ForCausalLM`` and ``Gemma3ForConditionalGeneration``."""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import RotaryEmbedParallel, VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec


def gemma3_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """TP plan shared by both Gemma 3 heads.

    The text backbone sits at ``model`` for ``Gemma3ForCausalLM`` and at ``model.language_model``
    for ``Gemma3ForConditionalGeneration``; every other rule is identical.
    """
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3ForConditionalGeneration as _HFGemma3ForConditionalGeneration,
    )

    if isinstance(model, _HFGemma3ForConditionalGeneration):
        model_prefix = "model.language_model"
    else:
        model_prefix = "model"

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

    base_model_sp_plan = {
        f"{model_prefix}.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        f"{model_prefix}.rotary_emb": RotaryEmbedParallel(use_local_output=True),
        f"{model_prefix}.rotary_emb_local": RotaryEmbedParallel(use_local_output=True),
        f"{model_prefix}.layers.*.input_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        f"{model_prefix}.layers.*.post_attention_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.pre_feedforward_layernorm": SequenceParallel(),
        f"{model_prefix}.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        f"{model_prefix}.layers.*.post_feedforward_layernorm": SequenceParallel(),
        f"{model_prefix}.norm": SequenceParallel(),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False),
    }

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


# Layer containers list every known location across transformers releases; the first
# candidate that resolves wins, so no version gating is needed. Canonical paths come first:
# standardized 4.x releases keep deprecated top-level alias properties (``language_model``,
# ``vision_tower``) that also resolve, and first-match-wins must not pick the alias. Shapes
# verified by meta-instantiating each class on transformers 4.51.3, 4.57.1, 5.8.1 and 5.12.1.
GEMMA3_LAYERS = {
    "language": ("model.language_model.layers", "language_model.model.layers"),
    "vision": (
        "model.vision_tower.vision_model.encoder.layers",
        "model.vision_tower.encoder.layers",
        "vision_tower.vision_model.encoder.layers",
    ),
}


class Gemma3ForCausalLM:
    """Contract for the transformers ``Gemma3ForCausalLM``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = ParallelSpec(tp_plan=gemma3_tp_plan)


class Gemma3ForConditionalGeneration:
    """Contract for the transformers ``Gemma3ForConditionalGeneration`` (SigLIP tower + Gemma 3 text)."""

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=gemma3_tp_plan,
        layer_groups=GEMMA3_LAYERS,
        text_config_path="config.text_config",
        # Pre-standardization releases hang the text tower off a top-level ``language_model``.
        hf_tp_plan_prefix=("model", "language_model"),
    )
