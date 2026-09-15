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

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import RotaryEmbedParallel, VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec


def _gemma3_plans(model_prefix: str) -> tuple[dict[str, ParallelStyle], dict[str, ParallelStyle]]:
    """Gemma 3 text-backbone plan and its sequence-parallel overlay, rooted at ``model_prefix``."""
    tp_plan: dict[str, ParallelStyle] = {
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
    sequence_parallel_plan: dict[str, ParallelStyle] = {
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
    return tp_plan, sequence_parallel_plan


# The text backbone sits at ``model`` for ``Gemma3ForCausalLM`` and at ``model.language_model`` for the VLM.
GEMMA3_TP_PLAN, GEMMA3_SEQUENCE_PARALLEL_PLAN = _gemma3_plans("model")
GEMMA3_VLM_TP_PLAN, GEMMA3_VLM_SEQUENCE_PARALLEL_PLAN = _gemma3_plans("model.language_model")


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

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=GEMMA3_TP_PLAN, sequence_parallel_plan=GEMMA3_SEQUENCE_PARALLEL_PLAN
    )


class Gemma3ForConditionalGeneration:
    """Contract for the transformers ``Gemma3ForConditionalGeneration`` (SigLIP tower + Gemma 3 text)."""

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=GEMMA3_VLM_TP_PLAN,
        sequence_parallel_plan=GEMMA3_VLM_SEQUENCE_PARALLEL_PLAN,
        layer_groups=GEMMA3_LAYERS,
        text_config_path="config.text_config",
        # Pre-standardization releases hang the text tower off a top-level ``language_model``.
        hf_tp_plan_prefix=("model", "language_model"),
    )
