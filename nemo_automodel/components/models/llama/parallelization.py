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

"""Model-owned parallelization for llama."""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.tp_styles import (
    SequenceParallelAllGatherActivation,
    VocabParallelEmbedding,
)


def _parallelize_llama(
    model: nn.Module | None,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a LlamaForCausalLM model across data and tensor parallel dimensions."""
    base_model_tp_plan: dict[str, ParallelStyle] = {
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

    base_model_sp_plan = {
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

    if sequence_parallel:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def get_llama_nemotron_super_tp_plan(
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Return the tensor parallel plan for Llama / Llama-3.3-Nemotron Super.

    Same topology as Llama-3.3-Nemotron (e.g. nvidia/Llama-3_3-Nemotron-Super-49B-v1_5):
    fused QKV, fused gate+up, VocabParallelEmbedding, Row/ColwiseParallel for attention and MLP.

    Use this plan explicitly by passing it as tp_shard_plan (dict) or by name
    ``llama_nemotron_super_tp_plan`` when calling fsdp2_strategy_parallelize / _get_parallel_plan.
    """
    return _parallelize_llama(None, sequence_parallel)


class LlamaParallelizer(ModelParallelizer):
    """Own the llama tensor-parallel policy."""

    tp_plan = staticmethod(_parallelize_llama)

    def _finalize_tp_plan(self, model: nn.Module, plan: dict[str, ParallelStyle]) -> dict[str, ParallelStyle]:
        from nemo_automodel.components.models.nemotron_flash.parallelization import (
            adjust_flash_tp_plan,
            is_nemotron_flash_config,
        )

        return adjust_flash_tp_plan(plan) if is_nemotron_flash_config(getattr(model, "config", None)) else plan


PARALLELIZER = LlamaParallelizer()

__all__ = ["PARALLELIZER"]
