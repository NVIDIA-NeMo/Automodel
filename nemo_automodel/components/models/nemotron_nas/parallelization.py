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

"""Model-owned parallelization for nemotron_nas."""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.tp_styles import (
    VocabParallelEmbedding,
)


def validate_nemotron_nas_tp_mesh(model, tp_size: int) -> None:
    """Validate the heterogeneous attention layout of a Nemotron-NAS model."""
    config = model.config
    if config.num_attention_heads % tp_size:
        raise ValueError("num_attention_heads in config does not match the TP size")
    if len(config.block_configs) < config.num_hidden_layers:
        raise ValueError("num_hidden_layers in config does not match the number of block configs")

    for index, block in enumerate(config.block_configs[: config.num_hidden_layers]):
        attention = block.attention
        if attention.replace_with_linear:
            continue
        if attention.n_heads_in_group is not None:
            num_key_value_heads = config.num_attention_heads // attention.n_heads_in_group
            if num_key_value_heads % tp_size:
                raise ValueError(f"layer {index}: num_key_value_heads in config does not match the TP size")
        elif not attention.no_op:
            raise ValueError(f"layer {index}: attention must define grouped heads, a linear replacement, or no_op")


def get_decilm_nemotron_tp_plan(
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Return a TP plan for remote-code DeciLM Nemotron-NAS checkpoints.

    DeciLM/Nemotron-NAS is close to Llama structurally, but its remote-code forward
    path performs model-level rotary embedding setup and per-layer block-config
    dispatch. In practice, the generic base-style plan is a safer match than the
    Llama-optimized named plan for this architecture.
    """
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Replicate()),
    }

    base_model_sp_plan = {
        "model.embed_tokens": VocabParallelEmbedding(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "model.layers.*.post_attention_layernorm": SequenceParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    }

    if sequence_parallel:
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return cast(dict[str, ParallelStyle], base_model_tp_plan)


def _parallelize_decilm_nemotron(
    model,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    if getattr(getattr(model, "config", None), "model_type", None) != "nemotron-nas":
        raise ValueError("DeciLM TP plan is only registered for Nemotron-NAS checkpoints")
    return get_decilm_nemotron_tp_plan(sequence_parallel=sequence_parallel)


class NemotronNasParallelizer(ModelParallelizer):
    """Own the nemotron_nas tensor-parallel policy."""

    tp_plan = staticmethod(_parallelize_decilm_nemotron)

    def _validate_tp_mesh(self, model, tp_mesh):
        if tp_mesh.size() > 1 and getattr(model.config, "model_type", None) == "nemotron-nas":
            validate_nemotron_nas_tp_mesh(model, tp_mesh.size())
        else:
            super()._validate_tp_mesh(model, tp_mesh)


PARALLELIZER = NemotronNasParallelizer()

__all__ = ["PARALLELIZER"]


def get_legacy_named_tp_plan(model: nn.Module, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Preserve the legacy Nemotron-Super plan name for both checkpoint layouts."""
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None) or ()
    if (
        architectures
        and architectures[0] == "DeciLMForCausalLM"
        and getattr(config, "model_type", None) == "nemotron-nas"
    ):
        return get_decilm_nemotron_tp_plan(sequence_parallel)
    from nemo_automodel.components.models.llama.parallelization import get_llama_nemotron_super_tp_plan

    return get_llama_nemotron_super_tp_plan(sequence_parallel)


LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME = "llama_nemotron_super_tp_plan"
