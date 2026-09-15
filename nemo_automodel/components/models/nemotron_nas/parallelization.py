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

"""Parallelization contract for the remote-code ``DeciLMForCausalLM`` (Llama-Nemotron NAS, ``model_type="nemotron-nas"``)."""

from __future__ import annotations

import logging
from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

logger = logging.getLogger(__name__)


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


def decilm_nemotron_tp_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """:func:`get_decilm_nemotron_tp_plan`, refusing DeciLM classes whose config is not a Nemotron-NAS one."""
    if getattr(getattr(model, "config", None), "model_type", None) != "nemotron-nas":
        raise ValueError("DeciLM TP plan is only registered for Nemotron-NAS checkpoints")
    return get_decilm_nemotron_tp_plan(sequence_parallel=sequence_parallel)


def validate_tp_mesh_for_nemotron_nas(model: nn.Module, tp_size: int) -> None:
    """Validate that a Nemotron-NAS model can be tensor-parallel sharded."""
    num_attention_heads = model.config.num_attention_heads
    assert num_attention_heads % tp_size == 0, "num_attention_heads in config does not match the TP size"

    assert len(model.config.block_configs) >= model.config.num_hidden_layers, (
        "num_hidden_layers in config does not match the number of block configs"
    )

    for i in range(model.config.num_hidden_layers):
        # Valid layer
        if model.config.block_configs[i].attention.replace_with_linear:
            logger.info("By pass checking for linear layer in layer %d", i)
            # TODO: Check if the linear layer could support TP.
        else:
            if model.config.block_configs[i].attention.n_heads_in_group is not None:
                num_key_value_heads = num_attention_heads // model.config.block_configs[i].attention.n_heads_in_group
                assert num_key_value_heads % tp_size == 0, (
                    f"layer {i}: num_key_value_heads in config does not match the TP size"
                )
            else:
                assert model.config.block_configs[i].attention.no_op == True


class DeciLMForCausalLM:
    """Contract for the remote-code ``DeciLMForCausalLM``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=decilm_nemotron_tp_plan, validate_tp=validate_tp_mesh_for_nemotron_nas
    )
