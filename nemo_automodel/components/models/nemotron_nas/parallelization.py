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

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import VocabParallelEmbedding
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

logger = logging.getLogger(__name__)


# DeciLM/Nemotron-NAS is close to Llama structurally, but its remote-code forward path performs model-level rotary
# embedding setup and per-layer block-config dispatch, so the separate-projection base-style plan is the safer match.
DECILM_NEMOTRON_TP_PLAN: dict[str, ParallelStyle] = {
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

DECILM_NEMOTRON_SEQUENCE_PARALLEL_PLAN: dict[str, ParallelStyle] = {
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


class NemotronNASParallelizationStrategy(DefaultParallelizationStrategy):
    """The default flow preceded by the per-layer head-count validation Nemotron-NAS block configs need."""

    def parallelize(self, model: nn.Module, device_mesh: DeviceMesh, tp_mesh_name: str = "tp", **kwargs) -> nn.Module:
        tp_size = device_mesh[tp_mesh_name].size() if tp_mesh_name in (device_mesh.mesh_dim_names or ()) else 1
        if tp_size > 1:
            validate_tp_mesh_for_nemotron_nas(model, tp_size)
        return super().parallelize(model, device_mesh, tp_mesh_name=tp_mesh_name, **kwargs)


class DeciLMForCausalLM:
    """Contract for the remote-code ``DeciLMForCausalLM``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=DECILM_NEMOTRON_TP_PLAN,
        sequence_parallel_plan=DECILM_NEMOTRON_SEQUENCE_PARALLEL_PLAN,
        strategy=NemotronNASParallelizationStrategy(),
    )
