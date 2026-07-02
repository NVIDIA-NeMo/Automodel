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

"""Phi-3 tensor-parallel plan."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.models.common.tp_plan import VocabParallelEmbedding


def get_tp_plan(model, *, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Return Phi-3's MLP-only sharding plan.

    Phi-3's fused attention remains replicated under the stock TP kernels.
    """
    del model, sequence_parallel
    return {
        "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate(), output_layouts=Replicate()),
        "model.layers.*.self_attn.qkv_proj": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        "model.layers.*.self_attn.o_proj": ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=False
        ),
        "model.layers.*.mlp.down_proj": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }
