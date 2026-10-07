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

"""Model-owned parallelization for qwen2."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce
from nemo_automodel.components.distributed.tp_styles import (
    SequenceParallelAllGatherActivation,
)
from nemo_automodel.components.models.common.tp_plan import gated_decoder_tp_plan


def _parallelize_qwen(model: nn.Module | None, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Shard Qwen2/3 projections; sum partial-head gradients for replicated Q/K norms."""
    plan = gated_decoder_tp_plan(sequence_parallel)
    for name in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
        plan[f"model.layers.*.{name}"] = ColwiseParallel()
    for name in ("q_norm", "k_norm"):
        plan[f"model.layers.*.self_attn.{name}"] = ReplicatedWithGradAllReduce()
    if sequence_parallel:
        for name in ("input_layernorm", "post_attention_layernorm"):
            plan[f"model.layers.*.{name}"] = SequenceParallelAllGatherActivation()
    return plan


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_qwen)

__all__ = ["PARALLELIZER"]
