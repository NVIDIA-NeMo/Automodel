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

"""Parallelization contract for the remote-code ``NemotronFlashForCausalLM``."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Shard

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.models.llama.parallelization import llama_tp_plan


def nemotron_flash_adjust_tp_plan(model: nn.Module, plan: dict[str, ParallelStyle]) -> dict[str, ParallelStyle]:
    """Keep ``lm_head`` sharding consistent with Nemotron-Flash's normalized logits.

    Its forward computes ``logits / self.lm_head.weight.norm(p=2, dim=1)``, so the logits and
    the weight norm must either both be plain tensors or both use the same vocab-sharded
    DTensor layout. A replicated-output ColwiseParallel plan mixes a plain logits tensor with a
    sharded weight norm, so that entry is dropped; a vocab-sharded output keeps both operands
    aligned and is required under FSDP+TP.
    """
    for k in ("lm_head", "language_model.lm_head"):
        style = plan.get(k)
        output_layouts = getattr(style, "output_layouts", ())
        if not isinstance(output_layouts, (tuple, list)):
            output_layouts = (output_layouts,)
        if any(isinstance(layout, Shard) for layout in output_layouts):
            continue
        plan.pop(k, None)
    return plan


class NemotronFlashForCausalLM:
    """Contract for the remote-code ``NemotronFlashForCausalLM``: the Llama plan plus the ``lm_head`` fix-up."""

    parallel_spec: ParallelSpec = ParallelSpec(tp_plan=llama_tp_plan, adjust_tp_plan=nemotron_flash_adjust_tp_plan)
