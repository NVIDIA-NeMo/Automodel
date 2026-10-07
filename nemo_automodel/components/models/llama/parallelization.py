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

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.common.tp_plan import gated_decoder_tp_plan


def _parallelize_llama(model: nn.Module | None, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Shard both separate and fused Llama projections."""
    plan = gated_decoder_tp_plan(sequence_parallel)
    for name in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
        plan[f"model.layers.*.{name}"] = ColwiseParallel()
    if not sequence_parallel:
        plan["model.embed_tokens"] = RowwiseParallel(input_layouts=Replicate())
    return plan


def get_llama_nemotron_super_tp_plan(sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Legacy named plan for Llama-3.3-Nemotron Super's fused projections."""
    return _parallelize_llama(None, sequence_parallel)


class LlamaParallelizer(ModelParallelizer):
    """Own the llama tensor-parallel policy."""

    tp_plan = staticmethod(_parallelize_llama)

    def _finalize_tp_plan(self, model: nn.Module, plan: dict[str, ParallelStyle]) -> dict[str, ParallelStyle]:
        from nemo_automodel.components.models.nemotron_flash import (
            adjust_flash_tp_plan,
            is_nemotron_flash_config,
        )

        return adjust_flash_tp_plan(plan) if is_nemotron_flash_config(getattr(model, "config", None)) else plan


PARALLELIZER = LlamaParallelizer()

__all__ = ["PARALLELIZER"]
