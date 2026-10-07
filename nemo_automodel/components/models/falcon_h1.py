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

"""Model-owned parallelization for falcon_h1."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.common.tp_plan import gated_decoder_tp_plan


def _parallelize_falcon_h1(model, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Shard attention and feed_forward; keep the parallel Mamba2 branch replicated.

    HF supplies only a head plan. The full backbone plan prevents large variants
    from OOMing. SP remains disabled because the Mamba2 branch emits full-sequence
    activations that must combine with the attention output.
    """
    plan = gated_decoder_tp_plan()
    plan["lm_head"] = ColwiseParallel(output_layouts=Replicate())
    return {name.replace(".mlp.", ".feed_forward."): style for name, style in plan.items()}


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_falcon_h1)

__all__ = ["PARALLELIZER"]
