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

"""Model-owned parallelization for baichuan."""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel

from nemo_automodel.components.distributed import ModelParallelizer


def _parallelize_baichuan(
    model: nn.Module | None,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    """Parallelizes a BaichuanForCausalLM model (MLP-only).

    Only the MLP is sharded. The attention path stays fully replicated
    because W_pack uses a non-interleaved [Q|K|V] layout (ColwiseParallel
    would split it incorrectly) and NormHead (lm_head) is not nn.Linear
    (ColwiseParallel is unsupported).
    """
    return cast(
        dict[str, ParallelStyle],
        {
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
        },
    )


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_baichuan)

__all__ = ["PARALLELIZER"]
