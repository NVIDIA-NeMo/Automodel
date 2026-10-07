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

"""Model-owned parallelization for mistral3."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor.parallel import ParallelStyle

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.common.tp_plan import gated_decoder_tp_plan


def _parallelize_ministral3(model: nn.Module | None, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Ministral3 uses the shared separate-projection decoder topology."""
    return gated_decoder_tp_plan(sequence_parallel)


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_ministral3)

__all__ = ["PARALLELIZER"]
