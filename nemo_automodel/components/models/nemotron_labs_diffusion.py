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

"""Model-owned parallelization for nemotron_labs_diffusion."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ParallelStyle

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.common.tp_plan import gated_decoder_tp_plan


def _parallelize_nemotron_labs_diffusion(model, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Nemotron-Labs-Diffusion uses a gated encoder and a diffusion output head."""
    return gated_decoder_tp_plan(sequence_parallel, prefix="encoder", head="diffusion_head")


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_nemotron_labs_diffusion)

__all__ = ["PARALLELIZER"]
