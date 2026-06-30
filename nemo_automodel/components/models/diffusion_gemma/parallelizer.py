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

"""DiffusionGemma-specific distributed parallelization."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy

from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy, ParallelizationStrategy
from nemo_automodel.components.models.diffusion_gemma.fsdp import fully_shard_diffusion_gemma


class DiffusionGemmaParallelizationStrategy(DefaultParallelizationStrategy):
    """Pure-FSDP2 strategy that shards grouped experts as their own units."""

    def _shard_module(
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        offload_policy: OffloadPolicy | None,
        reshard_after_forward: bool | None,
    ) -> nn.Module:
        """Wrap grouped experts before their parent FSDP unit."""
        return fully_shard_diffusion_gemma(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
        )


_STRATEGY = DiffusionGemmaParallelizationStrategy()


def get_parallelization_strategy() -> ParallelizationStrategy:
    """Return the lazily requested DiffusionGemma strategy instance."""
    return _STRATEGY
