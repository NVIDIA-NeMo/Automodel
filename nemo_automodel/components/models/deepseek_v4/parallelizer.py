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

"""DeepSeek-V4-specific distributed parallelization."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy

from nemo_automodel.components.distributed import parallelizer


class DeepseekV4ParallelizationStrategy(parallelizer.DefaultParallelizationStrategy):
    """Keep DeepSeek-V4's reference-sensitive parameters in fp32."""

    def _shard_module(
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        offload_policy: OffloadPolicy | None,
        reshard_after_forward: bool | None,
    ) -> nn.Module:
        """Apply DeepSeek-V4's dtype-aware FSDP wrapping to one module."""
        from .fsdp import fully_shard_deepseek_v4

        return fully_shard_deepseek_v4(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
        )


_STRATEGY = DeepseekV4ParallelizationStrategy()


def get_parallelization_strategy() -> parallelizer.ParallelizationStrategy:
    """Return the lazily requested DeepSeek-V4 strategy instance."""
    return _STRATEGY
