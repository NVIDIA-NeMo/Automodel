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

"""Keep GLM-5.3's tiny FP32 holders gathered between forward and backward."""

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.glm5_next.layers import (
    Glm5NextHyperConnectionFp32Params,
    Glm5NextKDAFp32Params,
)


class Glm5NextModelParallelizer(ModelParallelizer):
    """Retain native FSDP gradient reduction for gathered FP32 holders."""

    _customizes_moe_fsdp = True

    def _fully_shard_module(
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy | None = None,
        reshard_after_forward: bool | int | None = None,
        ignored_params: set[nn.Parameter] | None = None,
    ) -> nn.Module:
        """Apply FSDP, retaining only the small model-owned FP32 holders.

        Args:
            module: Native module to shard. Only the mHC and KDA FP32
                parameter holders override the caller's reshard policy.
            mesh: Existing FSDP mesh, unchanged for holder parameters.
            mp_policy: Existing compute/reduction precision policy.
            offload_policy: Existing parameter/gradient offload policy.
            reshard_after_forward: Caller policy for every non-holder module.
            ignored_params: Parameters already owned by another parallelism
                boundary; their membership is preserved.

        Returns:
            The module wrapped by the native FSDP2 primitive.
        """
        if isinstance(module, (Glm5NextHyperConnectionFp32Params, Glm5NextKDAFp32Params)):
            reshard_after_forward = False
        return fully_shard(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params,
        )


PARALLELIZER = Glm5NextModelParallelizer()

__all__ = ["PARALLELIZER"]
