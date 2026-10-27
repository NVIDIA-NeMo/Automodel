# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""FSDP sidecar that keeps the Step 5 final norm in an FP32 unit."""

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed import ModelParallelizer

_FP32_NORM_POLICY = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    cast_forward_inputs=False,
)


class Step5ModelParallelizer(ModelParallelizer):
    """Keep the final decoder norm in its own FP32 FSDP unit."""

    _customizes_moe_fsdp = True

    def _fully_shard_module(self, module: nn.Module, **kwargs) -> nn.Module:
        """Shard a module, first giving any contained final norm an FP32 unit.

        Decoder blocks already group their strict-FP32 norms by dtype. The
        final norm is owned by the root unit, which uses the BF16 parameter
        policy, so it is wrapped separately before that root.

        Args:
            module: Module to shard in place.
            **kwargs: ``fully_shard`` keyword arguments (mesh, mp_policy,
                offload_policy, reshard_after_forward, ignored_params).

        Returns:
            The input module with FSDP applied.
        """
        from nemo_automodel.components.models.step5.model import Step5TextModel

        # The root unit may be the text model itself or its outer CausalLM/VLM wrapper.
        norm_kwargs = {key: value for key, value in kwargs.items() if key != "ignored_params"}
        for submodule in module.modules():
            if (
                isinstance(submodule, Step5TextModel)
                and submodule.norm is not None
                and not isinstance(submodule.norm, FSDPModule)
            ):
                fully_shard(submodule.norm, **{**norm_kwargs, "mp_policy": _FP32_NORM_POLICY})
        return fully_shard(module, **kwargs)


PARALLELIZER = Step5ModelParallelizer()

__all__ = ["PARALLELIZER"]
