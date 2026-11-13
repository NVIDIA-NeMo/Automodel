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

"""FSDP sidecar that keeps Inkling 2's final attention-residual query in an FP32 unit."""

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed import ModelParallelizer

_FP32_POLICY = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    cast_forward_inputs=False,
)


class Inkling2ModelParallelizer(ModelParallelizer):
    """Give the text model's ``final_input`` its own FP32 FSDP unit.

    Decoder blocks already shard their strict-FP32 holders (router, attention-residual queries,
    short convolutions) by dtype. ``final_input`` is owned by the text model itself, whose root
    unit uses the BF16 parameter policy next to the BF16 final norm, so it is wrapped first.
    """

    _customizes_moe_fsdp = True

    def _fully_shard_module(self, module: nn.Module, **kwargs) -> nn.Module:
        """Shard a module, first wrapping any contained ``final_input`` in an FP32 unit.

        Args:
            module: Module to shard in place.
            **kwargs: ``fully_shard`` keyword arguments (mesh, mp_policy, offload_policy,
                reshard_after_forward, ignored_params).

        Returns:
            The input module with FSDP applied.
        """
        from nemo_automodel.components.models.inkling2.text import Inkling2TextModel

        # The root unit may be the text model itself or the outer conditional-generation wrapper.
        fp32_kwargs = {key: value for key, value in kwargs.items() if key != "ignored_params"}
        for submodule in module.modules():
            if isinstance(submodule, Inkling2TextModel) and not isinstance(submodule.final_input, FSDPModule):
                fully_shard(submodule.final_input, **{**fp32_kwargs, "mp_policy": _FP32_POLICY})
        return fully_shard(module, **kwargs)


PARALLELIZER = Inkling2ModelParallelizer()

__all__ = ["PARALLELIZER"]
