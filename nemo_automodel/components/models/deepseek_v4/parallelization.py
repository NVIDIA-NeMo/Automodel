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

"""Model-owned distributed parallelization for DeepSeek-V4 and DeepSeek-V4.1."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.deepseek_v4.fsdp import fully_shard_deepseek_v4

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.mesh import MeshContext


class DeepseekV4ModelParallelizer(ModelParallelizer):
    """Shard every unit once while the model's strict fp32 parameters compute in fp32.

    Args:
        fp32_compute_module_names: The owning model class's
            ``_keep_in_fp32_modules_strict`` entries.
    """

    _customizes_moe_fsdp = True

    def __init__(self, fp32_compute_module_names: tuple[str, ...]) -> None:
        super().__init__()
        self.fp32_compute_module_names = fp32_compute_module_names
        self._module_names: weakref.WeakKeyDictionary[nn.Module, str] | None = None

    def parallelize(self, model: nn.Module, mesh_context: MeshContext, /) -> nn.Module:
        """Record every module's model-level name, then run the shared parallelization flow.

        The strict fp32 names are model-level; units such as ``lm_head`` or the
        vision tower only resolve their contract when their own name is known.
        """
        self._module_names = weakref.WeakKeyDictionary((module, name) for name, module in model.named_modules())
        try:
            return super().parallelize(model, mesh_context)
        finally:
            self._module_names = None

    def _fully_shard_module(self, module: nn.Module, **kwargs) -> nn.Module:
        wrapped = module._checkpoint_wrapped_module if isinstance(module, CheckpointWrapper) else module
        if self._module_names is None:
            module_name = ""
        elif wrapped in self._module_names:
            module_name = self._module_names[wrapped]
        else:
            raise RuntimeError(
                f"{type(wrapped).__name__} was not part of the model when parallelization started, so its "
                "fp32 compute contract cannot be resolved."
            )
        return fully_shard_deepseek_v4(
            module,
            fp32_compute_module_names=self.fp32_compute_module_names,
            module_name=module_name,
            **kwargs,
        )


__all__ = ["DeepseekV4ModelParallelizer"]
