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
"""Distributed-parallelization registration for Qwen3.8-Flash-Next."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed import ModelParallelizer


class Qwen3_8_FlashNextModelParallelizer(ModelParallelizer):
    """Reject dense FSDP2 for PLE checkpoints, which never binds the PLE owner group."""

    def _apply(self, model: nn.Module, device_mesh: DeviceMesh, **kwargs) -> nn.Module:
        """Apply dense FSDP2, which the shared dispatch selects only for ``ep_size == 1``.

        Only the MoE parallelizer (``ep_size > 1``) binds the PLE table's owner
        group to the stage's data/context shard mesh. The check reads the
        config rather than the stage's modules, so every pipeline stage raises
        together instead of only the stage that holds the table.

        Args:
            model: Qwen3.8-Flash-Next model or pipeline stage.
            device_mesh: Device mesh of the dense FSDP2 path.
            **kwargs: Remaining arguments of ``ModelParallelizer._apply``.

        Returns:
            The parallelized model.
        """
        if model.config.text_config.ple_layer_ids:
            raise ValueError(
                "Qwen3.8-Flash-Next with PLE layers requires distributed.ep_size > 1: "
                "only the MoE parallelizer binds the owner-sharded PLE table."
            )
        return super()._apply(model, device_mesh, **kwargs)


PARALLELIZER = Qwen3_8_FlashNextModelParallelizer()

__all__ = ["PARALLELIZER"]
