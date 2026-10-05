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

"""Model-owned distributed parallelization for dense Qwen3.5 models."""

from nemo_automodel.components.distributed import ModelParallelizer


class Qwen3_5ModelParallelizer(ModelParallelizer):
    """Generic dense parallelization plus Qwen3.5's context-parallel mesh install."""

    def _apply(self, model, device_mesh, dp_shard_cp_mesh_name="dp_shard_cp", **kwargs):
        """Apply generic TP/AC/FSDP and install Qwen3.5's CP mesh."""
        cp_mesh_name = dp_shard_cp_mesh_name.replace("dp_shard_", "")
        cp_enabled = cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1
        result = super()._apply(
            model,
            device_mesh,
            dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
            **kwargs,
        )
        if cp_enabled:
            from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

            cp_mesh = device_mesh[cp_mesh_name]
            for module in model.modules():
                if isinstance(module, CPAwareGatedDeltaNet):
                    module._cp_mesh = cp_mesh
            model.cp_mesh = cp_mesh
        return result


PARALLELIZER = Qwen3_5ModelParallelizer()

__all__ = ["PARALLELIZER"]
