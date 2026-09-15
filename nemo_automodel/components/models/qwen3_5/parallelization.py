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

"""Qwen3.5 dense parallelization: dtype-aware FSDP2 and the context-parallel mesh hand-off."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy


class Qwen3_5ParallelizationStrategy(DefaultParallelizationStrategy):
    """The default flow, then the context-parallel mesh handed to the GatedDeltaNet layers and the model."""

    def parallelize(
        self, model: nn.Module, device_mesh: DeviceMesh, dp_shard_cp_mesh_name: str = "dp_shard_cp", **kwargs
    ) -> nn.Module:
        """Run the default flow; under context parallelism, publish the CP submesh the forward shards with."""
        result = super().parallelize(model, device_mesh, dp_shard_cp_mesh_name=dp_shard_cp_mesh_name, **kwargs)

        cp_mesh_name = dp_shard_cp_mesh_name.replace("dp_shard_", "")
        if cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1:
            from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

            cp_mesh = device_mesh[cp_mesh_name]
            for _, mod in model.named_modules():
                if isinstance(mod, CPAwareGatedDeltaNet):
                    mod._cp_mesh = cp_mesh
            # Hand the CP submesh to the model so a forward that embeds and
            # sequence-shards its own primary stream (Megatron-style per-microbatch
            # CP; see shard_sequence_for_cp_round_robin / shard_batch_aux_only) can build this
            # rank's round-robin shard.
            model.cp_mesh = cp_mesh

        return result


# The transformers class and this native port share plan and strategy. No declared TP plan: Qwen3.5 mixes full
# self_attn (every 4th layer) with GatedDeltaNet ``linear_attn``, which is not TP-shardable with stock kernels, so
# the parallelizer translates transformers' own ``base_model_tp_plan`` (self_attn + MLP only). ``shard_by_dtype``
# keeps the fp32 SSM-gate holders (``_fp32_params``, pinned through ``_keep_in_fp32_modules_strict``) in their own
# fp32 FSDP units while the rest of each block computes in the mixed-precision dtype.
QWEN3_5_PARALLEL_SPEC = ParallelSpec(shard_by_dtype=True, strategy=Qwen3_5ParallelizationStrategy())
