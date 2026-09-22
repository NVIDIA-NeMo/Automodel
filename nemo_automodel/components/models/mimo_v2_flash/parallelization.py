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

"""MiMo-specific distributed-parallelization registration."""

from __future__ import annotations


def register_mimo_v2_parallel_strategies() -> None:
    """Register the MiMo FSDP2 strategy once for both checkpoint class names.

    EP configurations use the custom MoE parallelizer, which already installs
    model-owned CP attention. EP=1 uses the default FSDP2 strategy instead, so
    MiMo must attach its CP mesh after that strategy finishes.
    """
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
        register_parallel_strategy,
    )

    for name in ("MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM"):
        if name in PARALLELIZATION_STRATEGIES:
            continue

        @register_parallel_strategy(name=name)
        class MiMoV2ParallelizationStrategy(DefaultParallelizationStrategy):
            """Attach the CP submesh used by MiMo's contiguous K/V gather."""

            def parallelize(self, model, device_mesh, **kwargs):
                result = super().parallelize(model, device_mesh, **kwargs)
                cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
                cp_mesh = cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None
                model.cp_mesh = cp_mesh
                if cp_mesh is not None:
                    for module in model.modules():
                        setup_cp_attention = getattr(module, "setup_cp_attention", None)
                        if callable(setup_cp_attention):
                            setup_cp_attention(cp_mesh)
                return result
