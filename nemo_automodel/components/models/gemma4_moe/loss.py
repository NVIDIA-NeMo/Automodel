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

"""Loss adapters for Gemma4 tensor parallelism."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.tensor import Partial, Replicate

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy


class Gemma4TensorParallelFusedLinearCrossEntropy(FusedLinearCrossEntropy):
    """Fused linear CE for Gemma4's tied, vocabulary-sharded LM head.

    Gemma4 E-series TP keeps the tied embedding/LM-head weight sharded on the
    TP vocabulary axis. Cut cross entropy needs a rank-local full vocabulary
    weight, but the generic loss intentionally rejects a DTensor mesh that does
    not match the DP/CP loss-reduction group. This adapter gathers only the
    Gemma4 TP layout and marks its backward gradient replicated: TP peers see
    the same tokens and therefore must not sum duplicate weight gradients.
    FSDP continues to reduce independent DP/CP contributions normally.
    """

    @staticmethod
    def materialize_lm_weight(
        lm_weight: torch.Tensor,
        *,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Materialize a Gemma4 TP LM head with correct TP gradient semantics.

        Args:
            lm_weight: Regular or DTensor LM-head weight shaped
                ``[vocab, hidden]``.
            grad_reduce_group: DP/CP group whose ranks own independent tokens.
                It remains owned by FSDP when the visible DTensor mesh contains
                only TP, as it does while an FSDP unit is unsharded.

        Returns:
            A rank-local full-vocabulary tensor. Backward slices, rather than
            sums, duplicate gradients along every named TP mesh axis.
        """
        if not hasattr(lm_weight, "full_tensor"):
            return lm_weight

        mesh = lm_weight.device_mesh
        mesh_names = mesh.mesh_dim_names
        if mesh_names is None or "tp" not in mesh_names:
            return FusedLinearCrossEntropy.materialize_lm_weight(
                lm_weight,
                grad_reduce_group=grad_reduce_group,
            )

        if not torch.is_grad_enabled() or not lm_weight.requires_grad:
            return lm_weight.full_tensor()

        placements = tuple(Replicate() if name == "tp" else Partial() for name in mesh_names)
        full_weight = lm_weight.full_tensor(grad_placements=placements)

        # A Partial placement performs a sum across that mesh axis. Recipes
        # multiply the normalized rank-local loss by the DP/CP group size to
        # cancel FSDP's averaged-gradient convention, so average any reductions
        # performed here just like the generic fused loss does. A TP-only mesh
        # has no Partial axes; its DP/CP reduction stays entirely inside FSDP.
        partial_world_size = 1
        for axis, placement in enumerate(placements):
            if isinstance(placement, Partial):
                partial_world_size *= mesh.size(axis)
        if partial_world_size > 1:
            full_weight.register_hook(lambda grad: grad / partial_world_size)
        return full_weight
