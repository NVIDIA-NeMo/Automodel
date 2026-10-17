# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.distributed as dist
from torch import nn


class LinearCrossEntropy(nn.Module):
    """Losses consuming final hidden states and an LM-head weight instead of logits."""

    @staticmethod
    def materialize_lm_weight(
        lm_weight: torch.Tensor,
        *,
        grad_reduce_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Materialize an LM-head DTensor with gradient-correct reduction semantics.

        Linear CE consumes the LM-head weight outside the owning FSDP
        module's forward. Each data/context-parallel rank therefore computes a
        rank-local full-weight gradient. A plain ``DTensor.full_tensor()`` marks
        that gradient as replicated, so backward only slices the local result
        into the owned shard instead of combining peer contributions.

        Args:
            lm_weight: LM-head weight with global shape ``[vocab, hidden]``. A
                regular tensor is returned unchanged. A DTensor may have any
                FSDP sharding placement over its device mesh and is gathered to
                a rank-local regular tensor with the global shape, device, and
                dtype.
            grad_reduce_group: Process group whose ranks contribute independent
                token losses. Its size must match the LM-head DTensor mesh.

        Returns:
            Regular tensor with shape ``[vocab, hidden]``. For a DTensor input,
            backward reduce-scatters the averaged peer gradients into the
            original local shard. The gathered result does not alias the local
            DTensor shard; a regular-tensor input is returned by identity.

        Raises:
            ValueError: If a trainable sharded weight has no matching reduction
                group. This fails closed instead of producing rank-local shards.
        """
        if not hasattr(lm_weight, "full_tensor"):
            return lm_weight

        # Evaluation has no weight gradient to combine, so preserve the ordinary
        # gather path and do not require a process group from inference callers.
        if not torch.is_grad_enabled() or not lm_weight.requires_grad:
            return lm_weight.full_tensor()

        mesh = lm_weight.device_mesh
        mesh_world_size = mesh.size()
        reduce_world_size = dist.get_world_size(grad_reduce_group) if grad_reduce_group is not None else 1
        if mesh_world_size != reduce_world_size:
            raise ValueError(
                "LinearCrossEntropy requires grad_reduce_group to match the LM-head "
                f"DTensor mesh: mesh size={mesh_world_size}, reduction group size={reduce_world_size}. "
                "Tensor-parallel or hierarchical layouts need an explicit compatible loss path."
            )
        if mesh_world_size == 1:
            return lm_weight.full_tensor()

        from torch.distributed.tensor import Partial

        # ``Partial`` tells DTensor autograd to reduce-scatter the full gradient
        # directly into the parameter's original FSDP shard. Training recipes
        # scale the local loss by the reduction world size before backward to
        # cancel FSDP's averaged-gradient convention, so restore that average
        # before the reduce-scatter sum.
        full_weight = lm_weight.full_tensor(
            grad_placements=tuple(Partial() for _ in range(mesh.ndim)),
        )
        full_weight.register_hook(lambda grad: grad / reduce_world_size)
        return full_weight
