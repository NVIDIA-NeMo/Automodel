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

"""Budgeted BF16 retention for existing decoder-owned FSDP units."""

import logging
import math
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer
from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.layers import Step5DecoderLayer

if TYPE_CHECKING:
    from nemo_automodel.components.models.step5.model import Step5ForCausalLM, Step5TextModel

logger = logging.getLogger(__name__)


def _retain_bf16_fsdp_units(text_model: "Step5TextModel", *, budget_mib: int) -> None:
    """Retain the smallest eligible groups within one local part's weight budget.

    Uses global DTensor shapes and existing FSDP ownership, so all FSDP ranks
    select the same units. Only dimension-zero sharding and replication are
    supported. Padding is included in the complete gathered BF16 payload;
    the original shard remains allocated. No process groups, parameter copies
    or optimizer groups are created. Backward resharding remains unchanged.
    A positive budget requires eligible units to initially reshard after forward;
    an existing no-reshard policy would otherwise invalidate the memory bound.

    Args:
        text_model: One local text part after native distributed wrapping.
            Its decoder blocks expose their routed-expert parameter ownership.
            Embeddings, final norm and language head are outside the selection.
        budget_mib: Nonnegative payload budget in MiB for this local part.
            Zero leaves all existing policies unchanged.
    """
    if budget_mib == 0:
        return
    candidates: list[tuple[int, str, FSDPModule]] = []
    for layer_name, wrapped_block in text_model.layers.items():
        block = cast(Step5DecoderLayer, wrapped_block)
        experts = set(block.moe.experts.parameters()) if block.moe is not None else set()
        for name, module in block.named_modules():
            if not isinstance(module, FSDPModule):
                continue
            # There is no public owned-parameter iterator. Use the same FSDP
            # state contract as AutoModel's synchronization helpers: one
            # uniform parameter group per dtype-aware unit. This excludes
            # nested FSDP parameters and EP's ignored local experts correctly.
            state = module._get_fsdp_state()
            group = state._fsdp_param_group
            if group is None or state._mp_policy.param_dtype not in (None, torch.bfloat16):
                continue
            owned = {parameter.sharded_param for parameter in group.fsdp_params}
            if not owned or owned.intersection(experts) or any(p.dtype != torch.bfloat16 for p in owned):
                continue
            payload = 0
            for parameter in owned:
                if not isinstance(parameter, DTensor) or parameter.ndim == 0:
                    break
                if any(
                    not isinstance(placement, Replicate) and not (isinstance(placement, Shard) and placement.dim == 0)
                    for placement in parameter.placements
                ):
                    break
                shards = math.prod(
                    parameter.device_mesh.size(axis)
                    for axis, placement in enumerate(parameter.placements)
                    if isinstance(placement, Shard)
                )
                padded_rows = ((parameter.shape[0] + shards - 1) // shards) * shards
                payload += padded_rows * math.prod(parameter.shape[1:]) * parameter.element_size()
            else:
                if group.post_forward_mesh_info is None:
                    raise ValueError(
                        "bf16_fsdp_retention_mib_per_part requires reshard_after_forward=True "
                        f"for eligible decoder units; layers.{layer_name}.{name} already retains weights"
                    )
                candidates.append((payload, f"layers.{layer_name}.{name}", module))
    retained_bytes, count = 0, 0
    for payload, name, module in sorted(candidates, key=lambda item: (item[0], item[1])):
        if retained_bytes + payload > budget_mib * 1024**2:
            continue
        module.set_reshard_after_forward(False, recurse=False)
        retained_bytes += payload
        count += 1
        logger.debug("Retained BF16 FSDP unit %s (%d padded bytes)", name, payload)
    logger.info(
        "Retained %d BF16 FSDP units (%d padded bytes, budget %d MiB per local part) until backward",
        count,
        retained_bytes,
        budget_mib,
    )


class Step5Parallelizer(ModelParallelizer):
    """Apply the shared parallelizer, then the opt-in decoder retention budget."""

    def parallelize(self, model: nn.Module, mesh_context: MeshContext, /) -> nn.Module:
        """Preserve shared wrapping, precision, expert ownership and gradient sync."""
        model = cast("Step5ForCausalLM", super().parallelize(model, mesh_context))
        if isinstance(model.backend, Step5BackendConfig):
            _retain_bf16_fsdp_units(model.model, budget_mib=model.backend.bf16_fsdp_retention_mib_per_part)
        return model
