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

"""Contiguous context parallelism for the V4.1 text backbone."""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_gather

from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout, shard_batch_contiguous


def gather_sequence(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Gather rank-ordered sequence shards, summing remote uses in backward.

    Args:
        tensor: Tensor of shape [batch, local_sequence, ...], with arbitrary
            trailing dimensions. Equal sequence lengths are required on every rank.
        group: CP process group, or None for an identity operation.

    Returns:
        Tensor of shape [batch, global_sequence, ...]. Floating activations retain
        autograd history; integer and boolean metadata are gathered without gradients.
        At CP1 the result aliases the input. No input is mutated.
    """
    if group is None or dist.get_world_size(group) == 1:
        return tensor
    if tensor.requires_grad:
        parts = all_gather(tensor.contiguous(), group=group)
    else:
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=1)


def shard_cp_batch(
    cp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh | None,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
    pad_multiple: int = 1,
) -> tuple[Callable, dict[str, Any], ShardLayout]:
    """Keep complete compression groups on each rank and propagate the CP group.

    Args:
        cp_mesh: Context-parallel mesh.
        tp_mesh: Optional tensor-parallel mesh.
        batch: Unpacked text tensors input_ids, labels, position_ids, and optional
            attention_mask/padding_mask of shape [batch, global_sequence]. Mutated
            to [batch, local_sequence]. Position IDs remain global and zero-based.
        loss_mask: Optional tensor of shape [batch, global_sequence].
        padding_token_id: Input token used for right padding.
        pad_multiple: LCM of this model's active compression ratios.

    Returns:
        Null context factory, the local batch with a runtime cp_group, and its
        original/padded global sequence lengths. Token tensors use independent
        local storage; existing tensors are not modified in place.
    """
    if batch.get("seq_lens") is not None or batch.get("cu_seqlens") is not None or batch.get("qkv_format") == "thd":
        raise ValueError("DeepSeek V4.1 context parallelism requires unpacked text sequences")
    if batch.get("pixel_values") is not None or batch.get("inputs_embeds") is not None:
        raise ValueError("DeepSeek V4.1 context parallelism currently supports text input_ids only")
    if "padding_mask" not in batch and "attention_mask" not in batch:
        batch["padding_mask"] = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    ctx, batch, layout = shard_batch_contiguous(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        pad_multiple=pad_multiple,
    )
    batch["attention_mask"] = ~batch.pop("padding_mask")
    batch["cp_group"] = cp_mesh.get_group()
    return ctx, batch, layout
