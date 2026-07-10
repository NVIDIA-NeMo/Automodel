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

"""Framework-owned load-balanced context-parallel batch sharding."""

from __future__ import annotations

from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.shared.cp_contracts import ContextFactory, CPBatch


def _batch_tensor(batch: CPBatch, key: str) -> torch.Tensor:
    """Read one tensor-valued batch field.

    Args:
        batch: Mapping with arbitrary tensor field layouts; no field is mutated.
        key: Field name to resolve.

    Returns:
        The exact stored tensor with shape, stride, dtype, and device unchanged.
    """
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"CP batch field {key!r} must be a torch.Tensor, got {type(value).__name__}")
    return value


def shard_batch_load_balanced(
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
    batch: CPBatch,
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
) -> tuple[ContextFactory, CPBatch]:
    """Shard a batch with torch CP head-tail load balancing.

    Args:
        cp_mesh: Active context-parallel mesh. The global sequence ``S`` is
            split into ``2 * CP`` chunks and paired head-to-tail per rank.
        tp_mesh: Optional tensor-parallel mesh; unused by this backend.
        batch: Full batch containing exactly one of IDs ``[B, S]`` or
            embeddings ``[B, S, H]``, labels ``[B, S]``, optional positions
            ``[B, S]`` or ``[R, B, S]``, and optional padding mask ``[B, S]``.
            Padding is materialized in this mapping before context entry. Here
            ``B`` is batch size, ``S`` is global sequence length, ``H`` is
            hidden size, and ``R`` is rotary-axis count.
        loss_mask: Optional token mask ``[B, S]`` sharded with the batch.
        padding_token_id: Accepted by the shared sharder protocol. Native torch
            CP currently zero-fills token padding, so this value is unused.

    Returns:
        Context factory and the same batch mapping. Entering the context mutates
        registered tensors to local ``S / CP`` shards; no-restore tensors remain
        local after exit.
    """
    del tp_mesh, padding_token_id
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("load-balanced CP sharding requires a CP mesh with size greater than one")

    from nemo_automodel.components.distributed.cp_utils import create_context_parallel_ctx, get_train_context

    batch.pop("attention_mask", None)

    has_inputs_embeds = "inputs_embeds" in batch
    has_input_ids = "input_ids" in batch
    if has_inputs_embeds == has_input_ids:
        raise ValueError("CP batch requires exactly one of 'inputs_embeds' or 'input_ids'")
    primary_key = "inputs_embeds" if has_inputs_embeds else "input_ids"
    primary_seq_tensor = _batch_tensor(batch, primary_key)
    seq_len = primary_seq_tensor.shape[1]

    batch_size = primary_seq_tensor.shape[0]
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(seq_len, device=primary_seq_tensor.device).unsqueeze(0).expand(batch_size, -1).contiguous()
        )
    else:
        position_ids = _batch_tensor(batch, "position_ids")
        if position_ids.ndim == 2 and position_ids.shape[0] == 1 and batch_size > 1:
            batch["position_ids"] = position_ids.expand(batch_size, -1).contiguous()

    position_ids = _batch_tensor(batch, "position_ids")
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1
    labels = _batch_tensor(batch, "labels")

    cp_buffers = [primary_seq_tensor, labels, position_ids]
    cp_seq_dims = [1, 1, pos_seq_dim]
    cp_no_restore_buffers = {primary_seq_tensor, labels}
    batch_buffer_keys: dict[int, str] = {0: primary_key, 1: "labels", 2: "position_ids"}

    if loss_mask is not None:
        cp_buffers.append(loss_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(loss_mask)

    if "padding_mask" in batch:
        padding_mask = _batch_tensor(batch, "padding_mask")
        batch_buffer_keys[len(cp_buffers)] = "padding_mask"
        cp_buffers.append(padding_mask)
        cp_seq_dims.append(1)
        cp_no_restore_buffers.add(padding_mask)

    pad_fill: dict[str, object] = {
        "labels": -100,
        "padding_mask": True,
    }
    cp_divisor = cp_mesh.size() * 2
    if seq_len % cp_divisor != 0:
        pad_len = cp_divisor - (seq_len % cp_divisor)
        new_no_restore: set[torch.Tensor] = set()
        for index, (buffer, seq_dim) in enumerate(zip(cp_buffers, cp_seq_dims)):
            pad_shape = list(buffer.shape)
            pad_shape[seq_dim] = pad_len
            if buffer.dtype.is_floating_point:
                pad = torch.zeros(pad_shape, dtype=buffer.dtype, device=buffer.device)
            else:
                fill = cast(float | int | bool, pad_fill.get(batch_buffer_keys.get(index, ""), 0))
                pad = torch.full(pad_shape, fill, dtype=buffer.dtype, device=buffer.device)
            old_buffer = buffer
            cp_buffers[index] = torch.cat((buffer, pad), dim=seq_dim)
            if old_buffer in cp_no_restore_buffers:
                new_no_restore.add(cp_buffers[index])
        cp_no_restore_buffers = new_no_restore
        for index, key in batch_buffer_keys.items():
            batch[key] = cp_buffers[index]

    cp_ctx = create_context_parallel_ctx(
        cp_mesh=cp_mesh,
        cp_buffers=cp_buffers,
        cp_seq_dims=cp_seq_dims,
        cp_no_restore_buffers=cp_no_restore_buffers,
        cp_rotate_method="allgather",
    )
    return get_train_context(False, False, cp_ctx), batch
