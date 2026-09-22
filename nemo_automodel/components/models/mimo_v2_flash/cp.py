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

"""Contiguous context parallelism for MiMo full and sliding attention."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout

_PAD_DOC_ID = 0


@dataclass
class MiMoCPContext:
    """Global sequence layout shared by every MiMo attention layer.

    Attributes:
        doc_ids: Document ids of shape [batch, global_sequence]. Zero marks padding.
        seq_start: Global offset of this rank's contiguous query interval.
        cp_size: Number of context-parallel ranks.
        original_seq_len: Sequence length before divisibility padding.
    """

    doc_ids: torch.Tensor
    seq_start: int = 0
    cp_size: int = 1
    original_seq_len: int | None = None

    @property
    def cp_enabled(self) -> bool:
        """Return whether the sequence is distributed across ranks."""
        return self.cp_size > 1

    @property
    def local_seq_len(self) -> int:
        """Return the padded sequence length owned by this rank."""
        return self.doc_ids.shape[1] // self.cp_size

    @property
    def local_doc_ids(self) -> torch.Tensor:
        """Return local document ids of shape [batch, local_sequence]."""
        return self.doc_ids[:, self.seq_start : self.seq_start + self.local_seq_len]


class _AllGatherSequence(torch.autograd.Function):
    """Autograd-aware all-gather of equal contiguous sequence shards."""

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: Any, dim: int) -> torch.Tensor:
        """Gather ``local_tensor`` along ``dim`` from every CP rank."""
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        local_tensor = local_tensor.contiguous()
        gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, local_tensor, group=group)
        ctx.group = group
        ctx.dim = dim
        ctx.rank = dist.get_rank(group)
        ctx.local_size = local_tensor.shape[dim]
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Sum consumers' gradients and return this rank's contiguous interval."""
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_size
        return grad_output.narrow(ctx.dim, start, ctx.local_size).contiguous(), None, None


def all_gather_sequence(tensor: torch.Tensor, cp_group: Any, *, dim: int) -> torch.Tensor:
    """Gather a sequence-sharded tensor while preserving its gradient path.

    Args:
        tensor: Local tensor whose sequence axis is ``dim``.
        cp_group: Context-parallel process group.
        dim: Sequence dimension.

    Returns:
        Tensor with the full global sequence on ``dim``.
    """
    return _AllGatherSequence.apply(tensor, cp_group, dim)


def build_cp_attention_mask(
    context: MiMoCPContext,
    *,
    dtype: torch.dtype,
    sliding_window: int | None,
) -> torch.Tensor:
    """Build local-query/global-key document-causal attention masking.

    Args:
        context: Global document layout and local query offset.
        dtype: Floating dtype used by attention logits.
        sliding_window: Optional number of visible causal keys.

    Returns:
        Additive mask of shape [batch, 1, local_sequence, global_sequence].
    """
    global_doc_ids = context.doc_ids
    local_doc_ids = context.local_doc_ids
    device = global_doc_ids.device
    query_positions = torch.arange(local_doc_ids.shape[1], device=device) + context.seq_start
    key_positions = torch.arange(global_doc_ids.shape[1], device=device)
    allowed = key_positions[None, :] <= query_positions[:, None]
    if sliding_window is not None:
        allowed = allowed & ((query_positions[:, None] - key_positions[None, :]) < sliding_window)
    allowed = (
        allowed.unsqueeze(0)
        & (local_doc_ids[:, :, None] == global_doc_ids[:, None, :])
        & (global_doc_ids[:, None, :] > _PAD_DOC_ID)
    )
    padding_queries = local_doc_ids[:, :, None] <= _PAD_DOC_ID
    allowed = torch.where(padding_queries, key_positions[None, None, :] == 0, allowed)
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min).unsqueeze(1)


def _pad_sequence(tensor: torch.Tensor, pad_len: int, value: float | int | bool) -> torch.Tensor:
    """Pad a [batch, sequence, ...] tensor along its sequence dimension."""
    if pad_len <= 0:
        return tensor
    padding = torch.full(
        (tensor.shape[0], pad_len, *tensor.shape[2:]),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=1)


def _global_doc_ids(batch: dict[str, Any], seq_len: int) -> torch.Tensor:
    """Resolve one global document-id tensor before sequence sharding."""
    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        return attention_mask.to(torch.int32)
    doc_ids = torch.ones(
        (batch["input_ids"].shape[0], seq_len),
        dtype=torch.int32,
        device=batch["input_ids"].device,
    )
    padding_mask = batch.get("padding_mask")
    if isinstance(padding_mask, torch.Tensor):
        doc_ids.masked_fill_(padding_mask.bool(), _PAD_DOC_ID)
    return doc_ids


def shard_batch_for_mimo_cp(
    cp_mesh,
    tp_mesh,
    batch: dict[str, Any],
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    shard_primary: bool = True,
):
    """Shard every sequence-aligned MiMo tensor into contiguous CP intervals.

    Args:
        cp_mesh: One-dimensional context-parallel mesh, or ``None``.
        tp_mesh: Unused tensor-parallel mesh required by the sharder protocol.
        batch: Full batch with token tensors shaped [batch, sequence].
        loss_mask: Optional loss mask shaped [batch, sequence].
        padding_token_id: Token used for CP divisibility padding.
        shard_primary: Whether to shard ``input_ids``. VLM batches keep the
            primary stream whole until vision features have been inserted.

    Returns:
        Null transport context, local batch, and global padding layout.
    """
    del tp_mesh
    input_ids = batch["input_ids"]
    if input_ids.ndim != 2:
        raise ValueError(f"MiMo CP expects input_ids [batch, sequence], got {tuple(input_ids.shape)}")
    original_seq_len = input_ids.shape[1]
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    doc_ids = _global_doc_ids(batch, original_seq_len)
    batch.pop("attention_mask", None)
    for key in ("seq_lens", "seq_lens_padded", "cu_seqlens", "cu_seqlens_padded", "max_seqlen", "qkv_format"):
        batch.pop(key, None)

    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(original_seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        )
    batch.setdefault("padding_mask", doc_ids <= _PAD_DOC_ID)

    pad_len = (-original_seq_len) % cp_size
    padded_seq_len = original_seq_len + pad_len
    if pad_len:
        pad_values = {"labels": -100, "position_ids": 0, "padding_mask": True}
        if shard_primary:
            pad_values["input_ids"] = padding_token_id
        for key, pad_value in pad_values.items():
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == original_seq_len:
                batch[key] = _pad_sequence(value, pad_len, pad_value)
        doc_ids = _pad_sequence(doc_ids, pad_len, _PAD_DOC_ID)
        if isinstance(loss_mask, torch.Tensor):
            loss_mask = _pad_sequence(loss_mask, pad_len, 0)

    seq_start = 0 if cp_mesh is None else cp_mesh.get_local_rank() * (padded_seq_len // cp_size)
    local_seq_len = padded_seq_len // cp_size
    seq_end = seq_start + local_seq_len
    shard_keys = ["labels", "position_ids", "padding_mask"]
    if shard_primary:
        shard_keys.append("input_ids")
    for key in shard_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == padded_seq_len:
            batch[key] = value[:, seq_start:seq_end].contiguous()
    if isinstance(loss_mask, torch.Tensor):
        batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()

    batch["mimo_cp_doc_ids"] = doc_ids
    batch["mimo_cp_seq_start"] = seq_start
    batch["mimo_cp_size"] = cp_size
    return (
        contextlib.nullcontext,
        batch,
        ShardLayout(original_seq_len=original_seq_len, padded_seq_len=padded_seq_len),
    )
