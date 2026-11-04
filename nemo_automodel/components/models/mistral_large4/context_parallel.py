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

"""Contiguous CP batch preparation and differentiable key transport for Large 4."""

import contextlib
from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
    contiguous_local_indices,
    shard_batch_contiguous,
)
from nemo_automodel.components.distributed.thd_utils import split_batch_into_thd_chunks

# Recipe batch entries: token tensors, complete-image lists and the packed-format tag.
CPBatch: TypeAlias = dict[str, torch.Tensor | list[torch.Tensor] | str | None]


def gather_tokens(value: torch.Tensor, mesh: DeviceMesh | None, *, differentiable: bool) -> torch.Tensor:
    """Gather [local_tokens, ...] tensors to [CP * local_tokens, ...] in rank order.

    Local token counts must agree. Differentiable transport sums remote-query
    contributions back to each key owner; parameter averaging remains FSDP-owned.
    """
    if mesh is None or mesh.size() == 1:
        return value
    if differentiable:
        return funcol.all_gather_tensor_autograd(value.contiguous(), 0, mesh)
    pieces = [torch.empty_like(value) for _ in range(mesh.size())]
    dist.all_gather(pieces, value.contiguous(), group=mesh.get_group())
    return torch.cat(pieces, dim=0)


def prepare_cp_inputs(
    batch: CPBatch, *, num_chunks: int = 1, shard_primary: bool = True
) -> dict[str, ContextParallelSharder]:
    """Prepare contiguous CP shards with positions computed before partitioning.

    Args:
        batch: Global BSHD token tensors [batch, sequence, ...] and optional
            complete image tensors [channels, height, width]. THD uses one packed
            row per microbatch, with seq_lens/seq_lens_padded document lengths.
        num_chunks: Positive number of PP microbatches along the batch axis.
        shard_primary: Whether to shard IDs/embeddings here. False keeps them
            global for image splicing and per-microbatch sharding in forward.

    Returns:
        Mapping containing the model-owned ContextParallelSharder. Labels,
        positions and padding become [batch, padded_sequence / CP]. With
        shard_primary=False, IDs/embeddings and the combined binary keep mask
        remain global [batch, sequence, ...]; images are never partitioned.
    """
    if batch.get("qkv_format") == "thd":
        return prepare_packed_cp_inputs(batch, num_chunks=num_chunks, shard_primary=shard_primary)
    if batch.get("qkv_format", "bshd") != "bshd" or batch.get("cu_seqlens") is not None:
        raise ValueError("Large 4 CP requires nonpacked BSHD inputs")
    if num_chunks < 1:
        raise ValueError("Large 4 CP requires num_chunks >= 1")

    def shard(
        cp_mesh: DeviceMesh,
        tp_mesh: DeviceMesh | None,
        local_batch: CPBatch,
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
    ) -> tuple[Callable, CPBatch, ShardLayout]:
        """Apply the shared contiguous layout without consuming PP media.

        Args:
            cp_mesh: CP mesh; peers receive the same global batch.
            tp_mesh: Unused TP mesh forwarded to the shared sharder.
            local_batch: Global IDs/masks [batch, sequence] or embeddings
                [batch, sequence, hidden], modified only by replacing entries.
            loss_mask: Optional global [batch, sequence] supervision weights.
            padding_token_id: Sentinel for CP padding of token IDs.

        Returns:
            Null context factory, prepared batch with layouts documented by
            prepare_cp_inputs, and its global-to-local ShardLayout. Input
            tensors retain their values; only batch mapping entries change.
        """
        primary = local_batch.get("input_ids", local_batch.get("inputs_embeds"))
        shape = primary.shape[:2]
        keep = torch.ones(shape, device=primary.device, dtype=torch.bool)
        if local_batch.get("attention_mask") is not None:
            mask = local_batch.pop("attention_mask")
            if mask.shape != shape or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("CP attention_mask must be binary and match token axes")
            keep &= mask.bool()
        if local_batch.get("padding_mask") is not None:
            padding = local_batch["padding_mask"]
            if padding.shape != shape or padding.dtype != torch.bool:
                raise ValueError("CP padding_mask must be boolean and match token axes")
            keep &= ~padding
        local_batch["padding_mask"] = ~keep
        if local_batch.get("position_ids") is None:
            local_batch["position_ids"] = (keep.long().cumsum(1) - 1).clamp_min(0)
        context, sharded, layout = shard_batch_contiguous(
            cp_mesh,
            tp_mesh,
            local_batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
            shard_primary=shard_primary,
        )
        if not shard_primary:
            # Complete image grids are validated before the forward shards them.
            # This also retains user padding when images straddle CP boundaries.
            sharded["attention_mask"] = keep
        return context, sharded, layout

    return {
        "cp_sharder": ContextParallelSharder(shard_batch=shard, local_token_global_indices=contiguous_local_indices)
    }


def shifted_cp_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Compute HF shifted CE across shard boundaries, scaled for FSDP CP averaging.

    Args:
        logits: FP32 capped logits [batch, local_sequence, vocab].
        labels: Integer unshifted labels [batch, local_sequence].
        valid: Boolean flattened local [batch * local_sequence] keep mask.
        mesh: CP mesh; peers must hold shards of the same padded global batch.

    Returns:
        Local loss sum times CP / global valid-target count, a differentiable
        scalar. Averaging peer losses equals the unsharded mean CE.
    """
    batch, length = labels.shape
    count = mesh.size()
    gathered_labels = gather_tokens(labels.reshape(-1), mesh, differentiable=False)
    gathered_valid = gather_tokens(valid, mesh, differentiable=False)
    full_labels = gathered_labels.reshape(count, batch, length).transpose(0, 1).reshape(batch, -1)
    full_valid = gathered_valid.reshape(count, batch, length).transpose(0, 1).reshape(batch, -1)
    targets = F.pad(full_labels[:, 1:], (0, 1), value=-100)
    edges = full_valid & F.pad(full_valid[:, 1:], (0, 1), value=False)
    targets = targets.masked_fill(~edges, -100)
    start = mesh.get_local_rank() * length
    local_targets = targets[:, start : start + length]
    total = F.cross_entropy(logits.flatten(0, 1), local_targets.reshape(-1), ignore_index=-100, reduction="sum")
    return total * count / (targets != -100).sum().clamp_min(1)


def prepare_packed_cp_inputs(
    batch: CPBatch, *, num_chunks: int, shard_primary: bool
) -> dict[str, ContextParallelSharder]:
    """Preserve packed documents while preparing contiguous model-owned CP.

    Args:
        batch: Global IDs/labels/positions [chunks, sequence], document lengths
            [chunks, documents], and optional complete images [channels, height, width].
            Exactly one packed row belongs to each PP microbatch.
        num_chunks: Positive number of independent pipeline microbatches; one without PP.
        shard_primary: False retains global IDs until complete-image splicing
            and/or stage-zero embedding; True shards text IDs here.

    Returns:
        Sharder yielding local IDs/labels/positions/padding/document IDs
        [chunks, padded_sequence / CP]. With shard_primary=False, IDs and the
        binary image-validation mask remain global. Packed boundaries remain
        global [chunks, documents + 1] for image validation only.
    """
    if batch.get("qkv_format") != "thd":
        raise ValueError("Mistral packed CP requires packed THD inputs")

    def shard(
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
        local_batch: CPBatch,
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
    ) -> tuple[Callable[[], contextlib.AbstractContextManager], CPBatch, ShardLayout]:
        """Convert global THD metadata then shard all token-aligned fields.

        Args:
            cp_mesh: Stage-local CP mesh, including CP1.
            tp_mesh: Unused TP mesh passed to the existing sharder.
            local_batch: Same global tensor layouts as prepare_packed_cp_inputs.
            loss_mask: Optional supervision weights [chunks, sequence], overriding batch loss_mask.
            padding_token_id: Extra CP-padding token ID.

        Returns:
            Null context, prepared batch in prepare_packed_cp_inputs layouts, and
            contiguous shard layout. Source tensor storage is never modified.
        """
        ids = local_batch["input_ids"]
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or num_chunks < 1 or ids.shape[0] != num_chunks:
            raise ValueError("Mistral packed CP requires one packed row per microbatch")
        source_mask = loss_mask if loss_mask is not None else local_batch.get("loss_mask")
        if source_mask is not None and source_mask.shape != ids.shape:
            raise ValueError("Mistral packed CP loss_mask must match global input_ids")
        source = dict(local_batch)
        if source.get("position_ids") is None:
            lengths = source["seq_lens_padded"]
            source["position_ids"] = torch.stack(
                [torch.cat([torch.arange(int(n), device=ids.device) for n in row if int(n) >= 0]) for row in lengths]
            )
        converted = split_batch_into_thd_chunks(source, padding_token_id=padding_token_id, num_chunks=num_chunks)
        cu = converted["cu_seqlens_padded"].reshape(num_chunks, -1)
        documents = []
        for boundaries in cu:
            boundaries = boundaries[boundaries >= 0].to(device=ids.device, dtype=torch.long)
            if (
                boundaries.numel() < 2
                or int(boundaries[0]) != 0
                or int(boundaries[-1]) != ids.shape[1]
                or bool((boundaries[1:] <= boundaries[:-1]).any())
            ):
                raise ValueError("Mistral packed CP padded boundaries must strictly span each complete pack")
            documents.append(torch.bucketize(torch.arange(ids.shape[1], device=ids.device), boundaries[1:], right=True))
        prepared = dict(local_batch)
        for name in ("seq_lens_padded", "attention_mask", "loss_mask"):
            prepared.pop(name, None)
        for name in ("input_ids", "labels", "position_ids", "padding_mask"):
            prepared[name] = converted[name].reshape_as(ids)
        for name in ("attention_mask", "padding_mask"):
            mask = local_batch.get(name)
            if mask is not None:
                if mask.shape != ids.shape or not bool(((mask == 0) | (mask == 1)).all()):
                    raise ValueError("Mistral packed CP masks must be binary and match global token axes")
                prepared["padding_mask"] = prepared["padding_mask"] | (
                    mask.bool() if name == "padding_mask" else ~mask.bool()
                )
        prepared["_mistral_packed_document_ids"] = torch.stack(documents)
        prepared["cu_seqlens"] = cu
        keep = ~prepared["padding_mask"]
        if cp_mesh is None or cp_mesh.size() == 1:
            if source_mask is not None:
                prepared["loss_mask"] = source_mask
            if not shard_primary:
                prepared["attention_mask"] = keep
            return (
                contextlib.nullcontext,
                prepared,
                ShardLayout(
                    original_seq_len=ids.shape[1],
                    padded_seq_len=ids.shape[1],
                    local_token_global_indices=torch.arange(ids.shape[1], device=ids.device),
                ),
            )
        context, sharded, layout = shard_batch_contiguous(
            cp_mesh,
            tp_mesh,
            prepared,
            loss_mask=source_mask,
            padding_token_id=padding_token_id,
            extra_seq_keys={"_mistral_packed_document_ids": 1},
            extra_pad_values={"_mistral_packed_document_ids": -1},
            shard_primary=shard_primary,
        )
        if not shard_primary:
            sharded["attention_mask"] = keep
        return context, sharded, layout

    return {
        "cp_sharder": ContextParallelSharder(shard_batch=shard, local_token_global_indices=contiguous_local_indices)
    }
