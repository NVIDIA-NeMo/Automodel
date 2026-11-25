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

"""Contiguous Inkling2 context shards, differentiable KV and convolution halos."""

from collections.abc import Callable
from typing import TypeAlias, cast

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
    shard_batch_identity,
)

CPBatch: TypeAlias = dict[str, torch.Tensor | list[torch.Tensor] | str | None]


def gather_sequence(value: torch.Tensor, mesh: DeviceMesh | None, *, differentiable: bool) -> torch.Tensor:
    """Gather equal-length [batch, local_sequence, ...] tensors in CP rank order.

    Args:
        value: This rank's shard [batch, local_sequence, ...]; every rank supplies the same
            shape and dtype.
        mesh: One-dimensional CP mesh; None or a single rank returns the input itself.
        differentiable: Route gradients back to each shard's source rank.

    Returns:
        [batch, global_sequence, ...] with the shards concatenated in rank order, newly
        allocated on the input's device. Differentiable transport sums every remote
        consumer's gradient at its source; FSDP owns parameter averaging. The result is
        waited on so compiled/raw-pointer consumers see completed data.
    """
    if mesh is None or mesh.size() == 1:
        return value
    sequence_first = value.movedim(1, 0).contiguous()
    if differentiable:
        gathered = funcol.all_gather_tensor_autograd(sequence_first, 0, mesh)
        gathered = funcol.wait_tensor(gathered)
    else:
        pieces = [torch.empty_like(sequence_first) for _ in range(mesh.size())]
        dist.all_gather(pieces, sequence_first, group=mesh.get_group())
        gathered = torch.cat(pieces)
    return gathered.movedim(0, 1)


def convolution_history(masked: torch.Tensor, mesh: DeviceMesh | None, width: int) -> torch.Tensor:
    """Prepend causal history to masked [batch, local_sequence, channels].

    Args:
        masked: This rank's shard [batch, local_sequence, channels] with invalid positions
            already zeroed.
        mesh: One-dimensional CP mesh; None or a single rank pads with zeros only.
        width: Convolution taps; width - 1 history positions are prepended.

    Returns:
        New [batch, local_sequence + width - 1, channels] tensor. Rank zero and positions
        before the sequence start receive zeros. Only each rank's final
        min(width - 1, local_sequence) tokens are exchanged, which also handles a halo
        wider than a shard and preserves backward participation on rank zero.
    """
    halo = width - 1
    if mesh is None or mesh.size() == 1 or halo == 0:
        return F.pad(masked, (0, 0, halo, 0))
    count = min(halo, masked.shape[1])
    tails = gather_sequence(masked[:, -count:], mesh, differentiable=True)
    preceding = tails[:, : mesh.get_local_rank() * count]
    prefix = preceding[:, -halo:]
    prefix = F.pad(prefix, (0, 0, halo - prefix.shape[1], 0))
    return torch.cat((prefix, masked), dim=1)


def unpack_row_layout(batch: CPBatch) -> None:
    """Normalize shared THD packs to document positions and validity, in place.

    The shared packer supplies real/padded lengths [batch, documents], padded
    with -1000. Padding is determined from those lengths, never from token IDs:
    a real EOS remains valid even when the tokenizer also uses it as PAD.

    Args:
        batch: Mutable mapping with input_ids [batch, sequence], seq_lens and
            seq_lens_padded [batch, documents], optional position_ids and labels
            [batch, sequence]. Replaces metadata with boolean padding_mask and
            integer position_ids [batch, sequence]; token IDs and labels are unchanged.
    """
    ids = cast(torch.Tensor, batch["input_ids"])
    lengths = cast(torch.Tensor | None, batch.pop("seq_lens", None))
    padded = cast(torch.Tensor | None, batch.pop("seq_lens_padded", None))
    if lengths is None or padded is None:
        raise ValueError("Inkling2 packed rows require seq_lens and seq_lens_padded")
    if (
        ids.ndim != 2
        or lengths.ndim != 2
        or lengths.shape != padded.shape
        or lengths.shape[0] != ids.shape[0]
        or lengths.shape[1] == 0
        or lengths.dtype not in (torch.int32, torch.int64)
        or padded.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Inkling2 packed lengths must be integer [batch, documents]")
    active = lengths != -1000
    if bool(((active & ((lengths <= 0) | (padded < lengths))) | (~active & (padded != -1000))).any()):
        raise ValueError("Inkling2 packed lengths must be positive and no larger than their padded lengths")
    if bool((active[:, 1:] & ~active[:, :-1]).any()):
        raise ValueError("Inkling2 packed length sentinels must occur only at the end")
    padded = padded.masked_fill(~active, 0)
    if bool((padded.sum(-1) != ids.shape[1]).any()):
        raise ValueError("Inkling2 padded document lengths must cover the entire row")
    ends = padded.cumsum(-1)
    columns = torch.arange(ids.shape[1], device=ids.device).expand_as(ids).contiguous()
    document = torch.searchsorted(ends.contiguous(), columns, right=True)
    starts = ends - padded
    positions = columns - starts.gather(1, document)
    keep = positions < lengths.gather(1, document)
    supplied = cast(torch.Tensor | None, batch.get("position_ids"))
    if supplied is not None and (supplied.shape != ids.shape or bool(((supplied != positions) & keep).any())):
        raise ValueError("Inkling2 packed position_ids must restart at each document")
    labels = cast(torch.Tensor | None, batch.get("labels"))
    if labels is not None and bool(((labels != -100) & ~keep).any()):
        raise ValueError("Inkling2 packed padding must have ignored labels")
    batch["position_ids"] = positions.masked_fill(~keep, 0)
    batch["padding_mask"] = ~keep
    batch.pop("qkv_format", None)


def validate_document_positions(keep: torch.Tensor, positions: torch.Tensor) -> None:
    """Require unit increments inside documents and zero at a new document.

    A contiguous CP shard may begin partway through a document. Later resets
    explicitly identify new documents; a gap cannot silently resume an old one.

    Args:
        keep: Boolean validity [batch, sequence].
        positions: Integer logical positions [batch, sequence]. The first real
            span may start at a nonzero offset, including after left padding.
    """
    if positions.shape != keep.shape or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Inkling2 position_ids must be integer [batch, sequence]")
    if bool(((positions < 0) & keep).any()):
        raise ValueError("Inkling2 logical positions must be nonnegative")
    next_valid = keep[:, 1:] & (keep.long().cumsum(-1)[:, :-1] > 0)
    continues = keep[:, :-1] & (positions[:, 1:] == positions[:, :-1] + 1)
    if bool((next_valid & ~continues & (positions[:, 1:] != 0)).any()):
        raise ValueError("Inkling2 positions must increase by one or restart at zero for a new document")


def prepare_cp_inputs(batch: CPBatch, *, num_chunks: int = 1) -> dict[str, ContextParallelSharder]:
    """Return a sharder for packed or unpacked global [batch, sequence] text/image inputs.

    IDs, image masks and images remain global for per-microbatch image splicing
    inside forward. Labels, positions and padding become local. The shared
    recipes own target alignment and loss normalization; no weights are touched.
    """
    if batch.get("cu_seqlens") is not None:
        raise ValueError("Inkling2 requires row-shaped packs with seq_lens metadata")
    if num_chunks < 1:
        raise ValueError("Inkling2 CP requires num_chunks >= 1")

    def shard(
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
        local_batch: CPBatch,
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
    ) -> tuple[Callable, CPBatch, ShardLayout]:
        """Validate the global real-token span before shared contiguous sharding.

        Args:
            cp_mesh: CP mesh; None or a single rank keeps every tensor global.
            tp_mesh: TP mesh forwarded to the shared sharders.
            local_batch: Global [batch, sequence] input_ids, labels, position_ids, binary
                attention_mask and boolean padding_mask; packed seq_lens and seq_lens_padded
                [batch, documents]; optional inputs_embeds [batch, sequence, hidden]; image
                patches, grids and the mm_token_type_ids image-body mask. Mutated in place:
                packed metadata becomes padding_mask and position_ids.
            loss_mask: Optional [batch, sequence] supervision mask, sharded like labels.
            padding_token_id: Filler for the sequence padding the shared sharder adds.

        Returns:
            The shared sharder's context, the sharded batch, and its layout. In the sharded
            batch labels, loss mask, position_ids and padding_mask are local
            [batch, local_sequence]; input_ids, the complete attention_mask, images and the
            image-body mask remain global for image splicing inside forward.
        """
        ids = cast(torch.Tensor, local_batch["input_ids"])
        if ids.ndim != 2 or min(ids.shape) == 0:
            raise ValueError("Inkling2 CP requires nonempty input_ids [batch, sequence]")
        packed = local_batch.get("qkv_format") == "thd" or local_batch.get("seq_lens") is not None
        if packed:
            unpack_row_layout(local_batch)
        keep = torch.ones_like(ids, dtype=torch.bool)
        mask = cast(torch.Tensor | None, local_batch.pop("attention_mask", None))
        if mask is not None:
            if mask.shape != ids.shape or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("Inkling2 CP attention_mask must be binary [batch, sequence]")
            keep &= mask.bool()
        padding = cast(torch.Tensor | None, local_batch.get("padding_mask"))
        if padding is not None:
            if padding.shape != ids.shape or padding.dtype != torch.bool:
                raise ValueError("Inkling2 CP padding_mask must be boolean [batch, sequence]")
            keep &= ~padding
        positions = cast(torch.Tensor | None, local_batch.get("position_ids"))
        if positions is None:
            positions = (keep.long().cumsum(1) - 1).clamp_min(0)
        validate_document_positions(keep, positions)
        local_batch["padding_mask"] = ~keep
        local_batch["position_ids"] = positions
        embeddings = local_batch.pop("inputs_embeds", None)
        if cp_mesh is None or cp_mesh.size() == 1:
            if embeddings is not None:
                local_batch["inputs_embeds"] = embeddings
            return shard_batch_identity(cp_mesh, tp_mesh, local_batch, loss_mask=loss_mask)
        context, sharded, layout = shard_batch_contiguous(
            cp_mesh,
            tp_mesh,
            local_batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
            shard_primary=False,
        )
        # Image validation requires the complete mask; the decoder slices it.
        if embeddings is not None:
            sharded["inputs_embeds"] = embeddings
        sharded["attention_mask"] = keep
        return context, sharded, layout

    return {
        "cp_sharder": ContextParallelSharder(
            shard_batch=shard,
            local_token_global_indices=contiguous_local_indices,
        )
    }
