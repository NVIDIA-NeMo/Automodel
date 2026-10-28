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

"""Qwen3.8-Flash-Next raw-token Engram N-gram Embedding lookup (``ple`` in the checkpoint config)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.qwen3_8_flash_next.cp import (
    Qwen3_8_FlashNextCPContext,
    qwen3_8_flash_next_cp_left_halo,
)
from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextGroupedRMSNorm
from nemo_automodel.shared.model_owned_sharding import stamp_model_owned_grad_divisor
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

QWEN3_8_FLASH_NEXT_LAYER_MULTIPLIERS = (
    23703573157769,
    20109073645365,
    8052911324071,
)
QWEN3_8_FLASH_NEXT_NGRAM_HEAD_VOCAB_SIZES = (
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
)
QWEN3_8_FLASH_NEXT_NGRAM_HEAD_OFFSETS = (
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
)
QWEN3_8_FLASH_NEXT_NGRAM_PADDED_ROWS = 320001536


_CAPACITY_MARGIN = 1.25  # capacity mode: padded rows per peer = ceil(calibrated max count x margin), aligned
_CAPACITY_ALIGN = 64
_UNTOUCHED = (
    -2
)  # receive-buffer fill: distinct from the -1 padding senders write, so an a2a that returns without writing is caught


def _segment_rows(
    split_sizes: tuple[int, ...], capacity: int, device: torch.device, counts: torch.Tensor | None = None
) -> torch.Tensor:
    """Row of every compact row inside the ``[world_size * capacity, ...]`` padded exchange buffer.

    Compact rows are ordered by peer rank; peer ``r``'s segment starts at row ``r * capacity`` of the padded buffer.
    Built on the device from the caller's count tensor (the AllGather row / column: no host round trip) so the
    packing is a single ``index_copy_`` and the unpacking a single ``index_select`` instead of one ``copy_`` launch
    per rank — 2 x world_size ``cudaMemcpyAsync`` per exchange, forward and backward (1280 per step at EP64 on the
    16-node profile). The id exchange builds the send-side and receive-side maps once per lookup; the value
    exchange reuses them forward (swapped) and backward (swapped back).
    """
    total = sum(split_sizes)
    if counts is None:
        # no device copy of the counts at hand (tests, CPU): one small host->device copy
        counts = torch.tensor(split_sizes, dtype=torch.long, device=device)
    else:
        counts = counts.to(device=device, dtype=torch.long)  # the caller's AllGather row / column: no host round trip
    starts = torch.cumsum(counts, 0) - counts
    rank_of_row = torch.repeat_interleave(torch.arange(len(split_sizes), device=device), counts, output_size=total)
    rows = torch.arange(total, device=device)
    return rank_of_row * capacity + (rows - starts[rank_of_row])


def _segment_rows_from_counts(counts: torch.Tensor, capacity: int, total: int) -> torch.Tensor:
    """Row of every compact row inside the ``[world_size * capacity, ...]`` padded buffer, from device-side counts only.

    ``total`` (= ``counts.sum()``) is the one host integer; it is the caller's own row count, so no device-to-host read
    is needed. Used by the capacity-mode lookup for its send side (pack and unpack).
    """
    counts = counts.to(dtype=torch.long)
    starts = torch.cumsum(counts, 0) - counts
    rank_of_row = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts, output_size=total)
    rows = torch.arange(total, device=counts.device)
    return rank_of_row * capacity + (rows - starts[rank_of_row])


def _aligned_capacity(max_count: int) -> int:
    """Padded rows per peer for a measured maximum count: margin for drift, aligned, never zero."""
    return max(_CAPACITY_ALIGN, -(-int(math.ceil(max_count * _CAPACITY_MARGIN)) // _CAPACITY_ALIGN) * _CAPACITY_ALIGN)


def _fixed_capacity_all_to_all(
    input_tensor: torch.Tensor,
    input_split_sizes: tuple[int, ...],
    output_split_sizes: tuple[int, ...],
    capacity: int,
    process_group: dist.ProcessGroup,
    *,
    fill_value: int | float,
    input_rows: torch.Tensor | None = None,
    output_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exchange compact rank segments through an equal-split All-to-All.

    Padding every peer segment to one globally agreed capacity removes
    backend-specific uneven-split behavior and gives forward and backward the
    same symmetric exchange metadata.  The compact, source-ordered result
    expected by the owner lookup is restored after the collective.  Transport
    provider selection remains an independent runtime concern.

    Args:
        input_tensor: Compact tensor of shape ``[sum(input_split_sizes), ...]``
            whose axis-0 segments are ordered by destination rank.
        input_split_sizes: Number of rows sent to every destination rank.
        output_split_sizes: Number of rows received from every source rank.
        capacity: Globally agreed maximum of every source/destination count.
        process_group: Process group whose rank order defines both count tuples.
        fill_value: Value used for padded and initially untouched output rows.
        input_rows: Optional precomputed ``_segment_rows(input_split_sizes, capacity, …)``
            so the packing issues no row-map kernels of its own; ``output_rows`` likewise
            for the unpacking. Built here from the host tuples when absent.

    Returns:
        A compact tensor of shape ``[sum(output_split_sizes), ...]`` whose
        axis-0 segments are ordered by source rank.
    """
    world_size = dist.get_world_size(process_group)
    if len(input_split_sizes) != world_size or len(output_split_sizes) != world_size:
        raise ValueError("Fixed-capacity All-to-All split metadata must contain one entry per process-group rank")
    if capacity < 0:
        raise ValueError(f"Fixed-capacity All-to-All capacity must be non-negative, got {capacity}")
    if any(count < 0 or count > capacity for count in (*input_split_sizes, *output_split_sizes)):
        raise ValueError(
            f"Fixed-capacity All-to-All counts must lie in [0, {capacity}]: "
            f"input={input_split_sizes}, output={output_split_sizes}"
        )
    if input_tensor.shape[0] != sum(input_split_sizes):
        raise ValueError(
            "Fixed-capacity All-to-All input rows do not match its split metadata: "
            f"{input_tensor.shape[0]} != {sum(input_split_sizes)}"
        )

    output_shape = (sum(output_split_sizes), *input_tensor.shape[1:])
    if capacity == 0:
        return input_tensor.new_empty(output_shape)

    padded_shape = (world_size, capacity, *input_tensor.shape[1:])
    padded_input = input_tensor.new_full(padded_shape, fill_value)
    if input_tensor.shape[0]:
        if input_rows is None:
            input_rows = _segment_rows(input_split_sizes, capacity, input_tensor.device)
        padded_input.view(world_size * capacity, *input_tensor.shape[1:]).index_copy_(0, input_rows, input_tensor)

    # A sentinel-initialized receive buffer makes a transport that returns
    # without writing deterministic.  The ID route validates these sentinels
    # before local indexing; padding itself is never exposed in compact output.
    padded_output = input_tensor.new_full(padded_shape, fill_value)
    contiguous_input = padded_input.contiguous()
    # The collective is stream-ordered after the packing copies above, and the
    # unpacking copies below are ordered after it; no device-wide synchronisation
    # is needed (each one drained every stream on every Engram exchange).
    dist.all_to_all_single(padded_output, contiguous_input, group=process_group)

    if output_shape[0] == 0:
        return input_tensor.new_empty(output_shape)
    if output_rows is None:
        output_rows = _segment_rows(output_split_sizes, capacity, input_tensor.device)
    return padded_output.view(world_size * capacity, *input_tensor.shape[1:]).index_select(0, output_rows)


class _FixedCapacityAllToAll(torch.autograd.Function):
    """Autograd-aware equal-split All-to-All for compact routed values."""

    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: torch.Tensor,
        input_split_sizes: tuple[int, ...],
        output_split_sizes: tuple[int, ...],
        capacity: int,
        process_group: dist.ProcessGroup,
        input_rows: torch.Tensor | None = None,
        output_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exchange compact rows through fixed-capacity peer segments.

        Args:
            ctx: PyTorch autograd context.
            input_tensor: Tensor of shape ``[input_rows, ...]``. Axis 0 contains
                contiguous per-destination segments described by
                ``input_split_sizes``; arbitrary trailing dimensions are kept.
            input_split_sizes: Number of rows sent to every destination rank.
            output_split_sizes: Number of rows received from every source rank.
            capacity: Global maximum peer-segment row count.
            process_group: Process group whose rank order defines both split tuples.

        Returns:
            Tensor of shape ``[output_rows, ...]``, where
            ``output_rows = sum(output_split_sizes)`` and axis 0 is grouped by
            source rank.
        """
        ctx.process_group = process_group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.capacity = capacity
        ctx.input_rows = input_rows
        ctx.output_rows = output_rows
        return _fixed_capacity_all_to_all(
            input_tensor,
            input_split_sizes,
            output_split_sizes,
            capacity,
            process_group,
            fill_value=0,
            input_rows=input_rows,
            output_rows=output_rows,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        """Route output gradients back to the ranks that supplied the rows.

        Args:
            ctx: PyTorch autograd context populated by :meth:`forward`.
            grad_output: Tensor of shape ``[output_rows, ...]`` with the same
                source-rank segmentation as the forward output.

        Returns:
            A gradient tensor of shape ``[input_rows, ...]`` followed by four
            ``None`` entries for the non-tensor split and process-group inputs.
        """
        grad_input = _fixed_capacity_all_to_all(
            grad_output,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            ctx.capacity,
            ctx.process_group,
            fill_value=0,
            input_rows=ctx.output_rows,
            output_rows=ctx.input_rows,
        )
        return grad_input, None, None, None, None, None, None


class _PaddedAllToAll(torch.autograd.Function):
    """Equal-split All-to-All of an already padded ``[world_size * capacity, ...]`` buffer, both directions.

    The capacity-mode lookup keeps the owner side in the padded layout, so the value exchange and its backward are the
    same symmetric collective on identically shaped buffers; nothing is read on the host.
    """

    @staticmethod
    def forward(ctx: Any, padded: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.process_group = process_group
        out = torch.empty_like(padded)
        dist.all_to_all_single(out, padded.contiguous(), group=process_group)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output.contiguous(), group=ctx.process_group)
        return grad_input, None


@dataclass(frozen=True)
class Qwen3_8_FlashNextEngramTableConfig:
    """Declarative shape and initialization settings for the PLE embedding table.

    Args:
        num_embeddings: Globally padded number of table rows. It must be divisible
            by the owner process-group size.
        embedding_dim: Number of values stored in each row.
        initializer_range: Standard deviation for checkpoint-free normal
            initialization.
        strict_validation: Validate every lookup on the host with cross-rank
            agreement (the default and the original behaviour): each lookup
            costs about twenty device-to-host synchronisations and seven small
            collectives, which stall the issuing thread on every Engram layer,
            forward and recompute. ``False`` keeps the same invariants as
            device-side asserts (``torch._assert_async``) on CUDA inputs and
            reads the route metadata (the count matrix) back once per lookup;
            a violated invariant then surfaces as a CUDA device-side assertion
            instead of a symmetric Python exception. DeepSeek-V4.1 opts out
            through ``text_config.engram_strict_validation`` (default False).
            CPU inputs are always validated on the host. The validation path is
            chosen per table (device and this flag), never per request count, so
            an empty requester takes the same collectives as its peers.
        capacity_mode: Opt-in for bounded workloads. The padded exchange is
            sized once, by the first lookup (its per-peer count maximum x 1.25,
            aligned), read back to the host that one time and fixed from then
            on, so no later lookup synchronises; a lookup whose per-peer count
            exceeds that calibration aborts with a device-side assertion. Keep
            it off (the default, the exact-count path) for ordinary inputs —
            growing sequence lengths, local-batch padding, distribution shifts —
            which it would reject; fixed-shape benchmark batches qualify.
    """

    num_embeddings: int
    embedding_dim: int
    initializer_range: float = 0.02
    strict_validation: bool = True
    capacity_mode: bool = False

    def __post_init__(self) -> None:
        if self.num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {self.num_embeddings}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.initializer_range < 0:
            raise ValueError(f"initializer_range must be non-negative, got {self.initializer_range}")

    def build(
        self,
        *,
        process_group: dist.ProcessGroup | None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        bind_owner_group_later: bool = False,
    ) -> Qwen3_8_FlashNextOwnerShardedEmbedding:
        """Build a local or row-owner-sharded embedding table.

        Args:
            process_group: Runtime owner group. Explicitly pass ``None`` only
                for a single-rank reference table containing all global rows.
                Omitting this argument is intentionally an error, preventing a
                full 102.4 GB table from being allocated accidentally.
            device: Device on which the rank-local weight of shape
                ``[local_rows, embedding_dim]`` is allocated.
            dtype: Data type of the rank-local weight tensor.
            bind_owner_group_later: Build the table without an owner group and
                without row storage. The parallelizer binds the owner group
                through :meth:`Qwen3_8_FlashNextOwnerShardedEmbedding.parallelize_weight`
                once the FSDP mesh of the pipeline stage that holds the table is
                known. ``process_group`` must be ``None`` in this mode.

        Returns:
            An embedding whose input has shape ``[...]`` and whose output has
            shape ``[..., embedding_dim]``. With a process group, the global row
            axis is sharded contiguously and evenly across its ranks.
        """
        return Qwen3_8_FlashNextOwnerShardedEmbedding(
            self,
            process_group=process_group,
            device=device,
            dtype=dtype,
            bind_owner_group_later=bind_owner_group_later,
        )


class Qwen3_8_FlashNextOwnerShardedEmbedding(nn.Module):
    """Trainable contiguous row-owner embedding with bidirectional All-to-All.

    Rank ``r`` owns rows ``[r * local_rows, (r + 1) * local_rows)``. Each
    request rank groups global row IDs by owner and sends them in the first
    All-to-All. Owners perform a local embedding lookup. An autograd-aware
    second All-to-All returns values to request ranks and reverses direction in
    backward, so only the owner accumulates and updates a row's gradient.

    Args:
        config: Global table shape and initialization settings.
        process_group: Runtime owner group. ``None`` stores the complete table
            locally and performs no collectives.
        device: Device for the local weight of shape
            ``[num_embeddings / owner_world_size, embedding_dim]``.
        dtype: Data type of the local weight tensor.
        bind_owner_group_later: Leave the owner group unbound. The weight is a
            ``[0, embedding_dim]`` placeholder until :meth:`parallelize_weight`
            binds the FSDP mesh of the pipeline stage that holds the table, so
            the row range is never sized for ranks that end up on another stage.
    """

    def __init__(
        self,
        config: Qwen3_8_FlashNextEngramTableConfig,
        *,
        process_group: dist.ProcessGroup | None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        bind_owner_group_later: bool = False,
    ) -> None:
        super().__init__()
        if bind_owner_group_later and process_group is not None:
            raise ValueError("bind_owner_group_later requires process_group=None; the mesh binds the group later")
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.initializer_range = config.initializer_range
        self.strict_validation = config.strict_validation
        # Capacity mode (opt-in, bounded workloads): the padded exchange is sized once from the first lookup and
        # fixed afterwards; every rank sees the same count matrix, so every rank derives the same capacity without
        # a collective on the host value, and a checkpointed forward and its recomputation agree by construction.
        self.capacity_mode = bool(config.capacity_mode)
        self._capacity: int | None = None
        self.owner_group_pending = bind_owner_group_later
        if bind_owner_group_later:
            self.process_group = None
            self.owner_world_size = 0
            self.owner_rank = 0
            self.num_embeddings_per_rank = 0
            self.weight = nn.Parameter(torch.empty(0, self.embedding_dim, device=device, dtype=dtype))
        else:
            self._set_owner_geometry(process_group)
            self.weight = nn.Parameter(
                torch.empty(
                    self.num_embeddings_per_rank,
                    self.embedding_dim,
                    device=device,
                    dtype=dtype,
                )
            )
        self.mark_sharding_contract()
        self.reset_parameters()

    def _set_owner_geometry(self, process_group: dist.ProcessGroup | None) -> None:
        """Record the owner group and this rank's contiguous global row range.

        Args:
            process_group: Owner group, or ``None`` for one rank holding every row.
        """
        self.process_group = process_group
        if process_group is None:
            self.owner_world_size = 1
            self.owner_rank = 0
        else:
            if not dist.is_initialized():
                raise RuntimeError("A distributed process group requires torch.distributed to be initialized")
            self.owner_world_size = dist.get_world_size(process_group)
            self.owner_rank = dist.get_rank(process_group)
        if self.num_embeddings % self.owner_world_size != 0:
            raise ValueError(
                "The padded Engram row count must be divisible by the owner group size: "
                f"{self.num_embeddings} % {self.owner_world_size} != 0"
            )
        self.num_embeddings_per_rank = self.num_embeddings // self.owner_world_size
        self.vocab_start_index = self.owner_rank * self.num_embeddings_per_rank
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_rank
        self.global_row_start = self.vocab_start_index
        self.global_row_end = self.vocab_end_index
        self.local_row_start = 0
        self.local_row_end = self.num_embeddings_per_rank

    def bind_owner_group(self, process_group: dist.ProcessGroup) -> None:
        """Bind a pending table to its owner group and allocate the local row shard.

        The placeholder weight keeps its device and dtype; a meta placeholder
        yields a meta shard that the usual materialization fills later, a real
        placeholder is replaced by a freshly initialized shard.

        Args:
            process_group: Owner group whose ranks hold contiguous row ranges.
        """
        if not self.owner_group_pending:
            raise RuntimeError("The Engram owner group is already bound")
        self._set_owner_geometry(process_group)
        placeholder = self.weight
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_rank,
                self.embedding_dim,
                device=placeholder.device,
                dtype=placeholder.dtype,
            ),
            requires_grad=placeholder.requires_grad,
        )
        self.owner_group_pending = False
        if placeholder.device.type != "meta":
            self.reset_parameters()

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Qwen3_8_FlashNextOwnerShardedEmbedding:
        """Restore ownership metadata after a module device or dtype conversion.

        Args:
            fn: Tensor conversion applied by ``Module.to`` or ``Module.to_empty``.
            recurse: Whether to convert child modules as well.

        Returns:
            This module after the normal PyTorch conversion, with the current
            row-sharded weight marked for owner-gradient averaging. Shapes,
            placements and values follow ``fn`` without an additional copy.
        """
        result = super()._apply(fn, recurse=recurse)
        self.mark_sharding_contract()
        return result

    def mark_sharding_contract(self) -> None:
        """Stamp the model-owned contract on the current weight.

        DTensor conversion can discard custom attributes even when PyTorch
        preserves the Parameter identity. Ordinary module conversions restore
        the marker through ``_apply``; the checkpoint materialization helper
        carries it over itself, and model initialization stamps it as well.
        The single-rank reference table needs no contract.
        """
        if isinstance(self.weight, DTensor):
            stamp_model_owned_grad_divisor(self.weight, self.owner_world_size)

    def parallelize_weight(self, fsdp_mesh: DeviceMesh) -> nn.Parameter:
        """Represent the already-local owner shard as one global DTensor.

        The local storage is already the final contiguous row shard, so this
        method uses :meth:`DTensor.from_local` rather than redistributing or
        slicing it again.  It runs before FSDP records its ignored parameters;
        the returned parameter identity must be passed unchanged to every FSDP
        unit containing the table.

        Args:
            fsdp_mesh: One-dimensional FSDP shard/CP mesh of the pipeline stage
                that holds the table. A pending table binds this mesh's group
                as its owner group; an already bound table requires the same
                rank order.

        Returns:
            The registered global ``[num_embeddings, embedding_dim]`` DTensor
            parameter with placement ``Shard(0)``.
        """
        if fsdp_mesh.ndim != 1:
            raise ValueError(f"The Engram DTensor requires a one-dimensional owner mesh, got ndim={fsdp_mesh.ndim}")
        if self.owner_group_pending:
            self.bind_owner_group(fsdp_mesh.get_group())
        if self.process_group is None:
            raise RuntimeError("A single-rank reference Engram table must remain an ordinary Parameter")
        if fsdp_mesh.size() != self.owner_world_size:
            raise ValueError(
                "The Engram owner group and FSDP mesh must have the same size: "
                f"{self.owner_world_size} != {fsdp_mesh.size()}"
            )
        owner_ranks = tuple(dist.get_process_group_ranks(self.process_group))
        mesh_ranks = tuple(dist.get_process_group_ranks(fsdp_mesh.get_group()))
        if owner_ranks != mesh_ranks:
            raise ValueError(
                "The Engram owner group rank order must exactly match the FSDP mesh: "
                f"owner={owner_ranks}, fsdp={mesh_ranks}"
            )

        expected_global_shape = (self.num_embeddings, self.embedding_dim)
        expected_local_shape = (self.num_embeddings_per_rank, self.embedding_dim)
        if isinstance(self.weight, DTensor):
            if tuple(self.weight.shape) != expected_global_shape:
                raise ValueError(
                    f"Engram DTensor global shape {tuple(self.weight.shape)} does not match {expected_global_shape}"
                )
            if tuple(self.weight.placements) != (Shard(0),):
                raise ValueError(f"Engram DTensor must use placement Shard(0), got {self.weight.placements}")
            if tuple(self.weight.to_local().shape) != expected_local_shape:
                raise ValueError(
                    "Engram DTensor local shape does not match its contiguous owner range: "
                    f"{tuple(self.weight.to_local().shape)} != {expected_local_shape}"
                )
            current_mesh_ranks = tuple(dist.get_process_group_ranks(self.weight.device_mesh.get_group()))
            if current_mesh_ranks != mesh_ranks:
                raise ValueError(
                    "Engram DTensor is already attached to a different mesh: "
                    f"current={current_mesh_ranks}, requested={mesh_ranks}"
                )
            self.mark_sharding_contract()
            return self.weight

        if tuple(self.weight.shape) != expected_local_shape:
            raise ValueError(
                "Engram local weight shape does not match its contiguous owner range: "
                f"{tuple(self.weight.shape)} != {expected_local_shape}"
            )
        local_weight = self.weight.detach()
        requires_grad = self.weight.requires_grad
        distributed_weight = DTensor.from_local(
            local_weight,
            device_mesh=fsdp_mesh,
            placements=(Shard(0),),
            run_check=False,
            shape=torch.Size(expected_global_shape),
            stride=(self.embedding_dim, 1),
        )
        self.weight = nn.Parameter(distributed_weight, requires_grad=requires_grad)
        self.mark_sharding_contract()
        return self.weight

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize the rank-local weight with a finite normal distribution."""
        local_weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        nn.init.normal_(local_weight, mean=0.0, std=self.initializer_range)

    def _use_capacity_mode(self) -> bool:
        """Whether lookups run without a device-to-host read (capacity mode): a per-table opt-in, never automatic."""
        return self.process_group is not None and not self.strict_validation and self.capacity_mode

    def _capacity_for(self, count_matrix: torch.Tensor) -> int:
        """Padded rows per peer: calibrated by the first lookup, fixed from then on.

        The first lookup reads the matrix maximum synchronously (one calibration per table, like HybridEP's
        capacity mode). Every later lookup only asserts on the device that its own maximum fits. Nothing about the
        padded layout depends on what ran in between, so a checkpointed forward and its recomputation size the
        exchange identically; a per-peer count beyond the calibration aborts the step instead of corrupting the
        exchange — that is the bounded-workload contract of ``capacity_mode``.
        """
        max_dev = count_matrix.max()
        if self._capacity is None:
            self._capacity = _aligned_capacity(int(max_dev.item()))
        torch._assert_async(
            max_dev <= self._capacity,
            "Engram capacity mode: a lookup exceeded the calibrated padded capacity (capacity_mode is an opt-in for "
            "bounded workloads; the exact-count path follows any request size)",
        )
        return self._capacity

    def _validate_padded_received_ids(self, received_ids: torch.Tensor, capacity: int) -> None:
        """Validate owner membership of a padded ``[world_size * capacity]`` receive buffer.

        Rows equal to -1 are the senders' padding and legitimate; ``_UNTOUCHED`` rows mean the exchange never wrote
        the slot and count as bad. On CUDA this is one device-side assert; on the host path the non-padding rows are
        handed to the compact diagnostics in source-rank order.
        """
        if self.process_group is None:
            return
        padding = received_ids == -1
        bad_mask = (~padding) & ((received_ids < self.vocab_start_index) | (received_ids >= self.vocab_end_index))
        if not self._validate_on_host(received_ids):
            torch._assert_async(
                bad_mask.sum(dtype=torch.int64) == 0, "Engram received a row ID outside this owner's range"
            )
            return
        keep = (~padding).view(self.owner_world_size, capacity)
        sizes = tuple(int(count) for count in keep.sum(dim=1).tolist())
        compact = received_ids.view(self.owner_world_size, capacity)[keep]
        self._validate_received_ids(compact, sizes)

    def _lookup_capacity_mode(
        self,
        sorted_global_ids: torch.Tensor,
        send_counts: torch.Tensor,
        unsort_indices: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Owner-sharded lookup with no device-to-host read: padded exchange sized by the calibrated capacity.

        Senders pack their owner-sorted IDs with a device-side row map (only their own counts and row total), owners
        look rows up in the padded layout (padding masked to zero), and the value return is the symmetric equal-split
        All-to-All, unpacked with the same send-row map. The 64 x 64 count AllGather stays; its result is consumed on
        the device.
        """
        world = self.owner_world_size
        gathered_counts = send_counts.new_empty(world * world)
        dist.all_gather_into_tensor(gathered_counts, send_counts, group=self.process_group)
        count_matrix = gathered_counts.view(world, world)
        capacity = self._capacity_for(count_matrix)
        rows = sorted_global_ids.numel()
        send_rows = _segment_rows_from_counts(send_counts, capacity, rows)
        padded_ids = sorted_global_ids.new_full((world * capacity,), -1)
        if rows:
            padded_ids.index_copy_(0, send_rows, sorted_global_ids)
        received_ids = torch.full_like(padded_ids, _UNTOUCHED)
        dist.all_to_all_single(received_ids, padded_ids, group=self.process_group)
        self._validate_padded_received_ids(received_ids, capacity)
        valid = (received_ids >= 0).unsqueeze(-1)
        local_ids = (received_ids - self.vocab_start_index).clamp_(0, self.num_embeddings_per_rank - 1)
        local_weight = self.weight.to_local(grad_placements=self.weight.placements)
        owned_values = F.embedding(local_ids, local_weight) * valid.to(local_weight.dtype)
        returned_padded = _PaddedAllToAll.apply(owned_values, self.process_group)
        returned_values = returned_padded.index_select(0, send_rows)
        return returned_values[unsort_indices].reshape(*original_shape, self.embedding_dim)

    def _validate_on_host(self, tensor: torch.Tensor) -> bool:
        """Return whether a lookup is validated with host-side, cross-rank-symmetric checks.

        Args:
            tensor: A tensor of the lookup being validated; its device decides.

        Returns:
            True under ``strict_validation`` or for CPU tensors (no synchronisation
            cost), False when the same invariants are enforced as device-side asserts.
        """
        return self.strict_validation or not tensor.is_cuda

    def _validate_global_ids(self, global_ids: torch.Tensor) -> None:
        """Validate IDs symmetrically before any variable-sized collective.

        Args:
            global_ids: Integer tensor of shape ``[...]`` containing global,
                packed-table row IDs.
        """
        is_integer = global_ids.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if not self._validate_on_host(global_ids):
            # Device path: no device-to-host read and no validity collective. Chosen by device and
            # strict_validation alone, so an empty requester never joins a collective its peers skip; the
            # dtype is a program invariant and fails the same way on every rank.
            if not is_integer:
                raise ValueError(f"Engram row IDs must use an integer dtype, got {global_ids.dtype}")
            if global_ids.numel() > 0:
                torch._assert_async(
                    (global_ids.min() >= 0) & (global_ids.max() < self.num_embeddings),
                    f"Engram row IDs must lie in [0, {self.num_embeddings})",
                )
            return
        in_range = is_integer
        if is_integer and global_ids.numel() > 0:
            in_range = bool((global_ids.min() >= 0).item() and (global_ids.max() < self.num_embeddings).item())
        valid = is_integer and in_range
        if self.process_group is not None:
            valid_tensor = torch.tensor(int(valid), device=global_ids.device, dtype=torch.int32)
            dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN, group=self.process_group)
            valid = bool(valid_tensor.item())
        if not valid:
            raise ValueError(
                f"Engram row IDs must use an integer dtype and lie in [0, {self.num_embeddings}) on every owner rank"
            )

    def _exchange_ids(
        self,
        sorted_global_ids: torch.Tensor,
        send_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int]:
        """Send global row IDs to their contiguous row owners.

        Args:
            sorted_global_ids: Tensor of shape ``[request_rows]`` grouped by
                destination owner rank.
            send_counts: Tensor of shape ``[owner_world_size]`` containing the
                number of IDs sent to each owner.

        Returns:
            A tuple containing owner-local received IDs of shape
            ``[owned_requests]``, the per-destination send counts, the
            per-source receive counts, and the globally agreed padded route
            capacity.
        """
        if self.process_group is None:
            counts = (sorted_global_ids.numel(),)
            return sorted_global_ids, counts, counts, sorted_global_ids.numel(), None, None

        # AllGather exposes one send-count row from every request rank.  Each
        # owner reads its column to obtain source-ordered receive splits.  At
        # EP64 this fixed-shape exchange is only 64 * 64 int64 values
        # (32 KiB/rank).
        gathered_counts = send_counts.new_empty(self.owner_world_size * self.owner_world_size)
        dist.all_gather_into_tensor(gathered_counts, send_counts, group=self.process_group)
        count_matrix = gathered_counts.view(self.owner_world_size, self.owner_world_size)

        if not self.strict_validation:
            # One device-to-host read serves all three pieces of host metadata:
            # this rank's row is its send counts, its column the receive counts,
            # and the matrix maximum the agreed capacity.
            host_matrix = count_matrix.cpu()
            input_split_sizes = tuple(int(count) for count in host_matrix[self.owner_rank].tolist())
            output_split_sizes = tuple(int(count) for count in host_matrix[:, self.owner_rank].tolist())
            capacity = int(host_matrix.max())
            device = sorted_global_ids.device
            send_rows = _segment_rows(input_split_sizes, capacity, device, count_matrix[self.owner_rank])
            receive_rows = _segment_rows(output_split_sizes, capacity, device, count_matrix[:, self.owner_rank])
            received_ids = _fixed_capacity_all_to_all(
                sorted_global_ids,
                input_split_sizes,
                output_split_sizes,
                capacity,
                self.process_group,
                fill_value=-1,
                input_rows=send_rows,
                output_rows=receive_rows,
            )
            return received_ids, input_split_sizes, output_split_sizes, capacity, send_rows, receive_rows

        # Strict mode: split sizes are Python host metadata used to pack compact
        # segments; materialise this rank's row before the consistency checks.
        input_split_sizes = tuple(int(count) for count in send_counts.cpu().tolist())

        # Validate both the rank-local contribution and cross-rank agreement
        # before count metadata can size a payload buffer.  MIN/MAX reductions
        # are cheap for this 32 KiB EP64 matrix and make a plausible but
        # inconsistent AllGather result fail symmetrically instead of causing
        # a payload overread on just one peer.
        minimum_count_matrix = count_matrix.clone()
        maximum_count_matrix = count_matrix.clone()
        dist.all_reduce(minimum_count_matrix, op=dist.ReduceOp.MIN, group=self.process_group)
        dist.all_reduce(maximum_count_matrix, op=dist.ReduceOp.MAX, group=self.process_group)
        local_row_matches = torch.equal(count_matrix[self.owner_rank], send_counts)
        local_row_sum_matches = int(count_matrix[self.owner_rank].sum().item()) == sorted_global_ids.numel()
        matrices_match = torch.equal(minimum_count_matrix, maximum_count_matrix)
        count_metadata_valid = local_row_matches and local_row_sum_matches and matrices_match
        valid_tensor = torch.tensor(int(count_metadata_valid), device=send_counts.device, dtype=torch.int32)
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(valid_tensor.item()):
            raise RuntimeError(
                "Engram count AllGather produced inconsistent route metadata; "
                "refusing to size the fixed-capacity payload exchange"
            )

        receive_counts = count_matrix[:, self.owner_rank].contiguous()
        output_split_sizes = tuple(int(count) for count in receive_counts.cpu().tolist())
        capacity = int(maximum_count_matrix.max().item())
        device = sorted_global_ids.device
        send_rows = _segment_rows(input_split_sizes, capacity, device, send_counts)
        receive_rows = _segment_rows(output_split_sizes, capacity, device, receive_counts)
        received_ids = _fixed_capacity_all_to_all(
            sorted_global_ids,
            input_split_sizes,
            output_split_sizes,
            capacity,
            self.process_group,
            fill_value=-1,
            input_rows=send_rows,
            output_rows=receive_rows,
        )
        return received_ids, input_split_sizes, output_split_sizes, capacity, send_rows, receive_rows

    def _validate_sorted_send_ids(
        self,
        sorted_global_ids: torch.Tensor,
        send_counts: torch.Tensor,
    ) -> None:
        """Symmetrically verify compact destination segments before routing.

        Args:
            sorted_global_ids: Tensor of shape ``[request_rows]`` containing
                global IDs grouped by contiguous owner rank.
            send_counts: Tensor of shape ``[owner_world_size]`` containing one
                request count per destination owner.

        Raises:
            RuntimeError: If any request rank's segment contains an ID owned by
                a different destination rank.
        """
        if self.process_group is None:
            return
        actual_owners = torch.div(sorted_global_ids, self.num_embeddings_per_rank, rounding_mode="floor")
        if not self._validate_on_host(sorted_global_ids):
            # ``output_size`` keeps repeat_interleave from reading the counts back to
            # the host; the segment lengths sum to the request count by construction.
            expected_owners = torch.repeat_interleave(
                torch.arange(self.owner_world_size, device=sorted_global_ids.device, dtype=torch.long),
                send_counts,
                output_size=sorted_global_ids.numel(),
            )
            torch._assert_async(
                (expected_owners == actual_owners).all(),
                "Engram sorted ID segments do not match their destination owners",
            )
            return
        expected_owners = torch.repeat_interleave(
            torch.arange(self.owner_world_size, device=sorted_global_ids.device, dtype=torch.long),
            send_counts,
        )
        locally_valid = expected_owners.shape == actual_owners.shape and torch.equal(expected_owners, actual_owners)
        valid_tensor = torch.tensor(int(locally_valid), device=sorted_global_ids.device, dtype=torch.int32)
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(valid_tensor.item()):
            raise RuntimeError(
                "Engram sorted ID segments do not match their destination owners; refusing the payload All-to-All"
            )

    def _validate_received_ids(
        self,
        received_ids: torch.Tensor,
        output_split_sizes: tuple[int, ...],
    ) -> None:
        """Collectively validate that routed IDs belong to this row owner.

        Every rank first contributes its local bad-ID count to an AllReduce.
        Consequently, all ranks take the same success or failure branch even
        when only one owner received a misrouted ID.  On failure, a compact
        fixed-size diagnostic from every owner is gathered before raising, so
        the exception identifies the failing owner rank(s) without leaving
        peers to enter the value-return All-to-All.

        Args:
            received_ids: Global row IDs received from request ranks, with
                shape ``[owned_requests]``.
            output_split_sizes: Number of received IDs from each source owner
                group rank, in source-rank order.

        Raises:
            RuntimeError: If any owner received an ID outside its contiguous
                global row range.
        """
        if self.process_group is None:
            return

        bad_mask = (received_ids < self.vocab_start_index) | (received_ids >= self.vocab_end_index)
        local_bad_count = bad_mask.sum(dtype=torch.int64)
        if not self._validate_on_host(received_ids):
            torch._assert_async(local_bad_count == 0, "Engram received a row ID outside this owner's range")
            return
        global_bad_count = local_bad_count.clone()
        dist.all_reduce(global_bad_count, op=dist.ReduceOp.SUM, group=self.process_group)
        if int(global_bad_count.item()) == 0:
            return

        # Diagnostics are only built on the failure path.  Nonnegative row IDs
        # make -1 an unambiguous sentinel for an empty receive or bad-ID set.
        sample_size = 8
        base_diagnostic_size = 8 + sample_size
        diagnostics = received_ids.new_full(
            (base_diagnostic_size + 3 * self.owner_world_size,),
            -1,
            dtype=torch.int64,
        )
        diagnostics[0] = self.owner_rank
        diagnostics[1] = dist.get_rank()
        diagnostics[2] = received_ids.numel()
        diagnostics[5] = local_bad_count
        if received_ids.numel() > 0:
            diagnostics[3] = received_ids.min()
            diagnostics[4] = received_ids.max()
        if int(local_bad_count.item()) > 0:
            bad_ids = received_ids[bad_mask]
            diagnostics[6] = bad_ids.min()
            diagnostics[7] = bad_ids.max()
            diagnostics[8 : 8 + min(sample_size, bad_ids.numel())] = bad_ids[:sample_size]

        source_sizes = diagnostics.new_tensor(output_split_sizes)
        source_bad_counts = diagnostics.new_zeros(self.owner_world_size)
        source_untouched_counts = diagnostics.new_zeros(self.owner_world_size)
        offset = 0
        for source_rank, split_size in enumerate(output_split_sizes):
            segment = received_ids[offset : offset + split_size]
            segment_bad_mask = bad_mask[offset : offset + split_size]
            source_bad_counts[source_rank] = segment_bad_mask.sum(dtype=torch.int64)
            source_untouched_counts[source_rank] = ((segment == -1) | (segment == _UNTOUCHED)).sum(dtype=torch.int64)
            offset += split_size
        diagnostics[base_diagnostic_size : base_diagnostic_size + self.owner_world_size] = source_sizes
        diagnostics[base_diagnostic_size + self.owner_world_size : base_diagnostic_size + 2 * self.owner_world_size] = (
            source_bad_counts
        )
        diagnostics[base_diagnostic_size + 2 * self.owner_world_size :] = source_untouched_counts

        gathered_diagnostics = diagnostics.new_empty(self.owner_world_size * diagnostics.numel())
        dist.all_gather_into_tensor(gathered_diagnostics, diagnostics, group=self.process_group)
        diagnostic_rows = gathered_diagnostics.view(self.owner_world_size, diagnostics.numel()).cpu().tolist()

        failures = []
        for row in diagnostic_rows:
            owner_rank, global_rank, received_count, received_min, received_max, bad_count, bad_min, bad_max = row[:8]
            if bad_count == 0:
                continue
            expected_start = owner_rank * self.num_embeddings_per_rank
            expected_end = expected_start + self.num_embeddings_per_rank
            bad_sample = row[8 : 8 + min(sample_size, bad_count)]
            source_sizes = row[base_diagnostic_size : base_diagnostic_size + self.owner_world_size]
            source_bad_counts = row[
                base_diagnostic_size + self.owner_world_size : base_diagnostic_size + 2 * self.owner_world_size
            ]
            source_untouched_counts = row[base_diagnostic_size + 2 * self.owner_world_size :]
            source_segments = [
                f"source_group_rank={source_rank}(size={source_size},bad={source_bad_count},"
                f"untouched={source_untouched_count})"
                for source_rank, (source_size, source_bad_count, source_untouched_count) in enumerate(
                    zip(source_sizes, source_bad_counts, source_untouched_counts)
                )
                if source_bad_count > 0
            ]
            failures.append(
                f"owner_group_rank={owner_rank} (global_rank={global_rank}) expected "
                f"[{expected_start}, {expected_end}), received_count={received_count}, "
                f"received_min={received_min}, received_max={received_max}, bad_count={bad_count}, "
                f"bad_min={bad_min}, bad_max={bad_max}, bad_sample={bad_sample}, "
                f"source_segments=[{', '.join(source_segments)}]"
            )

        raise RuntimeError(
            "Engram ID All-to-All routed global row IDs to the wrong owner; "
            "refusing the local embedding lookup. " + "; ".join(failures)
        )

    def forward(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Look up arbitrary global rows while keeping weights on row owners.

        Args:
            global_ids: Integer tensor of shape ``[...]`` containing global row
                IDs in the packed multi-head table.

        Returns:
            Tensor of shape ``[..., embedding_dim]`` in the original request
            order. The returned tensor does not alias ``global_ids`` or the
            rank-local table weight.
        """
        if self.owner_group_pending:
            raise RuntimeError(
                "The Engram table was used before its owner group was bound; "
                "apply the model's distributed parallelization first"
            )
        self._validate_global_ids(global_ids)
        original_shape = global_ids.shape
        flattened_ids = global_ids.reshape(-1).to(dtype=torch.long)
        if self.process_group is None:
            if isinstance(self.weight, DTensor):
                raise RuntimeError("A single-rank reference Engram table must not carry a distributed weight")
            return F.embedding(flattened_ids, self.weight).reshape(*original_shape, self.embedding_dim)
        if not isinstance(self.weight, DTensor):
            raise RuntimeError(
                "The distributed Engram table was used before its owner shard became a global DTensor; "
                "apply the model's distributed parallelization first"
            )

        owners = torch.div(flattened_ids, self.num_embeddings_per_rank, rounding_mode="floor")
        # scatter_add_ instead of bincount: bincount sizes its output from the data
        # and therefore reads the maximum back to the host on every call.
        send_counts = torch.zeros(self.owner_world_size, dtype=torch.int64, device=owners.device).scatter_add_(
            0, owners, torch.ones_like(owners, dtype=torch.int64)
        )
        sort_indices = torch.argsort(owners, stable=True)
        sorted_global_ids = flattened_ids[sort_indices]
        unsort_indices = torch.empty_like(sort_indices)
        unsort_indices[sort_indices] = torch.arange(sort_indices.numel(), device=sort_indices.device)
        self._validate_sorted_send_ids(sorted_global_ids, send_counts)
        if self._use_capacity_mode():
            return self._lookup_capacity_mode(sorted_global_ids, send_counts, unsort_indices, original_shape)

        received_ids, input_split_sizes, output_split_sizes, capacity, send_rows, receive_rows = self._exchange_ids(
            sorted_global_ids,
            send_counts,
        )
        self._validate_received_ids(received_ids, output_split_sizes)
        local_ids = received_ids - self.vocab_start_index
        local_weight = self.weight.to_local(grad_placements=self.weight.placements)
        owned_values = F.embedding(local_ids, local_weight)
        returned_values = _FixedCapacityAllToAll.apply(
            owned_values,
            output_split_sizes,
            input_split_sizes,
            capacity,
            self.process_group,
            receive_rows,  # owner -> requester: the id exchange's splits and row maps, swapped
            send_rows,
        )
        return returned_values[unsort_indices].reshape(*original_shape, self.embedding_dim)


class Qwen3_8_FlashNextNGramEmbedding(nn.Module):
    """Hash raw token IDs into the packed Qwen3.8-Flash-Next PLE table.

    The first ``heads_per_ngram`` heads hash bigrams, the next group hashes
    trigrams, and so on. Previous-token context resets *after* an EOS token.
    Hashing intentionally uses signed int64 overflow, positive remainders, and
    the checkpoint-provided global head offsets. It does not canonicalize or
    compress tokenizer IDs as the original DeepSeek Engram implementation does.

    Args:
        ngram_embedding: Lookup module accepting global row IDs of shape
            ``[batch, sequence, ngram_heads]`` and returning values of shape
            ``[batch, sequence, ngram_heads, head_dim]``.
        ngram_size: Largest n-gram order, including the current token.
        heads_per_ngram: Number of hash heads for every order from two through
            ``ngram_size``.
        eos_token_id: Raw tokenizer ID that terminates the preceding segment.
        layer_multipliers: Signed int64 multipliers of shape ``[ngram_size]``.
        ngram_heads_vocab_sizes: Prime modulus for each hash head, shape
            ``[(ngram_size - 1) * heads_per_ngram]``.
        ngram_heads_offsets: Global packed-table row offset for each hash head,
            with the same shape as ``ngram_heads_vocab_sizes``.
    """

    def __init__(
        self,
        ngram_embedding: nn.Module,
        *,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        eos_token_id: int = 248044,
        layer_multipliers: tuple[int, ...] = QWEN3_8_FLASH_NEXT_LAYER_MULTIPLIERS,
        ngram_heads_vocab_sizes: tuple[int, ...] = QWEN3_8_FLASH_NEXT_NGRAM_HEAD_VOCAB_SIZES,
        ngram_heads_offsets: tuple[int, ...] = QWEN3_8_FLASH_NEXT_NGRAM_HEAD_OFFSETS,
    ) -> None:
        super().__init__()
        if ngram_size < 2:
            raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
        if heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be positive, got {heads_per_ngram}")
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        if len(layer_multipliers) != ngram_size:
            raise ValueError(f"Expected {ngram_size} layer multipliers, got {len(layer_multipliers)}")
        if len(ngram_heads_vocab_sizes) != ngram_heads:
            raise ValueError(f"Expected {ngram_heads} head vocab sizes, got {len(ngram_heads_vocab_sizes)}")
        if len(ngram_heads_offsets) != ngram_heads:
            raise ValueError(f"Expected {ngram_heads} head offsets, got {len(ngram_heads_offsets)}")
        if any(size <= 0 for size in ngram_heads_vocab_sizes):
            raise ValueError("All n-gram head vocab sizes must be positive")
        if any(offset < 0 for offset in ngram_heads_offsets):
            raise ValueError("All n-gram head offsets must be non-negative")

        self.ngram_embedding = ngram_embedding
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_heads = ngram_heads
        self.eos_token_id = eos_token_id
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(layer_multipliers, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(ngram_heads_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(ngram_heads_offsets, dtype=torch.long),
            persistent=True,
        )

    def _shift_right_after_eos(
        self,
        input_ids: torch.Tensor,
        shift: int,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read an earlier token without crossing an EOS or packed-segment boundary.

        Args:
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]``.
            shift: Number of preceding positions to read.
            sequence_ids: Optional packed segment IDs with the shape of ``input_ids``.

        Returns:
            Raw IDs of shape ``[batch, sequence]``. Positions lacking valid
            same-segment context contain ``eos_token_id``.
        """
        if shift == 0:
            return input_ids
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device, dtype=torch.long)
        eos_positions = torch.where(input_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
            dim=1,
        )
        positions_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted_ids = input_ids.gather(dim=1, index=gather_positions)
        valid = (positions_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        if sequence_ids is not None:
            valid = valid & (sequence_ids.gather(dim=1, index=gather_positions) == sequence_ids)
        return torch.where(valid, shifted_ids, input_ids.new_full((), self.eos_token_id))

    def _hash_input_ids(self, input_ids: torch.Tensor, sequence_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Compute packed global table rows for every n-gram head.

        Args:
            input_ids: Raw integer tokenizer IDs of shape ``[batch, sequence]``.
            sequence_ids: Optional packed segment IDs of shape ``[batch, sequence]``;
                n-gram history stays inside a segment.

        Returns:
            Global table IDs of shape ``[batch, sequence, ngram_heads]``. Heads
            are ordered by increasing n-gram order and then head index.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        if input_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError(f"input_ids must have an integer dtype, got {input_ids.dtype}")
        input_ids = input_ids.to(dtype=torch.long)
        shifted_tokens = [
            self._shift_right_after_eos(input_ids, shift, sequence_ids) for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram_order in range(2, self.ngram_size + 1):
            head_start = (ngram_order - 2) * self.heads_per_ngram
            head_end = head_start + self.heads_per_ngram
            mixed = shifted_tokens[0] * self.layer_multipliers[0]
            for token_position in range(1, ngram_order):
                mixed = torch.bitwise_xor(
                    mixed,
                    shifted_tokens[token_position] * self.layer_multipliers[token_position],
                )
            head_sizes = self.ngram_heads_vocab_sizes[head_start:head_end]
            head_offsets = self.ngram_heads_offsets[head_start:head_end]
            block_ids = torch.remainder(mixed.unsqueeze(-1), head_sizes) + head_offsets
            blocks.append(block_ids)
        return torch.cat(blocks, dim=-1)

    def forward(self, input_ids: torch.Tensor, sequence_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return concatenated PLE table values for raw token IDs.

        Args:
            input_ids: Raw integer tokenizer IDs of shape ``[batch, sequence]``.
            sequence_ids: Optional packed segment IDs ``[batch, sequence]``.

        Returns:
            Tensor of shape ``[batch, sequence, ngram_heads * head_dim]``. The
            final axis concatenates bigram heads before trigram heads.
        """
        ngram_ids = self._hash_input_ids(input_ids, sequence_ids)
        return self._lookup_ngram_ids(ngram_ids)

    def _lookup_ngram_ids(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Look up precomputed packed n-gram table rows.

        Args:
            ngram_ids: Global table row IDs of shape ``[batch, sequence,
                ngram_heads]``.

        Returns:
            Concatenated head values of shape ``[batch, sequence,
            ngram_heads * head_dim]``.
        """
        head_embeddings = self.ngram_embedding(ngram_ids)
        expected_prefix = ngram_ids.shape
        if head_embeddings.ndim != 4 or head_embeddings.shape[:-1] != expected_prefix:
            raise RuntimeError(
                "The Engram lookup must map [batch, sequence, ngram_heads] IDs to "
                "[batch, sequence, ngram_heads, head_dim] values; got "
                f"{tuple(head_embeddings.shape)}"
            )
        return head_embeddings.flatten(start_dim=-2)

    def _forward_global_slice(
        self,
        global_input_ids: torch.Tensor,
        *,
        sequence_start: int,
        sequence_end: int,
        global_sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hash a complete raw sequence, then look up only one local slice.

        Args:
            global_input_ids: Replicated raw IDs of shape ``[batch,
                global_sequence]``. Hashing the full tensor preserves the two
                preceding raw tokens and EOS resets at a CP boundary.
            sequence_start: Inclusive global position of the requested shard.
            sequence_end: Exclusive global position of the requested shard.
            global_sequence_ids: Optional replicated packed segment IDs ``[batch, global_sequence]``.

        Returns:
            Local PLE values of shape ``[batch, local_sequence,
            ngram_heads * head_dim]`` where ``local_sequence`` equals
            ``sequence_end - sequence_start``.
        """
        if sequence_start < 0 or sequence_end < sequence_start or sequence_end > global_input_ids.shape[1]:
            raise ValueError(
                "Invalid Qwen3.8-Flash-Next PLE global slice "
                f"[{sequence_start}, {sequence_end}) for sequence length {global_input_ids.shape[1]}"
            )
        global_ngram_ids = self._hash_input_ids(global_input_ids, global_sequence_ids)
        return self._lookup_ngram_ids(global_ngram_ids[:, sequence_start:sequence_end])


class Qwen3_8_FlashNextPLELayer(nn.Module):
    """Contextualize Qwen3.8-Flash-Next n-gram values and return an HC-sized delta.

    Args:
        ple_embedding: Raw-token n-gram embedding whose output has shape
            ``[batch, sequence, ple_embed_dim]``.
        hidden_size: Width of one HyperConnection branch.
        hc_count: Number of persistent HyperConnection branches.
        ple_embed_dim: Concatenated n-gram embedding width.
        backend: Backend configuration for the key and value projections.
        dtype: Explicit parameter dtype resolved from the model configuration.
        conv_kernel_size: Kernel width of the causal depthwise convolution.
        rms_norm_eps: Variance epsilon for branch-local Gemma RMS norms.
    """

    def __init__(
        self,
        ple_embedding: Qwen3_8_FlashNextNGramEmbedding,
        *,
        hidden_size: int,
        hc_count: int,
        ple_embed_dim: int,
        backend: BackendConfig,
        dtype: torch.dtype | str,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if hc_count <= 1:
            raise ValueError(f"hc_count must be greater than one, got {hc_count}")
        if ple_embed_dim <= 0:
            raise ValueError(f"ple_embed_dim must be positive, got {ple_embed_dim}")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {conv_kernel_size}")

        self.ple_embedding = ple_embedding
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_hidden_size = hidden_size * hc_count
        self.ple_embed_dim = ple_embed_dim
        self.conv_kernel_size = conv_kernel_size
        self.short_conv_dilation = ple_embedding.ngram_size
        parameter_dtype = get_dtype(dtype, torch.bfloat16)
        self.key_proj = initialize_linear_module(
            backend.linear,
            ple_embed_dim,
            self.hc_hidden_size,
            bias=False,
            dtype=parameter_dtype,
        )
        self.value_proj = initialize_linear_module(
            backend.linear,
            ple_embed_dim,
            hidden_size,
            bias=False,
            dtype=parameter_dtype,
        )
        self.norm_key = Qwen3_8_FlashNextGroupedRMSNorm(
            self.hc_hidden_size,
            group_size=hidden_size,
            eps=rms_norm_eps,
        )
        self.norm_query = Qwen3_8_FlashNextGroupedRMSNorm(
            self.hc_hidden_size,
            group_size=hidden_size,
            eps=rms_norm_eps,
        )
        self.norm_conv = Qwen3_8_FlashNextGroupedRMSNorm(
            self.hc_hidden_size,
            group_size=hidden_size,
            eps=rms_norm_eps,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.hc_hidden_size,
            out_channels=self.hc_hidden_size,
            kernel_size=conv_kernel_size,
            dilation=self.short_conv_dilation,
            groups=self.hc_hidden_size,
            bias=False,
            dtype=parameter_dtype,
        )
        # Checkpoint-free initialization is zero-start. The trained checkpoint's
        # nonzero convolution is loaded over this tensor by the state adapter.
        nn.init.zeros_(self.conv1d.weight)

    def _apply_branch_norm(
        self,
        norm: Qwen3_8_FlashNextGroupedRMSNorm,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a flattened grouped norm while retaining explicit HC branches.

        Args:
            norm: Grouped normalization module with a learned weight of shape
                ``[hc_count * hidden_size]``.
            hidden_states: Tensor of shape
                ``[batch, sequence, hc_count, hidden_size]``.

        Returns:
            Branch-normalized tensor of shape
            ``[batch, sequence, hc_count, hidden_size]``.
        """
        normalized = norm(hidden_states.flatten(start_dim=-2))
        return normalized.unflatten(-1, (self.hc_count, self.hidden_size))

    def _causal_short_conv(
        self,
        hidden_states: torch.Tensor,
        cp_context: Qwen3_8_FlashNextCPContext | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the PLE causal depthwise convolution with left zero history.

        Args:
            hidden_states: Branch-flattened tensor of shape
                ``[batch, sequence, hc_count * hidden_size]``.
            cp_context: Optional contiguous CP metadata. Under CP, ``sequence``
                is local and the method exchanges only the preceding nine-token
                boundary required by the released dilation/kernel settings.
            sequence_ids: Optional packed segment IDs of shape ``[batch,
                local_sequence]`` without CP, or replicated ``[batch,
                global_sequence]`` under CP. Each segment then starts with zero
                history, as at the start of a row.

        Returns:
            Tensor of shape ``[batch, sequence, hc_count * hidden_size]``.
        """
        left_padding = (self.conv_kernel_size - 1) * self.short_conv_dilation
        if cp_context is None:
            channels_first = F.pad(hidden_states.transpose(1, 2), (left_padding, 0))
        else:
            left_halo = qwen3_8_flash_next_cp_left_halo(hidden_states, cp_context, history=left_padding)
            channels_first = torch.cat((left_halo, hidden_states), dim=1).transpose(1, 2)
        if sequence_ids is not None:
            return F.silu(self._segmented_conv(channels_first.transpose(1, 2), cp_context, sequence_ids))
        convolved = F.conv1d(
            channels_first,
            self.conv1d.weight,
            bias=None,
            dilation=self.short_conv_dilation,
            groups=self.hc_hidden_size,
        )
        return F.silu(convolved.transpose(1, 2))

    def _segmented_conv(
        self,
        padded: torch.Tensor,
        cp_context: Qwen3_8_FlashNextCPContext | None,
        sequence_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the convolution over every packed segment as if it started a fresh row.

        Each run of one segment in ``padded`` gets ``history`` zeros ahead of it and
        its own ``F.conv1d``, so every tap reads its own segment or zeros.

        Args:
            padded: Left history followed by the local sequence, ``[batch, history + sequence, channels]``.
            cp_context: Optional contiguous CP metadata.
            sequence_ids: Packed segment IDs ``[batch, sequence]``, or ``[batch, global_sequence]`` under CP.

        Returns:
            Pre-activation output ``[batch, sequence, channels]``.
        """
        history = (self.conv_kernel_size - 1) * self.short_conv_dilation
        length = padded.shape[1]
        start = 0 if cp_context is None else cp_context.local_sequence_start
        positions = torch.arange(start - history, start + length - history, device=padded.device)
        # Segment of every row; -1 marks history before the global row start.
        segment = torch.where(positions >= 0, sequence_ids[:, positions.clamp_min(0)], -1)
        rows = []
        for row, row_segment in zip(padded, segment):
            _, run_lengths = torch.unique_consecutive(row_segment, return_counts=True)
            runs = [
                F.conv1d(
                    F.pad(run.t(), (history, 0)),
                    self.conv1d.weight,
                    bias=None,
                    dilation=self.short_conv_dilation,
                    groups=self.hc_hidden_size,
                ).t()
                for run in row.split(run_lengths.tolist())
            ]
            # The outputs for the left history are dropped.
            rows.append(torch.cat(runs)[history:])
        # A packed micro-batch has one row; skip the copy that stacking would make.
        return rows[0].unsqueeze(0) if len(rows) == 1 else torch.stack(rows)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        cp_context: Qwen3_8_FlashNextCPContext | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the PLE delta injected before the layer's attention HC read.

        Args:
            hidden_states: True HyperConnection state of shape
                ``[batch, sequence, hc_count * hidden_size]``.
            input_ids: Raw integer tokenizer IDs of shape ``[batch, sequence]``.
            cp_context: Optional contiguous CP metadata. Its replicated
                ``global_input_ids`` and ``global_padding_mask`` fields have
                shape ``[batch, global_sequence]``; ``hidden_states`` and
                ``input_ids`` remain local ``[batch, sequence, ...]`` tensors.
            sequence_ids: Optional packed segment IDs ``[batch, sequence]``, or
                ``[batch, global_sequence]`` under CP: one ID per packed sample, then one
                each for the pack's tail padding and the CP padding, if present. The hash
                and the convolution then never read across segments; ``None`` keeps the
                unpacked path.

        Returns:
            PLE delta of shape
            ``[batch, sequence, hc_count * hidden_size]``. The output does not
            alias or mutate ``hidden_states``.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hc_count * hidden_size], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.hc_hidden_size:
            raise ValueError(
                "PLE requires the full HyperConnection state: expected final width "
                f"{self.hc_hidden_size}, got {hidden_states.shape[-1]}"
            )
        if input_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "input_ids and hidden_states must share [batch, sequence] axes, got "
                f"{tuple(input_ids.shape)} and {tuple(hidden_states.shape)}"
            )

        if cp_context is None:
            embeddings = self.ple_embedding(input_ids, sequence_ids)
        else:
            if hidden_states.shape[1] != cp_context.local_sequence_length:
                raise ValueError(
                    "PLE local sequence length disagrees with its CP context; "
                    f"got local={hidden_states.shape[1]}, context={cp_context.local_sequence_length}"
                )
            embeddings = self.ple_embedding._forward_global_slice(
                cp_context.global_input_ids,
                sequence_start=cp_context.local_sequence_start,
                sequence_end=cp_context.local_sequence_end,
                global_sequence_ids=sequence_ids,
            )
        if embeddings.shape != (*hidden_states.shape[:2], self.ple_embed_dim):
            raise RuntimeError(
                f"Expected PLE embeddings with shape {(*hidden_states.shape[:2], self.ple_embed_dim)}, "
                f"got {tuple(embeddings.shape)}"
            )
        # The model-owned table is excluded from FSDP and can retain fp32
        # master-weight storage while the surrounding block computes in bf16.
        # Match the block activation dtype at the projection boundary; autograd
        # casts the table gradient back to the parameter's storage dtype.
        embeddings = embeddings.to(dtype=hidden_states.dtype)
        key = self.key_proj(embeddings).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query = hidden_states.unflatten(-1, (self.hc_count, self.hidden_size))
        normalized_key = self._apply_branch_norm(self.norm_key, key)
        normalized_query = self._apply_branch_norm(self.norm_query, query)
        gate = (normalized_key * normalized_query).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)

        gated_value = gate * value.unsqueeze(-2)
        normalized_gated_value = self._apply_branch_norm(self.norm_conv, gated_value)
        gated_value = gated_value.flatten(start_dim=-2)
        normalized_gated_value = normalized_gated_value.flatten(start_dim=-2)
        return gated_value + self._causal_short_conv(normalized_gated_value, cp_context, sequence_ids)

    @torch.no_grad()
    def init_weights(self, initializer_range: float = 0.02) -> None:
        """Initialize PLE projections and the zero-start causal convolution.

        Args:
            initializer_range: Standard deviation for projection weights.
        """
        if initializer_range < 0:
            raise ValueError(f"initializer_range must be non-negative, got {initializer_range}")
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=initializer_range)
        self.norm_key.reset_parameters()
        self.norm_query.reset_parameters()
        self.norm_conv.reset_parameters()
        nn.init.zeros_(self.conv1d.weight)
