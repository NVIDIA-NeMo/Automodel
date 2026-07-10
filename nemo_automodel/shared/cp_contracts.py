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

"""Typed contracts shared by context-parallel models and dispatchers."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.distributed.device_mesh import DeviceMesh

CPBatch = dict[str, object]
ContextFactory = Callable[[], AbstractContextManager[object]]


class CPPreparedInputs(TypedDict, total=False):
    """Values a model CP-preparation hook may merge into the batch.

    Tensor values use batch-first sequence layout unless documented by the
    owning model: token IDs and masks are ``[B, S]``; embeddings are
    ``[B, S, H]``; mRoPE positions may be ``[R, B, S]``. Here ``B`` is batch,
    ``S`` is the full pre-shard sequence, ``H`` is hidden size, and ``R`` is
    the rotary-position axis count. A ``None`` value marks a consumed raw input
    that the dispatcher must remove.
    """

    cp_sharder: CPSharder
    input_ids: torch.Tensor | None
    inputs_embeds: torch.Tensor | None
    position_ids: torch.Tensor | None
    pixel_values: torch.Tensor | None
    pixel_values_videos: torch.Tensor | None
    patch_pixel_values: torch.Tensor | None
    image_embeds: torch.Tensor | None
    image_flags: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    image_grid_hws: torch.Tensor | None
    image_position_ids: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    imgs_sizes: torch.Tensor | None
    sound_features: torch.Tensor | None
    sound_attention_mask: torch.Tensor | None
    mm_token_type_ids: torch.Tensor | None
    num_patches: torch.Tensor | list[int] | tuple[int, ...] | None
    per_layer_inputs: torch.Tensor | None
    _gemma4_vision_group_ids: torch.Tensor | None


@runtime_checkable
class CPPrepareModel(Protocol):
    """Model capability used to opt into the CP pre-forward hook."""

    def prepare_model_inputs_for_cp(self, input_ids: torch.Tensor) -> CPPreparedInputs:
        """Prepare full token IDs ``[B, S]`` before CP sequence sharding.

        ``B`` is batch size and ``S`` is sequence length. Model implementations
        may add explicit optional media arguments while preserving this common
        one-tensor call contract.
        """


class ShardBatch(Protocol):
    """Callable that pads and shards one full-sequence batch."""

    def __call__(
        self,
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
        batch: CPBatch,
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
    ) -> tuple[ContextFactory, CPBatch]:
        """Pad and shard a full token batch.

        Args:
            cp_mesh: Context-parallel mesh governing sequence sharding.
            tp_mesh: Optional tensor-parallel mesh.
            batch: Full token mapping with IDs/masks ``[B, S]`` or embeddings
                ``[B, S, H]``, where ``B`` is batch size, ``S`` is global
                sequence length, and ``H`` is hidden size. Implementations may
                mutate this mapping into local layout.
            loss_mask: Optional token mask ``[B, S]``.
            padding_token_id: Fill value for padded token IDs.

        Returns:
            Context factory and prepared mapping. Backend-owned sharders return
            local sequence ``S_local``; torch CP produces it on context entry.
        """


class TokenIndexMap(Protocol):
    """Callable resolving the global positions owned by one CP rank."""

    def __call__(
        self,
        cp_mesh: DeviceMesh | None,
        padded_seq_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Resolve a rank's global token positions.

        Args:
            cp_mesh: Context-parallel mesh, or ``None`` when inactive.
            padded_seq_len: Global padded sequence length ``S``.
            device: Device for the result.

        Returns:
            Int64 positions ``[S_local]`` in local token order.
        """


@dataclass(frozen=True)
class CPBatchWithIndices:
    """A prepared batch and its local-to-global token index map.

    Attributes:
        batch: Prepared batch. Token-aligned tensors have local sequence
            ``S_local`` when CP is active and global sequence ``S`` otherwise.
        local_indices: Global token positions ``[S_local]`` in local tensor
            order, or ``None`` when the backend cannot expose one token space.
            Here ``S`` is global sequence length and ``S_local`` is per-rank
            local sequence length.
    """

    batch: CPBatch
    local_indices: torch.Tensor | None


@dataclass(frozen=True)
class CPForwardResult:
    """Named result of unified context-parallel forward preparation.

    Attributes:
        context_factory: Callable producing the context manager that must wrap
            the model forward.
        batch: Prepared batch. Token tensors are full before entering a native
            torch CP context and local for backend-owned sharding.
        sharder: Resolved backend contract for identically sharding or gathering
            auxiliary token-aligned tensors.
    """

    context_factory: ContextFactory
    batch: CPBatch
    sharder: CPSharder


def _cp_rank(cp_mesh: DeviceMesh) -> int:
    """Resolve this rank's index within ``cp_mesh``."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=cp_mesh.get_group())
    return cp_mesh.get_local_rank()


def contiguous_local_indices(
    cp_mesh: DeviceMesh | None,
    padded_seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return global positions owned by a contiguous CP rank.

    Args:
        cp_mesh: Context-parallel mesh. It must be present for this layout.
        padded_seq_len: Global padded sequence length ``S``; divisible by CP.
        device: Device for the returned index tensor.

    Returns:
        Int64 positions ``[S / CP]`` in local sequence order.
    """
    if cp_mesh is None:
        raise ValueError("contiguous token indices require a context-parallel mesh")
    cp_size = cp_mesh.size()
    if padded_seq_len % cp_size != 0:
        raise ValueError(f"padded_seq_len must be divisible by cp_size, got {padded_seq_len=} {cp_size=}")
    local_len = padded_seq_len // cp_size
    start = _cp_rank(cp_mesh) * local_len
    return torch.arange(start, start + local_len, device=device, dtype=torch.long)


def identity_local_indices(
    cp_mesh: DeviceMesh | None,
    padded_seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return all global positions when CP is inactive.

    Args:
        cp_mesh: Inactive or singleton CP mesh; ignored.
        padded_seq_len: Full sequence length ``S``.
        device: Device for the returned index tensor.

    Returns:
        Int64 identity positions ``[S]``.
    """
    del cp_mesh
    return torch.arange(padded_seq_len, device=device, dtype=torch.long)


def round_robin_local_indices(
    cp_mesh: DeviceMesh | None,
    padded_seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return positions for torch CP head-tail load balancing.

    Args:
        cp_mesh: Active context-parallel mesh.
        padded_seq_len: Global padded sequence ``S``; divisible by ``2 * CP``.
        device: Device for the returned index tensor.

    Returns:
        Int64 positions ``[S / CP]`` in local head-then-tail order.
    """
    if cp_mesh is None:
        raise ValueError("round-robin token indices require a context-parallel mesh")
    cp_size = cp_mesh.size()
    if padded_seq_len % (2 * cp_size) != 0:
        raise ValueError(f"padded_seq_len must be divisible by 2 * cp_size, got {padded_seq_len=} {cp_size=}")
    chunk_len = padded_seq_len // (2 * cp_size)
    rank = _cp_rank(cp_mesh)
    head = torch.arange(rank * chunk_len, (rank + 1) * chunk_len, device=device, dtype=torch.long)
    tail_start = (2 * cp_size - 1 - rank) * chunk_len
    tail = torch.arange(tail_start, tail_start + chunk_len, device=device, dtype=torch.long)
    return torch.cat((head, tail))


def captured_token_indices(local_indices: torch.Tensor) -> TokenIndexMap:
    """Capture a data-dependent local token layout.

    Args:
        local_indices: Int-compatible global positions ``[S_local]``. The
            returned closure owns a flattened int64 copy-like conversion.

    Returns:
        Index resolver validating ``S_global == S_local * CP``.
    """
    local_indices = local_indices.reshape(-1).to(torch.long)

    def _indices(
        cp_mesh: DeviceMesh | None,
        padded_seq_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        cp_size = cp_mesh.size() if cp_mesh is not None else 1
        expected = local_indices.numel() * cp_size
        if padded_seq_len != expected:
            raise ValueError(
                f"captured CP indices cover a padded stream of {expected} tokens "
                f"({local_indices.numel()} local x cp_size {cp_size}), got {padded_seq_len=}. "
                "The tensor does not match the batch this sharder last sharded."
            )
        return local_indices if device is None else local_indices.to(device)

    return _indices


def shard_token_tensor_by_indices(
    tensor: torch.Tensor,
    local_indices: torch.Tensor,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Select one CP rank's positions from a full token tensor.

    Args:
        tensor: Full padded token tensor with arbitrary leading/trailing axes
            and global sequence axis ``S`` at ``seq_dim``.
        local_indices: Int64 global positions ``[S_local]``.
        seq_dim: Sequence axis of ``tensor``.

    Returns:
        Contiguous local tensor with ``S_local`` at ``seq_dim``. It does not
        alias ``tensor``.
    """
    return tensor.index_select(seq_dim, local_indices.to(tensor.device)).contiguous()


def _reorder_gathered_token_tensor(
    parts: list[torch.Tensor],
    index_parts: list[torch.Tensor],
    seq_dim: int = 1,
) -> torch.Tensor:
    """Reassemble rank-ordered local token tensors into global order.

    Args:
        parts: Local tensors with equal ``S_local`` at ``seq_dim``.
        index_parts: Corresponding int64 maps ``[S_local]``.
        seq_dim: Sequence axis shared by every local tensor.

    Returns:
        Contiguous global tensor with ``S_global`` at ``seq_dim``.
    """
    full = torch.cat(parts, dim=seq_dim)
    full_indices = torch.cat([part.to(full.device) for part in index_parts])
    return full.index_select(seq_dim, torch.argsort(full_indices)).contiguous()


def gather_token_tensor_by_indices(
    cp_mesh: DeviceMesh | None,
    tensor: torch.Tensor,
    local_indices: torch.Tensor,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Differentiably gather a local token tensor on every CP rank.

    The output is replicated. Autograd therefore sums backward contributions
    from every rank; when every rank evaluates the same scalar loss from this
    output, divide that loss by CP size to match a single global evaluation.

    Args:
        cp_mesh: Context-parallel mesh, or ``None`` when inactive.
        tensor: Local tensor with ``S_local`` at ``seq_dim``.
        local_indices: Int64 global positions ``[S_local]``.
        seq_dim: Sequence axis of ``tensor``.

    Returns:
        Replicated contiguous tensor with ``S_global`` at ``seq_dim``. It is
        identical to ``tensor`` when CP is inactive.
    """
    if cp_mesh is None or cp_mesh.size() <= 1 or not (dist.is_available() and dist.is_initialized()):
        return tensor

    group = cp_mesh.get_group()
    parts = list(dist_nn.all_gather(tensor.contiguous(), group=group))
    local_indices = local_indices.to(tensor.device)
    index_parts = [torch.empty_like(local_indices) for _ in range(cp_mesh.size())]
    dist.all_gather(index_parts, local_indices.contiguous(), group=group)
    return _reorder_gathered_token_tensor(parts, index_parts, seq_dim=seq_dim)


def shard_batch_identity(
    cp_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None,
    batch: CPBatch,
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
) -> tuple[ContextFactory, CPBatch]:
    """Return an inactive CP batch without mutation.

    Args:
        cp_mesh: Inactive or singleton CP mesh; ignored.
        tp_mesh: Tensor-parallel mesh; ignored.
        batch: Batch containing full token tensors such as IDs ``[B, S]`` or
            embeddings ``[B, S, H]``. It is returned by identity.
        loss_mask: Optional full mask ``[B, S]``; ignored.
        padding_token_id: Token padding sentinel; ignored.

    Returns:
        ``(nullcontext, batch)`` with the original batch object.
    """
    del cp_mesh, tp_mesh, loss_mask, padding_token_id
    return contextlib.nullcontext, batch


@dataclass
class CPSharder:
    """Context-parallel backend contract.

    Attributes:
        shard_batch: Typed callable that pads and shards a full batch.
        local_token_global_indices: Resolver returning global positions
            ``[S_local]`` in local order, or ``None`` before a data-dependent
            sharder has processed its batch.
        layout: Diagnostic label; framework behavior must not branch on it.
    """

    shard_batch: ShardBatch
    local_token_global_indices: TokenIndexMap | None
    layout: str = "custom"

    def _indices(
        self,
        cp_mesh: DeviceMesh | None,
        padded_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve local positions for one global token stream.

        Args:
            cp_mesh: Context-parallel mesh, or ``None`` when inactive.
            padded_seq_len: Global padded sequence length ``S``.
            device: Device for the result.

        Returns:
            Int64 global positions ``[S_local]`` in local tensor order.
        """
        if self.local_token_global_indices is None:
            raise NotImplementedError(
                f"CPSharder(layout={self.layout!r}) has a data-dependent token layout; its index map is "
                "captured during shard_batch — token-tensor shard/gather are unavailable before the first shard."
            )
        return self.local_token_global_indices(cp_mesh, padded_seq_len, device)

    def shard_token_tensor(
        self,
        cp_mesh: DeviceMesh | None,
        tensor: torch.Tensor,
        seq_dim: int = 1,
    ) -> torch.Tensor:
        """Shard a full padded token tensor like the model input.

        Args:
            cp_mesh: Context-parallel mesh, or ``None`` when inactive.
            tensor: Full tensor with global sequence ``S`` at ``seq_dim``.
            seq_dim: Sequence axis.

        Returns:
            Contiguous tensor with local sequence ``S_local`` at ``seq_dim``.
        """
        indices = self._indices(cp_mesh, tensor.shape[seq_dim], tensor.device)
        return shard_token_tensor_by_indices(tensor, indices, seq_dim=seq_dim)

    def gather_token_tensor(
        self,
        cp_mesh: DeviceMesh | None,
        tensor: torch.Tensor,
        seq_dim: int = 1,
    ) -> torch.Tensor:
        """Differentiably gather a local token tensor on every CP rank.

        Args:
            cp_mesh: Context-parallel mesh, or ``None`` when inactive.
            tensor: Local tensor with ``S_local`` at ``seq_dim``.
            seq_dim: Sequence axis.

        Returns:
            Replicated global-order tensor with ``S_global`` at ``seq_dim``.
            Backward sums contributions from every replicated consumer.
        """
        cp_size = cp_mesh.size() if cp_mesh is not None else 1
        indices = self._indices(cp_mesh, tensor.shape[seq_dim] * cp_size, tensor.device)
        return gather_token_tensor_by_indices(cp_mesh, tensor, indices, seq_dim=seq_dim)
