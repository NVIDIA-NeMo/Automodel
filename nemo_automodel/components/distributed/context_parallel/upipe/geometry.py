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

"""Head-chunk geometry for UPipe (Untied Ulysses) context parallelism.

UPipe walks attention one head-chunk at a time. With ring degree 1 the Ulysses degree
equals the CP size ``P``, so each stage projects ``P`` query heads locally, all-to-alls
them into head-sharded layout, and every rank ends up computing exactly one query head
over the full sequence.

The non-obvious part is head assignment under GQA. Stage ``s`` reads KV chunk
``s // gqa_ratio``, so after the all-to-all rank ``r`` owns effective KV head
``(s // gqa_ratio) * P + r``. The query head handed to that rank must belong to that
same KV group, which a naive contiguous split of the query weight does *not* give. This
module derives the permutation that keeps the pairing correct:

    kv_head(s, r)    = (s // gqa_ratio) * P + r
    query_head(s, r) = kv_head(s, r) * gqa_ratio + (s % gqa_ratio)

so that ``query_head(s, r) // gqa_ratio == kv_head(s, r)`` for every stage and rank.

When ``num_kv_heads < P`` the KV weights are replicated (repeat-interleaved) up to ``P``
so the all-to-all stays evenly divisible; effective KV head ``e`` then reads source KV
head ``e // kv_replication``. Because replicas index the *same* weight rows, autograd
accumulates their gradients without an explicit reduction step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UPipeHeadGeometry:
    """Per-layer head-chunk schedule for a single CP group size.

    Attributes:
        cp_size: Ulysses degree, equal to the CP group size (ring degree is 1).
        num_heads: Query head count for this layer.
        num_kv_heads: Key/value head count for this layer, before replication.
        head_dim: Per-head dimension.
        kv_replication: Repeat factor applied to KV heads when ``num_kv_heads < cp_size``.
        num_kv_heads_effective: ``num_kv_heads * kv_replication``.
        pipe_degree: Number of head-chunk stages, ``num_heads // cp_size``.
        gqa_ratio: Query heads per effective KV head.
        num_kv_chunks: Number of distinct KV chunks, ``num_kv_heads_effective // cp_size``.
    """

    cp_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    kv_replication: int
    num_kv_heads_effective: int
    pipe_degree: int
    gqa_ratio: int
    num_kv_chunks: int

    @classmethod
    def build(cls, *, cp_size: int, num_heads: int, num_kv_heads: int, head_dim: int) -> "UPipeHeadGeometry":
        """Derive the schedule, validating every divisibility requirement.

        Args:
            cp_size: CP group size.
            num_heads: Query head count for this layer.
            num_kv_heads: Key/value head count for this layer.
            head_dim: Per-head dimension.

        Returns:
            UPipeHeadGeometry: The validated schedule.

        Raises:
            ValueError: If the head counts cannot be scheduled at this CP size.
        """
        if cp_size < 1:
            raise ValueError(f"cp_size must be positive, got {cp_size}")
        if num_heads % cp_size != 0:
            raise ValueError(
                f"UPipe CP requires num_heads ({num_heads}) to be divisible by cp_size ({cp_size}); "
                "choose a CP size that divides the query head count."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        if num_kv_heads >= cp_size:
            if num_kv_heads % cp_size != 0:
                raise ValueError(
                    f"UPipe CP requires num_kv_heads ({num_kv_heads}) to be divisible by cp_size ({cp_size})"
                )
            kv_replication = 1
        else:
            if cp_size % num_kv_heads != 0:
                raise ValueError(
                    f"UPipe CP requires cp_size ({cp_size}) to be divisible by num_kv_heads ({num_kv_heads}) "
                    "when the KV heads must be replicated"
                )
            kv_replication = cp_size // num_kv_heads

        num_kv_heads_effective = num_kv_heads * kv_replication
        return cls(
            cp_size=cp_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_replication=kv_replication,
            num_kv_heads_effective=num_kv_heads_effective,
            pipe_degree=num_heads // cp_size,
            gqa_ratio=num_heads // num_kv_heads_effective,
            num_kv_chunks=num_kv_heads_effective // cp_size,
        )

    def kv_chunk(self, stage: int) -> int:
        """Index of the KV chunk consumed by ``stage``."""
        return stage // self.gqa_ratio

    def recomputes_kv(self, stage: int) -> bool:
        """Whether ``stage`` starts a new KV chunk (and so must project K and V)."""
        return stage % self.gqa_ratio == 0

    def kv_head(self, stage: int, rank: int) -> int:
        """Effective KV head owned by ``rank`` at ``stage``."""
        return self.kv_chunk(stage) * self.cp_size + rank

    def source_kv_head(self, stage: int, rank: int) -> int:
        """Pre-replication KV head backing :meth:`kv_head`."""
        return self.kv_head(stage, rank) // self.kv_replication

    def query_head(self, stage: int, rank: int) -> int:
        """Query head owned by ``rank`` at ``stage``."""
        return self.kv_head(stage, rank) * self.gqa_ratio + (stage % self.gqa_ratio)

    def stage_query_heads(self, stage: int, device: torch.device | None = None) -> torch.Tensor:
        """Query head indices for ``stage``, ordered by destination rank.

        Rank ``r`` receives element ``r`` after the sequence-to-head all-to-all, so this
        ordering is what the query weight must be gathered in.
        """
        return torch.tensor(
            [self.query_head(stage, r) for r in range(self.cp_size)],
            dtype=torch.long,
            device=device,
        )

    def stage_kv_source_heads(self, stage: int, device: torch.device | None = None) -> torch.Tensor:
        """Pre-replication KV head indices for ``stage``, ordered by destination rank."""
        return torch.tensor(
            [self.source_kv_head(stage, r) for r in range(self.cp_size)],
            dtype=torch.long,
            device=device,
        )

    def head_order(self, device: torch.device | None = None) -> torch.Tensor:
        """Query heads in ``(stage, rank)`` emission order, shape ``[num_heads]``."""
        return torch.tensor(
            [self.query_head(s, r) for s in range(self.pipe_degree) for r in range(self.cp_size)],
            dtype=torch.long,
            device=device,
        )

    def inverse_head_order(self, device: torch.device | None = None) -> torch.Tensor:
        """Permutation restoring natural head order from emission order.

        Concatenating per-stage outputs yields heads in :meth:`head_order`; selecting with
        this index puts them back in the layout ``o_proj`` expects.
        """
        return torch.argsort(self.head_order(device=device))

    def validate(self) -> None:
        """Assert the query/KV pairing is internally consistent.

        Guards the derivation above: every ``(stage, rank)`` pair must map a query head to
        the KV group its co-resident KV head belongs to, and the stages together must cover
        each query head exactly once.
        """
        seen: set[int] = set()
        for stage in range(self.pipe_degree):
            for rank in range(self.cp_size):
                query = self.query_head(stage, rank)
                kv = self.kv_head(stage, rank)
                if query // self.gqa_ratio != kv:
                    raise AssertionError(
                        f"UPipe head pairing broken at stage={stage} rank={rank}: "
                        f"query head {query} maps to KV group {query // self.gqa_ratio}, expected {kv}"
                    )
                if query >= self.num_heads:
                    raise AssertionError(f"query head {query} out of range for {self.num_heads} heads")
                seen.add(query)
        if len(seen) != self.num_heads:
            raise AssertionError(f"UPipe head schedule covers {len(seen)} of {self.num_heads} query heads")


def geometry_for_attention(module, cp_size: int) -> UPipeHeadGeometry:
    """Build the schedule for a HuggingFace ``InklingAttention``-shaped module.

    Sliding and global Inkling layers carry different KV head counts, so the schedule is
    always derived from the module's own attributes rather than from a single config-level
    value.
    """
    geometry = UPipeHeadGeometry.build(
        cp_size=cp_size,
        num_heads=module.num_heads,
        num_kv_heads=module.num_key_value_heads,
        head_dim=module.head_dim,
    )
    geometry.validate()
    return geometry
